
# -----------------------------------------------------------------------------
# workflows_powder.jl
#
# Powder "setup ctx" vs "evaluate model" workflow, Makie-free.
# Designed for:
#   - Do expensive setup once (instrument/pixels, axes, TOF, resolution, CDF-work,
#     Cv(flat), reduce-map)
#   - Then iterate many models: only Cs + reduce
#
# Uses grouping/masking via apply_grouping_masking (Mantid-style analogs).
# -----------------------------------------------------------------------------

# -----------------------------
# Reduce-map for fast powder reduction
# -----------------------------
struct ReduceMapPowder
    iQ::Matrix{Int32}
    iW::Matrix{Int32}
    fQ::Matrix{Float32}
    fW::Matrix{Float32}
    valid::BitMatrix
end

"""
    build_reduce_map_powder(inst, pixels, Ei_meV, tof_edges_s, Q_edges, ω_edges)

Precompute mapping from (pixel, tof-bin-center) -> (|Q|, ω) bin indices plus
bilinear weights. This makes repeated reductions much faster.
"""
function build_reduce_map_powder(inst::Instrument, pixels::Vector{DetectorPixel},
        Ei_meV::Float64, tof_edges_s::Vector{Float64},
        Q_edges::Vector{Float64}, ω_edges::Vector{Float64})

    nP = length(pixels)
    nT = length(tof_edges_s) - 1
    nQ = length(Q_edges) - 1
    nW = length(ω_edges) - 1

    iQ = fill(Int32(0), nP, nT)
    iW = fill(Int32(0), nP, nT)
    fQ = fill(Float32(0), nP, nT)
    fW = fill(Float32(0), nP, nT)
    valid = falses(nP, nT)

    @inbounds for ip in 1:nP
        p = pixels[ip]
        L2p = L2(inst, p.id)
        for it in 1:nT
            t = 0.5 * (tof_edges_s[it] + tof_edges_s[it+1])

            Ef = try
                Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Qvec, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            Qmag = norm(Qvec)

            iq = searchsortedlast(Q_edges, Qmag)
            iw = searchsortedlast(ω_edges, ω)
            if iq < 1 || iq >= nQ || iw < 1 || iw >= nW
                continue
            end

            dq = Q_edges[iq+1] - Q_edges[iq]
            dw = ω_edges[iw+1] - ω_edges[iw]
            (dq <= 0 || dw <= 0) && continue

            fq = Float32((Qmag - Q_edges[iq]) / dq)
            fw = Float32((ω    - ω_edges[iw]) / dw)
            fq = min(max(fq, 0.0f0), 1.0f0)
            fw = min(max(fw, 0.0f0), 1.0f0)

            iQ[ip,it] = Int32(iq)
            iW[ip,it] = Int32(iw)
            fQ[ip,it] = fq
            fW[ip,it] = fw
            valid[ip,it] = true
        end
    end

    return ReduceMapPowder(iQ, iW, fQ, fW, valid)
end

"""
    reduce_three!(Hraw, Hsum, Hwt, pixels, rmap, Cs, Cv; eps=1e-12)

One-pass powder reduction with bilinear weights:
- Hraw gets raw SUM (uses Cs)
- Hsum gets SUM of vanadium-normalized values (Cs/Cv)
- Hwt gets SUM of weights
Then MEAN can be computed as Hsum./(Hwt+eps).
"""
function reduce_three!(Hraw::Hist2D, Hsum::Hist2D, Hwt::Hist2D,
        pixels::Vector{DetectorPixel}, rmap::ReduceMapPowder,
        Cs::AbstractMatrix, Cv::AbstractMatrix; eps=1e-12)

    nP = length(pixels)
    nT = size(rmap.valid, 2)

    @inbounds for ip in 1:nP
        pid = pixels[ip].id
        for it in 1:nT
            rmap.valid[ip,it] || continue
            cv = Cv[pid, it]
            cv <= 0.0 && continue

            cs = Cs[pid, it]
            iq = Int(rmap.iQ[ip,it])
            iw = Int(rmap.iW[ip,it])
            fq = Float64(rmap.fQ[ip,it])
            fw = Float64(rmap.fW[ip,it])

            w00 = (1 - fq) * (1 - fw)
            w10 = fq       * (1 - fw)
            w01 = (1 - fq) * fw
            w11 = fq       * fw

            Hraw.counts[iq,   iw]   += w00 * cs
            Hraw.counts[iq+1, iw]   += w10 * cs
            Hraw.counts[iq,   iw+1] += w01 * cs
            Hraw.counts[iq+1, iw+1] += w11 * cs

            v = cs / (cv + eps)
            Hsum.counts[iq,   iw]   += w00 * v
            Hsum.counts[iq+1, iw]   += w10 * v
            Hsum.counts[iq,   iw+1] += w01 * v
            Hsum.counts[iq+1, iw+1] += w11 * v

            Hwt.counts[iq,   iw]   += w00
            Hwt.counts[iq+1, iw]   += w10
            Hwt.counts[iq,   iw+1] += w01
            Hwt.counts[iq+1, iw+1] += w11
        end
    end
    return nothing
end

# -----------------------------
# Context
# -----------------------------
"""
    PowderCtx

Holds all model-independent setup products for powder forward prediction:
instrument, selected pixels (after grouping/masking/decimation), axes, TOF edges,
resolution object + optional CDF work, Cv(flat), and a precomputed reduce-map.
"""
struct PowderCtx{R,W}
    instr::Symbol
    idf_path::String
    inst::Instrument
    pixels::Vector{DetectorPixel}
    Ei::Float64
    tof_edges::Vector{Float64}
    Q_edges::Vector{Float64}
    ω_edges::Vector{Float64}
    resolution::R
    cdf_work::W
    Cv::Matrix{Float64}
    rmap::ReduceMapPowder
end

"""
    setup_powder_ctx(; ...)

Build a powder forward-prediction context.
This is the "setup" part to run once and reuse for many models.

You pass kernel domain bounds (Qmin/Qmax/wmin/wmax) rather than a kernel object.
This keeps the workflow code independent from any particular kernel source.
"""
function setup_powder_ctx(;
    instr::Symbol,
    idf_path::AbstractString,
    kern_Qmin::Float64,
    kern_Qmax::Float64,
    kern_wmin::Float64,
    kern_wmax::Float64,
    rebuild_geom::Bool=false,
    ψstride::Int=1,
    ηstride::Int=1,
    grouping::AbstractString="",
    grouping_file::Union{Nothing,AbstractString}=nothing,
    mask_btp::AbstractString="",
    mask_mode::Symbol=:drop,
    angle_step::Real=0.5,
    Ei::Float64,
    nQbins::Int=220,
    nωbins::Int=240,
    res_mode::AbstractString="cdf",
    σt_us::Float64=100.0,
    gh_order::Int=3,
    nsigma::Float64=6.0,
    ntof_env::AbstractString="auto",
    ntof_alpha::Float64=0.5,
    ntof_beta::Float64=1/3,
    ntof_min::Int=200,
    ntof_max::Int=2000,
    # caching (Serialization-based)
    disk_cache::Bool=true,
    cache_dir::AbstractString=joinpath(@__DIR__, "..", "scripts", ".toftwin_cache"),
    cache_ver::AbstractString="ctx_v1",
    cache_Cv::Bool=false,
    cache_rmap::Bool=false)

    # --- instrument ---
    out = if isdefined(MantidIDF, :load_mantid_idf_diskcached)
        MantidIDF.load_mantid_idf_diskcached(String(idf_path); rebuild=rebuild_geom)
    else
        @warn "load_mantid_idf_diskcached not found; falling back to load_mantid_idf (slower)."
        MantidIDF.load_mantid_idf(String(idf_path))
    end

    inst = out.inst
    bank0 = (hasproperty(out, :bank) && out.bank !== nothing) ? out.bank : DetectorBank(inst.name, out.pixels)

    # --- grouping/masking ---
    pixels_gm = if isdefined(@__MODULE__, :apply_grouping_masking)
        apply_grouping_masking(bank0.pixels;
            instrument=instr,
            grouping=grouping,
            grouping_file=grouping_file,
            mask_btp=mask_btp,
            mask_mode=mask_mode,
            outdir=joinpath(dirname(String(idf_path)), "out"),
            angle_step=angle_step,
            return_meta=false,
        )
    else
        @warn "apply_grouping_masking not found; proceeding without grouping/masking"
        bank0.pixels
    end
    bank = DetectorBank(inst.name, pixels_gm)

    # --- decimation ---
    pixels = sample_pixels(bank, AngularDecimate(ψstride, ηstride))

    # --- axes ---
    Q_edges = collect(range(kern_Qmin, kern_Qmax; length=nQbins+1))
    ωlo = min(-2.0, kern_wmin)
    ωhi = max(Ei,  kern_wmax)
    ω_edges = collect(range(ωlo, ωhi; length=nωbins+1))

    # --- TOF window ---
    L2min = minimum(inst.L2); L2max = maximum(inst.L2)
    Ef_min, Ef_max = 1.0, Ei
    tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ef_max)
    tmax = tof_from_EiEf(inst.L1, L2max, Ei, Ef_min)

    σt = σt_us * 1e-6

    ntof = let env = lowercase(String(ntof_env))
        n0 = (env in ("auto","suggest","0")) ? 0 : parse(Int, env)
        if n0 <= 0 && lowercase(res_mode) == "cdf" && σt_us > 0
            ntof_s, dt_s, maxd, dωbin = suggest_ntof(inst, pixels, Ei, tmin, tmax, ω_edges, σt;
                α=ntof_alpha, β=ntof_beta, ntof_min=ntof_min, ntof_max=ntof_max)
            @info "Auto-selected TOFTWIN_NTOF=$ntof_s (dt=$(round(1e6*dt_s,digits=2)) µs; max|dω/dt|=$(round(maxd,digits=4)) meV/s; Δωbin=$(round(dωbin,digits=4)) meV)"
            ntof_s
        elseif n0 <= 0
            500
        else
            n0
        end
    end

    tof_edges = collect(range(tmin, tmax; length=ntof+1))

    # --- resolution ---
    resolution =
        (lowercase(res_mode) == "none" || σt_us <= 0) ? NoResolution() :
        (lowercase(res_mode) == "gh")  ? GaussianTimingResolution(σt; order=gh_order) :
        (lowercase(res_mode) == "cdf") ? GaussianTimingCDFResolution(σt; nsigma=nsigma) :
        throw(ArgumentError("TOFTWIN_RES_MODE must be none|gh|cdf (got '$res_mode')"))

    # --- optional CDF work ---
    cdf_work = if (resolution isa GaussianTimingCDFResolution)
        if isdefined(@__MODULE__, :precompute_tof_smear_work)
            @info "Using in-session precomputed CDF smear work"
            precompute_tof_smear_work(inst, pixels, Ei, tof_edges, resolution)
        else
            @warn "TOFtwin.precompute_tof_smear_work not found; continuing without precomputed work"
            nothing
        end
    else
        nothing
    end

    # --- Cv(flat) ---
    compute_Cv_flat() = predict_pixel_tof(inst;
        pixels=pixels, Ei_meV=Ei, tof_edges_s=tof_edges,
        model=(Q,ω)->1.0, resolution=resolution,
        cdf_work=cdf_work
    )

    Cv = if disk_cache && cache_Cv
        io = IOBuffer()
        serialize(io, cache_ver)
        serialize(io, String(instr))
        serialize(io, String(idf_path))
        st = try stat(String(idf_path)) catch; nothing end
        serialize(io, st === nothing ? nothing : (st.mtime, st.size))
        serialize(io, grouping)
        serialize(io, grouping_file === nothing ? "" : String(grouping_file))
        serialize(io, mask_btp)
        serialize(io, mask_mode)
        serialize(io, Float64(angle_step))
        serialize(io, Int32[p.id for p in pixels])
        serialize(io, Ei)
        serialize(io, tof_edges)
        serialize(io, resolution)
        path = _wf_cache_path(cache_dir, "Cv_flat", take!(io))
        @info "Cv cache: $path"
        _wf_load_or_compute(path, compute_Cv_flat; disk_cache=disk_cache)
    else
        compute_Cv_flat()
    end

    # --- reduce map ---
    compute_rmap() = build_reduce_map_powder(inst, pixels, Ei, tof_edges, Q_edges, ω_edges)

    rmap = if disk_cache && cache_rmap
        io = IOBuffer()
        serialize(io, cache_ver)
        serialize(io, String(instr))
        serialize(io, String(idf_path))
        st = try stat(String(idf_path)) catch; nothing end
        serialize(io, st === nothing ? nothing : (st.mtime, st.size))
        serialize(io, grouping)
        serialize(io, grouping_file === nothing ? "" : String(grouping_file))
        serialize(io, mask_btp)
        serialize(io, mask_mode)
        serialize(io, Float64(angle_step))
        serialize(io, Int32[p.id for p in pixels])
        serialize(io, Ei)
        serialize(io, tof_edges)
        serialize(io, Q_edges)
        serialize(io, ω_edges)
        serialize(io, resolution)
        path = _wf_cache_path(cache_dir, "reduce_map_powder", take!(io))
        @info "Reduce-map cache: $path"
        _wf_load_or_compute(path, compute_rmap; disk_cache=disk_cache)
    else
        compute_rmap()
    end

    return PowderCtx{typeof(resolution), typeof(cdf_work)}(instr, String(idf_path), inst, pixels, Ei, tof_edges,
                                                          Q_edges, ω_edges, resolution, cdf_work, Cv, rmap)
end

"""
    eval_powder(ctx, model; eps=1e-12, do_hist=true)

Evaluate one scattering model against an existing powder ctx.

Returns `(Cs, Hraw, Hmean, Hwt)` when `do_hist=true`.
"""
function eval_powder(ctx::PowderCtx, model; eps=1e-12, do_hist::Bool=true)
    Cs = predict_pixel_tof(ctx.inst;
        pixels=ctx.pixels, Ei_meV=ctx.Ei, tof_edges_s=ctx.tof_edges,
        model=model, resolution=ctx.resolution,
        cdf_work=ctx.cdf_work
    )

    if !do_hist
        return (Cs=Cs, Hraw=nothing, Hmean=nothing, Hwt=nothing)
    end

    Hraw = Hist2D(ctx.Q_edges, ctx.ω_edges)
    Hsum = Hist2D(ctx.Q_edges, ctx.ω_edges)
    Hwt  = Hist2D(ctx.Q_edges, ctx.ω_edges)

    reduce_three!(Hraw, Hsum, Hwt, ctx.pixels, ctx.rmap, Cs, ctx.Cv; eps=eps)

    Hmean = Hist2D(ctx.Q_edges, ctx.ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)

    return (Cs=Cs, Hraw=Hraw, Hmean=Hmean, Hwt=Hwt)
end

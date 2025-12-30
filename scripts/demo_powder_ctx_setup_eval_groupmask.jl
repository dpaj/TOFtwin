
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using TOFtwin
using JLD2
using LinearAlgebra
using Serialization
using SHA

# -----------------------------------------------------------------------------
# Demo: "setup" vs "model" separation (with grouping/masking)
#
# Goal:
#   - Pay setup costs once (instrument + grouping/masking + axes + TOF + resolution
#     + CDF-work + Cv(flat) + reduce-map)
#   - Then iterate over many scattering kernels/models cheaply (only Cs + reduce)
#
# Typical usage (many kernels in one Julia process):
#   TOFTWIN_INSTRUMENT=CNCS TOFTWIN_GROUPING=8x2 TOFTWIN_MASK_BTP="Bank=36-50" \
#   julia --project=. scripts/demo_powder_ctx_setup_eval_groupmask.jl \
#       ../sunny_powder_corh2o4.jld2 ../sunny_powder_other.jld2
#
# If you pass multiple Sunny files as ARGS, this script will:
#   - build ctx once (from the first file's axis defaults)
#   - evaluate each Sunny file in sequence using the same ctx
#
# Notes:
#   - For large caches (Cv / reduce-map), keeping them IN RAM is fastest for sweeps.
#     Disk caching via Serialization is supported but can be slower on some systems
#     (especially on synced/AV-scanned directories). Prefer a local cache dir.
# -----------------------------------------------------------------------------

# -----------------------------
# Backend selection
# -----------------------------
backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
if backend == "gl"
    using GLMakie
else
    using CairoMakie
end

# -----------------------------
# Optional: lightweight timing helpers
# -----------------------------
const DO_PROFILE = lowercase(get(ENV, "TOFTWIN_PROFILE", "0")) in ("1","true","yes")

pretty_bytes(b::Real) = b < 1024 ? "$(round(b,digits=1)) B" :
                     b < 1024^2 ? "$(round(b/1024,digits=2)) KiB" :
                     b < 1024^3 ? "$(round(b/1024^2,digits=2)) MiB" :
                                  "$(round(b/1024^3,digits=2)) GiB"

function step(name::AbstractString, f::Function)
    if !DO_PROFILE
        return f()
    end
    t = @timed f()
    @info "STEP" name=name time_s=round(t.time, digits=3) gctime_s=round(t.gctime, digits=3) bytes=pretty_bytes(t.bytes)
    return t.value
end

# -----------------------------
# Output knobs
# -----------------------------
do_hist     = lowercase(get(ENV, "TOFTWIN_DO_HIST", "1")) in ("1","true","yes")
plot_kernel = lowercase(get(ENV, "TOFTWIN_PLOT_KERNEL", "1")) in ("1","true","yes")
do_save     = lowercase(get(ENV, "TOFTWIN_SAVE", "1")) in ("1","true","yes")
outdir      = get(ENV, "TOFTWIN_OUTDIR", joinpath(@__DIR__, "out"))
do_save && mkpath(outdir)

# -----------------------------
# Disk caching (for Cv + reduce-map only; IDF caching handled elsewhere)
# -----------------------------
disk_cache  = lowercase(get(ENV, "TOFTWIN_DISK_CACHE", "1")) in ("1","true","yes")
cache_dir   = get(ENV, "TOFTWIN_CACHE_DIR", joinpath(@__DIR__, ".toftwin_cache"))
cache_ver   = get(ENV, "TOFTWIN_CACHE_VERSION", "ctx_v1")

# IMPORTANT default: off for these large objects (prefer in-RAM reuse for sweeps)
cache_Cv    = lowercase(get(ENV, "TOFTWIN_CACHE_CV", "0")) in ("1","true","yes")
cache_rmap  = lowercase(get(ENV, "TOFTWIN_CACHE_RMAP", "0")) in ("1","true","yes")

disk_cache && mkpath(cache_dir)

# -----------------------------
# Grouping/masking (Mantid-style analogs)
# -----------------------------
#   TOFTWIN_GROUPING=""|"2x1"|"4x1"|"8x1"|"4x2"|"8x2"|"powder"
#   TOFTWIN_GROUPING_FILE=""   optional explicit xml path override
#   TOFTWIN_MASK_BTP=""        e.g. "Bank=40-50;Mode=drop" or "DetectorList=123,124"
#   TOFTWIN_MASK_MODE=drop|zeroΩ (used if spec omits Mode)
#   TOFTWIN_POWDER_ANGLESTEP=0.5
grouping      = strip(get(ENV, "TOFTWIN_GROUPING", "4x2"))
grouping_file = strip(get(ENV, "TOFTWIN_GROUPING_FILE", ""))
mask_btp      = get(ENV, "TOFTWIN_MASK_BTP", "Bank=36-50")#Bank=36-50
mask_mode     = Symbol(lowercase(get(ENV, "TOFTWIN_MASK_MODE", "drop")))
angle_step    = parse(Float64, get(ENV, "TOFTWIN_POWDER_ANGLESTEP", "0.5"))

# -----------------------------
# Sunny loader
# -----------------------------
function load_sunny_powder_jld2(path::AbstractString; outside=0.0)
    radii = energies = S_Qω = nothing
    @load path radii energies S_Qω

    Q = collect(radii)
    W = collect(energies)
    S = Matrix(S_Qω)

    @info "Loaded Sunny table: size(S)=$(size(S)), len(Q)=$(length(Q)), len(ω)=$(length(W))"

    # Sunny often stores as (ω, Q). Fix to (Q, ω).
    if size(S) == (length(W), length(Q))
        @info "Transposing Sunny table (ω,Q) -> (Q,ω)"
        S = permutedims(S)
    end
    @assert size(S) == (length(Q), length(W))

    return TOFtwin.GridKernelPowder(Q, W, S; outside=outside)
end

# -----------------------------
# NTOF suggestion helper (CDF mode striping control)
# -----------------------------
function suggest_ntof(inst, pixels, Ei_meV, tmin, tmax, ω_edges, σt_s;
        α=0.5, β=1/3, ntof_min=200, ntof_max=2000)

    Δω_bin = (ω_edges[end] - ω_edges[1]) / (length(ω_edges)-1)
    ω_samples = unique(filter(ω -> (0.0 <= ω < Ei_meV),
        [0.0, 0.5, 1.0, 2.0, 5.0, 0.25Ei_meV, 0.5Ei_meV, min(10.0, Ei_meV-0.5)]))

    reps = pixels[[1, cld(length(pixels),2), length(pixels)]]
    max_dωdt = 0.0
    for p in reps
        L2p = L2(inst, p.id)
        for ω in ω_samples
            Ef = Ei_meV - ω
            Ef <= 0 && continue
            t = tof_from_EiEf(inst.L1, L2p, Ei_meV, Ef)
            max_dωdt = max(max_dωdt, abs(dω_dt(inst.L1, L2p, Ei_meV, t)))
        end
    end
    max_dωdt == 0.0 && return ntof_min, (tmax-tmin)/ntof_min, max_dωdt, Δω_bin

    dt_from_kernel = β * σt_s
    dt_from_omega  = α * Δω_bin / max_dωdt
    dt = min(dt_from_kernel, dt_from_omega)

    ntof = Int(ceil((tmax - tmin) / dt))
    ntof = clamp(ntof, ntof_min, ntof_max)
    dt = (tmax - tmin) / ntof
    return ntof, dt, max_dωdt, Δω_bin
end

# -----------------------------
# Instrument selection
# -----------------------------
function parse_instrument()
    s = uppercase(get(ENV, "TOFTWIN_INSTRUMENT", "CNCS"))
    if s in ("CNCS",)
        return :CNCS
    elseif s in ("SEQUOIA", "SEQ")
        return :SEQUOIA
    else
        throw(ArgumentError("TOFTWIN_INSTRUMENT must be CNCS or SEQUOIA (got '$s')"))
    end
end

# -----------------------------
# Fast reduce map (same core idea as fastreduce demo)
# -----------------------------
struct ReduceMapPowder
    iQ::Matrix{Int32}
    iW::Matrix{Int32}
    fQ::Matrix{Float32}
    fW::Matrix{Float32}
    valid::BitMatrix
end

function build_reduce_map_powder(inst::Instrument, pixels::Vector{DetectorPixel},
        Ei_meV::Float64, tof_edges_s::Vector{Float64}, Q_edges::Vector{Float64}, ω_edges::Vector{Float64})

    nP   = length(pixels)
    nT   = length(tof_edges_s) - 1
    nQ   = length(Q_edges) - 1
    nW   = length(ω_edges) - 1

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
# Disk cache helpers
# -----------------------------
function _cache_path(prefix::AbstractString, bytes::Vector{UInt8})
    return joinpath(cache_dir, prefix * "_" * bytes2hex(sha1(bytes)) * ".jls")
end

function _load_or_compute(path::AbstractString, builder::Function)
    if disk_cache && isfile(path)
        open(path, "r") do io
            return deserialize(io)
        end
    end
    val = builder()
    if disk_cache
        open(path, "w") do io
            serialize(io, val)
        end
    end
    return val
end

# -----------------------------
# Context object
# -----------------------------
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
Build a "setup" context (do once, reuse many times).
"""
function setup_ctx(; kern_for_axes::GridKernelPowder,
    instr::Symbol,
    idf_path::AbstractString,
    rebuild_geom::Bool=false,
    ψstride::Int=1,
    ηstride::Int=1,
    grouping::AbstractString="",
    grouping_file::Union{Nothing,AbstractString}=nothing,
    mask_btp="",
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
    ntof_max::Int=2000)

    out = step("load instrument from IDF", () -> begin
        if isdefined(TOFtwin.MantidIDF, :load_mantid_idf_diskcached)
            TOFtwin.MantidIDF.load_mantid_idf_diskcached(String(idf_path); rebuild=rebuild_geom)
        else
            @warn "load_mantid_idf_diskcached not found; falling back to load_mantid_idf (will be slower)."
            TOFtwin.MantidIDF.load_mantid_idf(String(idf_path))
        end
    end)

    inst = out.inst
    bank0 = (hasproperty(out, :bank) && out.bank !== nothing) ? out.bank : TOFtwin.DetectorBank(inst.name, out.pixels)

    pixels_gm = step("apply grouping/masking", () -> begin
        TOFtwin.apply_grouping_masking(bank0.pixels;
            instrument=instr,
            grouping=grouping,
            grouping_file=grouping_file,
            mask_btp=mask_btp,
            mask_mode=mask_mode,
            outdir=outdir,
            angle_step=angle_step,
            return_meta=false,
        )
    end)

    bank = TOFtwin.DetectorBank(inst.name, pixels_gm)
    pixels = step("select pixels (AngularDecimate)", () -> sample_pixels(bank, AngularDecimate(ψstride, ηstride)))

    @info "pixels used = $(length(pixels)) (of $(length(bank0.pixels))) after_groupmask=$(length(bank.pixels)) stride=(ψ=$ψstride,η=$ηstride)"
    @info "grouping='$(grouping)' grouping_file='$(grouping_file === nothing ? "" : grouping_file)' mask_btp='$(mask_btp)' mask_mode=$(mask_mode)"

    Q_edges, ω_edges = step("build Q,ω axes", () -> begin
        Qe = collect(range(kern_for_axes.Q[1], kern_for_axes.Q[end]; length=nQbins+1))
        ωlo = min(-2.0, kern_for_axes.ω[1])
        ωhi = max(Ei,  kern_for_axes.ω[end])
        ωe = collect(range(ωlo, ωhi; length=nωbins+1))
        return (Qe, ωe)
    end)

    L2min = minimum(inst.L2); L2max = maximum(inst.L2)
    Ef_min, Ef_max = 1.0, Ei
    tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ef_max)
    tmax = tof_from_EiEf(inst.L1, L2max, Ei, Ef_min)

    σt = σt_us * 1e-6

    ntof = step("choose NTOF", () -> begin
        ntof_env_l = lowercase(String(ntof_env))
        ntof0 = (ntof_env_l in ("auto","suggest","0")) ? 0 : parse(Int, ntof_env_l)
        if ntof0 <= 0 && lowercase(res_mode) == "cdf" && σt_us > 0
            ntof_s, dt_s, maxd, dωbin = suggest_ntof(inst, pixels, Ei, tmin, tmax, ω_edges, σt;
                α=ntof_alpha, β=ntof_beta, ntof_min=ntof_min, ntof_max=ntof_max)
            @info "Auto-selected TOFTWIN_NTOF=$ntof_s (dt=$(round(1e6*dt_s,digits=2)) µs; max|dω/dt|=$(round(maxd,digits=4)) meV/s; Δωbin=$(round(dωbin,digits=4)) meV)"
            return ntof_s
        elseif ntof0 <= 0
            return 500
        else
            return ntof0
        end
    end)

    tof_edges = step("build TOF edges", () -> collect(range(tmin, tmax; length=ntof+1)))

    resolution = step("construct resolution model", () -> begin
        (lowercase(res_mode) == "none" || σt_us <= 0) ? NoResolution() :
        (lowercase(res_mode) == "gh")  ? GaussianTimingResolution(σt; order=gh_order) :
        (lowercase(res_mode) == "cdf") ? GaussianTimingCDFResolution(σt; nsigma=nsigma) :
        throw(ArgumentError("TOFTWIN_RES_MODE must be none|gh|cdf (got '$res_mode')"))
    end)

    cdf_work = step("precompute CDF smear work (optional)", () -> begin
        if !(resolution isa GaussianTimingCDFResolution)
            return nothing
        end
        if isdefined(TOFtwin, :precompute_tof_smear_work)
            return TOFtwin.precompute_tof_smear_work(inst, pixels, Ei, tof_edges, resolution)
        else
            @warn "TOFtwin.precompute_tof_smear_work not found; continuing without precomputed work"
            return nothing
        end
    end)

    function compute_Cv_flat()
        return predict_pixel_tof(inst;
            pixels=pixels, Ei_meV=Ei, tof_edges_s=tof_edges,
            model=(Q,ω)->1.0, resolution=resolution,
            cdf_work=cdf_work
        )
    end

    Cv = step("Cv (flat) load/compute", () -> begin
        if disk_cache && cache_Cv
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
            serialize(io, typeof(resolution))
            path = _cache_path("Cv_flat", take!(io))
            @info "Cv cache: $path  (TOFTWIN_CACHE_CV=1; Serialization may be slower than recompute on some systems)"
            return _load_or_compute(path, compute_Cv_flat)
        else
            return compute_Cv_flat()
        end
    end)

    function compute_rmap()
        return build_reduce_map_powder(inst, pixels, Ei, tof_edges, Q_edges, ω_edges)
    end

    rmap = step("reduce-map load/build", () -> begin
        if disk_cache && cache_rmap
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
            path = _cache_path("reduce_map_powder", take!(io))
            @info "Reduce-map cache: $path  (TOFTWIN_CACHE_RMAP=1; Serialization may be slower than recompute on some systems)"
            return _load_or_compute(path, compute_rmap)
        else
            return compute_rmap()
        end
    end)

    return PowderCtx{typeof(resolution), typeof(cdf_work)}(instr, String(idf_path), inst, pixels, Ei, tof_edges, Q_edges, ω_edges,
                                                         resolution, cdf_work, Cv, rmap)
end

"""
Evaluate ONE model against an existing ctx (per-kernel part).
"""
function eval_model(ctx::PowderCtx, model; eps=1e-12)
    Cs = step("Cs predict_pixel_tof (model)", () -> predict_pixel_tof(ctx.inst;
        pixels=ctx.pixels, Ei_meV=ctx.Ei, tof_edges_s=ctx.tof_edges,
        model=model, resolution=ctx.resolution,
        cdf_work=ctx.cdf_work
    ))

    if !do_hist
        return (Cs=Cs, Hraw=nothing, Hmean=nothing, Hwt=nothing)
    end

    Hraw = Hist2D(ctx.Q_edges, ctx.ω_edges)
    Hsum = Hist2D(ctx.Q_edges, ctx.ω_edges)
    Hwt  = Hist2D(ctx.Q_edges, ctx.ω_edges)

    step("reduce (Hraw/Hsum/Hwt one pass)", () -> reduce_three!(Hraw, Hsum, Hwt, ctx.pixels, ctx.rmap, Cs, ctx.Cv; eps=eps))

    Hmean = Hist2D(ctx.Q_edges, ctx.ω_edges)
    step("compute Hmean", () -> (Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)))

    return (Cs=Cs, Hraw=Hraw, Hmean=Hmean, Hwt=Hwt)
end

function plot_result(ctx::PowderCtx, kern::GridKernelPowder, pred; tag::AbstractString="")
    Q_cent = 0.5 .* (ctx.Q_edges[1:end-1] .+ ctx.Q_edges[2:end])
    ω_cent = 0.5 .* (ctx.ω_edges[1:end-1] .+ ctx.ω_edges[2:end])

    kernel_grid = plot_kernel ? step("kernel grid", () -> [kern(q,w) for q in Q_cent, w in ω_cent]) : nothing
    wt_log = step("log10 weights", () -> log10.(pred.Hwt.counts .+ 1.0))

    fig = step("build figure", () -> begin
        fig = Figure(size=(1400, 900))

        if plot_kernel
            ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="Sunny kernel S(|Q|,ω)")
            heatmap!(ax1, Q_cent, ω_cent, kernel_grid)
        else
            ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="(kernel plot disabled)")
        end

        ax2 = Axis(fig[1,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin predicted (raw SUM)")
        heatmap!(ax2, Q_cent, ω_cent, pred.Hraw.counts)

        ax3 = Axis(fig[2,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="weights log10(N+1)")
        heatmap!(ax3, Q_cent, ω_cent, wt_log)

        ax4 = Axis(fig[2,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="vanadium-normalized MEAN")
        heatmap!(ax4, Q_cent, ω_cent, pred.Hmean.counts)

        return fig
    end)

    if do_save
        fname = isempty(tag) ? "demo_powder_ctx_$(lowercase(String(ctx.instr))).png" :
                               "demo_powder_ctx_$(lowercase(String(ctx.instr)))_$(tag).png"
        outpng = joinpath(outdir, fname)
        step("save PNG", () -> save(outpng, fig))
        @info "Wrote $outpng"
    end
    display(fig)
end

# -----------------------------------------------------------------------------
# Main: build ctx once, then evaluate 1..N Sunny kernels
# -----------------------------------------------------------------------------
instr = parse_instrument()
idf_path = instr === :CNCS ? joinpath(@__DIR__, "CNCS_Definition_2025B.xml") :
                            joinpath(@__DIR__, "SEQUOIA_Definition.xml")

sunny_paths = length(ARGS) > 0 ? ARGS : [joinpath(@__DIR__, "..", "sunny_powder_corh2o4.jld2")]

kern0 = step("load Sunny kernel (axes anchor)", () -> load_sunny_powder_jld2(sunny_paths[1]; outside=0.0))

default_Ei = instr === :SEQUOIA ? "30.0" : "12.0"
Ei = parse(Float64, get(ENV, "TOFTWIN_EI", default_Ei))  # meV

nQbins = parse(Int, get(ENV, "TOFTWIN_NQBINS", "220"))
nωbins = parse(Int, get(ENV, "TOFTWIN_NWBINS", "240"))

ψstride = parse(Int, get(ENV, "TOFTWIN_PSI_STRIDE", "1"))
ηstride = parse(Int, get(ENV, "TOFTWIN_ETA_STRIDE", "1"))

res_mode = lowercase(get(ENV, "TOFTWIN_RES_MODE", "cdf"))
σt_us    = parse(Float64, get(ENV, "TOFTWIN_SIGMA_T_US", "100.0"))
gh_order = parse(Int, get(ENV, "TOFTWIN_GH_ORDER", "3"))
nsigma   = parse(Float64, get(ENV, "TOFTWIN_NSIGMA", "6.0"))

ntof_env   = get(ENV, "TOFTWIN_NTOF", "auto")
ntof_alpha = parse(Float64, get(ENV, "TOFTWIN_NTOF_ALPHA", "1.5"))
ntof_beta  = parse(Float64, get(ENV, "TOFTWIN_NTOF_BETA",  "1"))
ntof_min   = parse(Int, get(ENV, "TOFTWIN_NTOF_MIN", "200"))
ntof_max   = parse(Int, get(ENV, "TOFTWIN_NTOF_MAX", "2000"))

rebuild_geom = lowercase(get(ENV, "TOFTWIN_REBUILD_GEOM", "0")) in ("1","true","yes")

ctx = setup_ctx(; kern_for_axes=kern0,
    instr=instr, idf_path=idf_path, rebuild_geom=rebuild_geom,
    ψstride=ψstride, ηstride=ηstride,
    grouping=grouping,
    grouping_file=isempty(grouping_file) ? nothing : grouping_file,
    mask_btp=mask_btp,
    mask_mode=mask_mode,
    angle_step=angle_step,
    Ei=Ei, nQbins=nQbins, nωbins=nωbins,
    res_mode=res_mode, σt_us=σt_us, gh_order=gh_order, nsigma=nsigma,
    ntof_env=ntof_env, ntof_alpha=ntof_alpha, ntof_beta=ntof_beta, ntof_min=ntof_min, ntof_max=ntof_max
)

@info "SETUP complete. Reusing ctx for $(length(sunny_paths)) kernel(s)."

for (i, spath) in enumerate(sunny_paths)
    kern = (i == 1) ? kern0 : step("load Sunny kernel", () -> load_sunny_powder_jld2(spath; outside=0.0))
    model = (q,w) -> kern(q,w)

    tag = "k$(i)"
    @info "EVALUATE kernel $i / $(length(sunny_paths))" path=spath

    pred = eval_model(ctx, model)

    if do_hist
        plot_result(ctx, kern, pred; tag=tag)
    end

    if do_save && do_hist
        outjld = joinpath(outdir, "demo_powder_ctx_$(lowercase(String(instr)))_$(tag).jld2")
        @save outjld instr=String(instr) idf_path ctx_Ei=ctx.Ei tof_edges=ctx.tof_edges Q_edges=ctx.Q_edges ω_edges=ctx.ω_edges
        @save outjld grouping grouping_file mask_btp mask_mode angle_step
        @save outjld Hraw=pred.Hraw Hmean=pred.Hmean Hwt=pred.Hwt
        @info "Wrote $outjld"
    end
end

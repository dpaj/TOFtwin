
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin
using JLD2
using LinearAlgebra
using Serialization
using SHA

# -----------------------------------------------------------------------------
# Demo: Powder prediction from a Sunny.jl powder table, using a Mantid IDF instrument,
#       with switchable timing-resolution broadening (none | gh | cdf).
#
# Latest speedups used here (no changes to TOFTwin.jl required):
#   1) CDF mode: precompute reusable TOF-smearing "work" and reuse it for forward calls
#      (optionally disk-cached via TOFtwin.precompute_tof_smear_work_diskcached if available).
#   2) Cache Cv (flat-kernel) to disk (model-independent) so multi-model sweeps don't pay
#      a second forward pass every iteration.
#   3) Fast reduction: precompute a (pixel,tof-bin-center) -> (Q,ω) bilinear deposit map
#      (disk-cached) and reduce Hraw/Hsum/Hwt in ONE pass without allocating Cnorm/W.
#
# Run examples:
#   TOFTWIN_INSTRUMENT=CNCS    julia --project=. scripts/demo_powder_from_sunny_with_idf_resolution_cdfwork_fastreduce.jl
#   TOFTWIN_INSTRUMENT=SEQUOIA julia --project=. scripts/demo_powder_from_sunny_with_idf_resolution_cdfwork_fastreduce.jl
#
# Key env vars:
#   TOFTWIN_RES_MODE=none|gh|cdf        (default cdf)
#   TOFTWIN_SIGMA_T_US=100.0            (timing σt in microseconds)
#   TOFTWIN_NSIGMA=6.0                  (CDF truncation)
#   TOFTWIN_GH_ORDER=3|5|7              (GH)
#
#   TOFTWIN_DISK_CACHE=1                (default 1)
#   TOFTWIN_CACHE_DIR=scripts/.toftwin_cache (default)
#   TOFTWIN_CACHE_CV=1                  (default 1 when disk cache enabled)
#   TOFTWIN_CACHE_VERSION=v1            (bump to invalidate)
#
#   TOFTWIN_DO_HIST=1                   (default 1)
#   TOFTWIN_PLOT_KERNEL=0               (default 0; kernel grid can be expensive)
#   TOFTWIN_SAVE=1                      (default 1; saves PNG to outdir)
# -----------------------------------------------------------------------------

# Backend selection
backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
if backend == "gl"
    using GLMakie
else
    using CairoMakie
end

# -----------------------------
# Output knobs
# -----------------------------
do_hist      = lowercase(get(ENV, "TOFTWIN_DO_HIST", "1")) in ("1","true","yes")
plot_kernel  = lowercase(get(ENV, "TOFTWIN_PLOT_KERNEL", "1")) in ("1","true","yes")
do_save      = lowercase(get(ENV, "TOFTWIN_SAVE", "1")) in ("1","true","yes")
outdir       = get(ENV, "TOFTWIN_OUTDIR", joinpath(@__DIR__, "out"))
do_save && mkpath(outdir)

# -----------------------------
# Disk caching
# -----------------------------
disk_cache = lowercase(get(ENV, "TOFTWIN_DISK_CACHE", "1")) in ("1","true","yes")
cache_dir  = get(ENV, "TOFTWIN_CACHE_DIR", joinpath(@__DIR__, ".toftwin_cache"))
cache_ver  = get(ENV, "TOFTWIN_CACHE_VERSION", "v1")
cache_Cv   = lowercase(get(ENV, "TOFTWIN_CACHE_CV", disk_cache ? "1" : "0")) in ("1","true","yes")

disk_cache && mkpath(cache_dir)

# -----------------------------
# Helper: load Sunny powder table as a TOFtwin kernel
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
# Helper: suggest a good NTOF for CDF mode (avoid striping without being crazy slow)
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

instr = parse_instrument()
idf_path = instr === :CNCS ? joinpath(@__DIR__, "CNCS_Definition_2025B.xml") :
                            joinpath(@__DIR__, "SEQUOIA_Definition.xml")

# -----------------------------
# Sunny input
# -----------------------------
sunny_path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "sunny_powder_corh2o4.jld2")
@info "Sunny input file: $sunny_path"
kern = load_sunny_powder_jld2(sunny_path; outside=0.0)
model = (q, w) -> kern(q, w)

# -----------------------------
# IDF -> Instrument (use diskcache loader if available)
# -----------------------------
rebuild_geom = lowercase(get(ENV, "TOFTWIN_REBUILD_GEOM", "0")) in ("1","true","yes")
out = if isdefined(TOFtwin.MantidIDF, :load_mantid_idf_diskcached)
    TOFtwin.MantidIDF.load_mantid_idf_diskcached(idf_path; rebuild=rebuild_geom)
else
    @warn "load_mantid_idf_diskcached not found; falling back to load_mantid_idf (will be slower)."
    TOFtwin.MantidIDF.load_mantid_idf(idf_path)
end
inst = out.inst
bank = (hasproperty(out, :bank) && out.bank !== nothing) ? out.bank : TOFtwin.DetectorBank(inst.name, out.pixels)

# Pixel decimation (still useful, but this demo focuses on speedups without needing it)
default_stride = instr === :SEQUOIA ? "1" : "1"
ψstride = parse(Int, get(ENV, "TOFTWIN_PSI_STRIDE", default_stride))
ηstride = parse(Int, get(ENV, "TOFTWIN_ETA_STRIDE", default_stride))
pix_used = sample_pixels(bank, AngularDecimate(ψstride, ηstride))
@info "Instrument = $instr"
@info "IDF: $idf_path"
@info "pixels used = $(length(pix_used)) (of $(length(bank.pixels)))  stride=(ψ=$ψstride, η=$ηstride)"

# -----------------------------
# Axes / TOF
# -----------------------------
default_Ei = instr === :SEQUOIA ? "30.0" : "12.0"
Ei = parse(Float64, get(ENV, "TOFTWIN_EI", default_Ei))  # meV

nQbins = parse(Int, get(ENV, "TOFTWIN_NQBINS", "220"))
nωbins = parse(Int, get(ENV, "TOFTWIN_NWBINS", "240"))

Q_edges = collect(range(kern.Q[1], kern.Q[end]; length=nQbins+1))
ωlo = min(-2.0, kern.ω[1])
ωhi = max(Ei,  kern.ω[end])
ω_edges = collect(range(ωlo, ωhi; length=nωbins+1))

# TOF window (rough)
L2min = minimum(inst.L2)
L2max = maximum(inst.L2)
Ef_min, Ef_max = 1.0, Ei
tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ef_max)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, Ef_min)

# Parse resolution mode + σt early (needed for auto NTOF suggestion)
res_mode = lowercase(get(ENV, "TOFTWIN_RES_MODE", "cdf"))  # none | gh | cdf
σt_us    = parse(Float64, get(ENV, "TOFTWIN_SIGMA_T_US", "100.0"))
σt       = σt_us * 1e-6

# NTOF selection
ntof_env = lowercase(get(ENV, "TOFTWIN_NTOF", "auto"))
α = parse(Float64, get(ENV, "TOFTWIN_NTOF_ALPHA", "0.5"))
β = parse(Float64, get(ENV, "TOFTWIN_NTOF_BETA",  "0.333333333333"))
ntof_min = parse(Int, get(ENV, "TOFTWIN_NTOF_MIN", "200"))
ntof_max = parse(Int, get(ENV, "TOFTWIN_NTOF_MAX", "2000"))

ntof = (ntof_env in ("auto","suggest","0")) ? 0 : parse(Int, ntof_env)

# Resolution params (needed for auto)
gh_order = parse(Int, get(ENV, "TOFTWIN_GH_ORDER", "3"))
nsigma   = parse(Float64, get(ENV, "TOFTWIN_NSIGMA", "6.0"))

if ntof <= 0 && res_mode == "cdf" && σt_us > 0
    ntof_s, dt_s, maxd, dωbin = suggest_ntof(inst, pix_used, Ei, tmin, tmax, ω_edges, σt;
        α=α, β=β, ntof_min=ntof_min, ntof_max=ntof_max)
    ntof = ntof_s
    @info "Auto-selected TOFTWIN_NTOF=$ntof  (dt=$(round(1e6*dt_s,digits=2)) µs; max|dω/dt|=$(round(maxd,digits=4)) meV/s; Δωbin=$(round(dωbin,digits=4)) meV)"
elseif ntof <= 0
    ntof = 500
    @info "TOFTWIN_NTOF=auto but res_mode=$res_mode or σt<=0; using default NTOF=$ntof"
end

tof_edges = collect(range(tmin, tmax; length=ntof+1))
@info "TOF window (ms) = $(1e3*tmin) .. $(1e3*tmax)"
@info "Sunny kernel domain: Q∈($(kern.Q[1]), $(kern.Q[end]))  ω∈($(kern.ω[1]), $(kern.ω[end]))"

# -----------------------------
# Resolution object
# -----------------------------
resolution =
    (res_mode == "none" || σt_us <= 0) ? NoResolution() :
    (res_mode == "gh")  ? GaussianTimingResolution(σt; order=gh_order) :
    (res_mode == "cdf") ? GaussianTimingCDFResolution(σt; nsigma=nsigma) :
    throw(ArgumentError("TOFTWIN_RES_MODE must be none|gh|cdf (got '$res_mode')"))

@info "Resolution mode = $res_mode"
@info "Resolution model = $(typeof(resolution))"
σt_us > 0 && @info "  σt = $(σt_us) µs"
res_mode == "gh"  && σt_us > 0 && @info "  GH order = $gh_order"
res_mode == "cdf" && σt_us > 0 && @info "  nsigma = $nsigma"

# -----------------------------
# Disk cache helper (stdlib-only)
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
# Precompute (optional) CDF smear work (reused across many model evals)
# -----------------------------
cdf_work = nothing
if resolution isa GaussianTimingCDFResolution
    if isdefined(TOFtwin, :precompute_tof_smear_work_diskcached) && disk_cache
        cdf_work = TOFtwin.precompute_tof_smear_work_diskcached(inst;
            pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges, resolution=resolution,
            cache_dir=cache_dir,
            cache_tag=cache_ver * "|instr=$(String(instr))|ψ=$(ψstride)|η=$(ηstride)"
        )
        @info "Using disk-cached CDF smear work"
    elseif isdefined(TOFtwin, :precompute_tof_smear_work)
        cdf_work = TOFtwin.precompute_tof_smear_work(inst, pix_used, Ei, tof_edges, resolution)
        @info "Using in-session precomputed CDF smear work"
    else
        @warn "TOFtwin.precompute_tof_smear_work not found; continuing without precomputed work"
    end
end

# -----------------------------
# Cache Cv to disk (model-independent)
# -----------------------------
function compute_Cv_flat()
    return predict_pixel_tof(inst;
        pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
        model=(Q,ω)->1.0, resolution=resolution,
        cdf_work=cdf_work
    )
end

Cv = if cache_Cv && disk_cache
    io = IOBuffer()
    serialize(io, cache_ver)
    serialize(io, String(instr))
    serialize(io, idf_path)
    serialize(io, rebuild_geom)
    # use idf stat to invalidate on IDF update
    st = try stat(idf_path) catch; nothing end
    serialize(io, st === nothing ? nothing : (st.mtime, st.size))
    serialize(io, Int32[p.id for p in pix_used])
    serialize(io, Ei)
    serialize(io, tof_edges)
    serialize(io, typeof(resolution))
    if resolution isa GaussianTimingCDFResolution
        serialize(io, resolution.nsigma)
        serialize(io, resolution.σt isa Function ? :callable : Float64(resolution.σt))
    elseif resolution isa GaussianTimingResolution
        serialize(io, resolution.order)
        serialize(io, Float64(resolution.σt))
    end
    path = _cache_path("Cv_flat", take!(io))
    @info "Cv cache: $path"
    _load_or_compute(path, compute_Cv_flat)
else
    compute_Cv_flat()
end

# -----------------------------
# Predict Cs (model-dependent, typically NOT disk-cached)
# -----------------------------
Cs = predict_pixel_tof(inst;
    pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
    model=model, resolution=resolution,
    cdf_work=cdf_work
)

# If histogramming is off, stop here.
if !do_hist
    @info "TOFTWIN_DO_HIST=0: skipping histogramming/plotting."
    @info "Computed Cs and Cv (pixel×TOF)."
    exit()
end

# -----------------------------
# Precompute and cache a fast reduction map:
# (pixel, tof-bin-center) -> bilinear deposit indices+fractions on (Q_edges, ω_edges)
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

            # base bin (needs i and i+1)
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

            # clamp to [0,1] just in case of tiny numerical drift
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

function load_or_build_reduce_map()
    io = IOBuffer()
    serialize(io, cache_ver)
    serialize(io, String(instr))
    serialize(io, idf_path)
    st = try stat(idf_path) catch; nothing end
    serialize(io, st === nothing ? nothing : (st.mtime, st.size))
    serialize(io, Int32[p.id for p in pix_used])
    serialize(io, Ei)
    serialize(io, tof_edges)
    serialize(io, Q_edges)
    serialize(io, ω_edges)
    path = _cache_path("reduce_map_powder", take!(io))
    @info "Reduce-map cache: $path"
    _load_or_compute(path, () -> build_reduce_map_powder(inst, pix_used, Ei, tof_edges, Q_edges, ω_edges))
end

rmap = disk_cache ? load_or_build_reduce_map() : build_reduce_map_powder(inst, pix_used, Ei, tof_edges, Q_edges, ω_edges)

# -----------------------------
# Fast one-pass reductions without allocating Cnorm/W
# -----------------------------
function reduce_three!(Hraw::Hist2D, Hsum::Hist2D, Hwt::Hist2D,
        inst::Instrument, pixels::Vector{DetectorPixel}, rmap::ReduceMapPowder,
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

            # bilinear weights
            w00 = (1 - fq) * (1 - fw)
            w10 = fq       * (1 - fw)
            w01 = (1 - fq) * fw
            w11 = fq       * fw

            # raw sum
            Hraw.counts[iq,   iw]   += w00 * cs
            Hraw.counts[iq+1, iw]   += w10 * cs
            Hraw.counts[iq,   iw+1] += w01 * cs
            Hraw.counts[iq+1, iw+1] += w11 * cs

            # mean-of-model via vanadium normalization (sum of normalized samples)
            v = cs / (cv + eps)
            Hsum.counts[iq,   iw]   += w00 * v
            Hsum.counts[iq+1, iw]   += w10 * v
            Hsum.counts[iq,   iw+1] += w01 * v
            Hsum.counts[iq+1, iw+1] += w11 * v

            # coverage weight (count of contributing samples)
            Hwt.counts[iq,   iw]   += w00
            Hwt.counts[iq+1, iw]   += w10
            Hwt.counts[iq,   iw+1] += w01
            Hwt.counts[iq+1, iw+1] += w11
        end
    end
    return nothing
end

Hraw = Hist2D(Q_edges, ω_edges)
Hsum = Hist2D(Q_edges, ω_edges)
Hwt  = Hist2D(Q_edges, ω_edges)
reduce_three!(Hraw, Hsum, Hwt, inst, pix_used, rmap, Cs, Cv; eps=1e-12)

Hmean = Hist2D(Q_edges, ω_edges)
Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ 1e-12)

# -----------------------------
# Plot
# -----------------------------
Q_cent = 0.5 .* (Q_edges[1:end-1] .+ Q_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

kernel_grid = plot_kernel ? [kern(q, w) for q in Q_cent, w in ω_cent] : nothing
raw_sum  = Hraw.counts
mean_map = Hmean.counts
wt_log   = log10.(Hwt.counts .+ 1.0)

fig = Figure(size=(1400, 900))

title_tag = (σt_us <= 0 || res_mode == "none") ? "no resolution" :
            (res_mode == "gh") ? "GH(order=$(gh_order)), σt=$(σt_us) µs" :
            "CDF(nsigma=$(nsigma)), σt=$(σt_us) µs"

if plot_kernel
    ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="Sunny kernel S(|Q|,ω) on TOFtwin grid")
    heatmap!(ax1, Q_cent, ω_cent, kernel_grid)
else
    ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="(kernel plot disabled) set TOFTWIN_PLOT_KERNEL=1")
end

ax2 = Axis(fig[1,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin predicted (raw SUM) — $(String(instr)) IDF ($title_tag)")
heatmap!(ax2, Q_cent, ω_cent, raw_sum)

ax3 = Axis(fig[2,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin weights log10(N+1) — $(String(instr)) IDF")
heatmap!(ax3, Q_cent, ω_cent, wt_log)

ax4 = Axis(fig[2,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin vanadium-normalized MEAN ($title_tag)")
heatmap!(ax4, Q_cent, ω_cent, mean_map)

if do_save
    outpng = joinpath(outdir, "demo_powder_fastreduce_$(lowercase(String(instr)))_$(res_mode).png")
    save(outpng, fig)
    @info "Wrote $outpng"
end
display(fig)
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin
using JLD2
using LinearAlgebra

# -----------------------------------------------------------------------------
# Demo: Powder prediction from a Sunny.jl powder table, using a Mantid IDF instrument,
#       WITH switchable timing-resolution broadening:
#
#   - none:  NoResolution()
#   - gh:    GaussianTimingResolution(σt; order=3|5|7)  (Gauss–Hermite sampling in TOF)
#   - cdf:   GaussianTimingCDFResolution(σt; nsigma=...) (TOF-domain CDF/bin-overlap convolution)
#
# Run examples:
#   TOFTWIN_INSTRUMENT=CNCS    julia --project=. scripts/demo_powder_from_sunny_with_idf_resolution_switch.jl
#   TOFTWIN_INSTRUMENT=SEQUOIA julia --project=. scripts/demo_powder_from_sunny_with_idf_resolution_switch.jl
#
# Switch resolution mode:
#   TOFTWIN_RES_MODE=none|gh|cdf
#
# Timing width:
#   TOFTWIN_SIGMA_T_US = timing σt in microseconds (e.g. 20.0)
#
# GH knobs:
#   TOFTWIN_GH_ORDER = 3|5|7 (default 3)
#
# CDF knobs:
#   TOFTWIN_NSIGMA = truncation radius in σ (default 4.0)
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
# TOFTWIN_DO_HIST: "1" (default) -> reduce into (|Q|,ω) histograms and plot/save PNG
#                  "0"           -> skip histogramming; just compute pixel×TOF arrays
# TOFTWIN_SAVE_PIXEL_TOF: "0" (default) -> don't save big arrays
#                         "1"           -> save Cs, Cv, etc. to JLD2 in ../out/
do_hist = lowercase(get(ENV, "TOFTWIN_DO_HIST", "1")) in ("1","true","yes")
save_pixel_tof = lowercase(get(ENV, "TOFTWIN_SAVE_PIXEL_TOF", "0")) in ("1","true","yes")
# -----------------------------
# Optional disk caching (stdlib-only) for expensive instrument-only precomputations.
#
# TOFTWIN_DISK_CACHE=1 enables:
#   - cached (pixel,tof)->(|Q|,ω) reduction map (saves kinematics during reduction)
#   - cached Cv (flat-kernel) pixel×TOF response (saves one full forward pass)
#
# Cache directory:
#   TOFTWIN_CACHE_DIR=... (default: scripts/.toftwin_cache)
# -----------------------------
disk_cache = lowercase(get(ENV, "TOFTWIN_DISK_CACHE", "1")) in ("1","true","yes")
cache_dir  = get(ENV, "TOFTWIN_CACHE_DIR", joinpath(@__DIR__, ".toftwin_cache"))

# Optional: also cache Cs (big!) keyed by Sunny input file
cache_Cs = lowercase(get(ENV, "TOFTWIN_CACHE_CS", "0")) in ("1","true","yes")
cache_Cv = lowercase(get(ENV, "TOFTWIN_CACHE_CV", "1")) in ("1","true","yes")  # default on when disk_cache=1

# Output (plots)
do_save = lowercase(get(ENV, "TOFTWIN_SAVE", "1")) in ("1","true","yes")
outdir  = get(ENV, "TOFTWIN_OUTDIR", joinpath(@__DIR__, "out"))
do_save && mkpath(outdir)


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

# -----------------------------
# Helper: suggest a good NTOF for CDF mode (avoid striping without being crazy slow)
# -----------------------------
"""
Suggest NTOF (number of TOF bins) for CDF timing smearing.

We try to avoid "striping" in (|Q|,ω) maps caused by mapping discrete TOF-bin centers into ω.
Heuristic: choose dt such that the ω-step per TOF bin is a fraction of the ω-bin width, and
also resolve the timing kernel itself.

dt = min( β*σt,  α*Δω_bin / max|dω/dt| )

α ~ 0.3–0.7 (smaller = safer against striping)
β ~ 0.3–0.5 (smaller = better kernel resolution)
"""
function suggest_ntof(inst, pixels, Ei_meV, tmin, tmax, ω_edges, σt_s; α=0.5, β=1/3, ntof_min=200, ntof_max=2000)
    Δω_bin = (ω_edges[end] - ω_edges[1]) / (length(ω_edges)-1)

    # Sample ω points (emphasize near elastic where |dω/dt| is often largest)
    ω_samples = unique(filter(ω -> (0.0 <= ω < Ei_meV),
        [0.0, 0.5, 1.0, 2.0, 5.0, 0.25Ei_meV, 0.5Ei_meV, min(10.0, Ei_meV-0.5)]))

    # Representative pixels (cheap): first / middle / last
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
# Input Sunny file
# -----------------------------
sunny_path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "sunny_powder_corh2o4.jld2")
@info "Sunny input file: $sunny_path"
# Signature for caching model-dependent arrays (Cs). We include file mtime+size when available.
model_cache_tag = let path = String(sunny_path)
    st = try
        stat(path)
    catch
        nothing
    end
    st === nothing ? "sunny|" * path : "sunny|" * path * "|mtime=" * string(st.mtime) * "|size=" * string(st.size)
end
kern = load_sunny_powder_jld2(sunny_path; outside=0.0)

# -----------------------------
# IDF -> Instrument
# -----------------------------
rebuild_geom = lowercase(get(ENV, "TOFTWIN_REBUILD_GEOM", "0")) in ("1","true","yes")

out = if isdefined(TOFtwin.MantidIDF, :load_mantid_idf_diskcached)
    TOFtwin.MantidIDF.load_mantid_idf_diskcached(idf_path; rebuild=rebuild_geom)
else
    @warn "load_mantid_idf_diskcached not found; falling back to load_mantid_idf (will be slower)."
    TOFtwin.MantidIDF.load_mantid_idf(idf_path)
end
inst = out.inst

bank = if hasproperty(out, :bank) && out.bank !== nothing
    out.bank
else
    TOFtwin.DetectorBank(inst.name, out.pixels)
end

# Speed knob: subsample pixels while iterating
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

# -----------------------------
# Choose comparison grid (Q_edges, ω_edges)
# -----------------------------
nQbins = parse(Int, get(ENV, "TOFTWIN_NQBINS", "220"))
nωbins = parse(Int, get(ENV, "TOFTWIN_NWBINS", "240"))

Q_edges = collect(range(kern.Q[1], kern.Q[end]; length=nQbins+1))
ωlo = min(-2.0, kern.ω[1])
ωhi = max(Ei,  kern.ω[end])
ω_edges = collect(range(ωlo, ωhi; length=nωbins+1))



# TOF range based on geometry (rough)
L2min = minimum(inst.L2)
L2max = maximum(inst.L2)
Ef_min, Ef_max = 1.0, Ei

tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ef_max)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, Ef_min)

# NTOF selection
# - If TOFTWIN_NTOF is an integer (e.g. "500"), use it.
# - If TOFTWIN_NTOF is "auto" (default) or "0", and you're in CDF mode with σt>0,
#   pick an NTOF that reduces striping without exploding runtime.
#
# Tuning knobs (optional):
#   TOFTWIN_NTOF_ALPHA  (default 0.5)   target ω-step fraction
#   TOFTWIN_NTOF_BETA   (default 0.333) target dt relative to σt
#   TOFTWIN_NTOF_MIN    (default 200)
#   TOFTWIN_NTOF_MAX    (default 2000)
ntof_env = lowercase(get(ENV, "TOFTWIN_NTOF", "auto"))
α = parse(Float64, get(ENV, "TOFTWIN_NTOF_ALPHA", "0.5"))
β = parse(Float64, get(ENV, "TOFTWIN_NTOF_BETA",  "0.333333333333"))
ntof_min = parse(Int, get(ENV, "TOFTWIN_NTOF_MIN", "200"))
ntof_max = parse(Int, get(ENV, "TOFTWIN_NTOF_MAX", "2000"))

# We'll parse resolution mode + σt early (needed for auto NTOF suggestion)
res_mode = lowercase(get(ENV, "TOFTWIN_RES_MODE", "cdf"))  # none | gh | cdf
σt_us    = parse(Float64, get(ENV, "TOFTWIN_SIGMA_T_US", "100.0"))
σt       = σt_us * 1e-6

ntof =
    (ntof_env in ("auto","suggest","0")) ? 0 :
    parse(Int, ntof_env)

if ntof <= 0 && res_mode == "cdf" && σt_us > 0
    ntof_s, dt_s, maxd, dωbin = suggest_ntof(inst, pix_used, Ei, tmin, tmax, ω_edges, σt; α=α, β=β, ntof_min=ntof_min, ntof_max=ntof_max)
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
# Resolution: pick mode + parameters
# -----------------------------
# (res_mode, σt_us, σt already parsed above for auto-NTOF selection)

gh_order = parse(Int, get(ENV, "TOFTWIN_GH_ORDER", "3"))
nsigma   = parse(Float64, get(ENV, "TOFTWIN_NSIGMA", "6.0"))

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
# Run TOFtwin forward prediction using the Sunny kernel
# -----------------------------
model = (q, w) -> kern(q, w)


function predict_powder_mean_Qω_with_resolution(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    Q_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    model,
    resolution::AbstractResolutionModel,
    eps::Float64 = 1e-12,
    disk_cache::Bool=false,
    cache_dir::AbstractString=joinpath(@__DIR__, ".toftwin_cache"),
    cache_Cs::Bool=false,
    cache_Cv::Bool=true,
    model_cache_tag::AbstractString="")

    Cs = if disk_cache && cache_Cs
        TOFtwin.predict_pixel_tof_diskcached(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            model=model, resolution=resolution,
            cache_dir=cache_dir,
            cache_tag="Cs|" * model_cache_tag
        )
    else
        predict_pixel_tof(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            model=model, resolution=resolution
        )
    end

    Cv = if disk_cache && cache_Cv
        TOFtwin.predict_pixel_tof_diskcached(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            model=(Q,ω)->1.0, resolution=resolution,
            cache_dir=cache_dir,
            cache_tag="Cv|flat"
        )
    else
        predict_pixel_tof(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            model=(Q,ω)->1.0, resolution=resolution
        )
    end

    # If histogramming is off, return just the pixel×TOF arrays (instrument-space use)
    if !do_hist
        return (Cs=Cs, Cv=Cv, Cnorm=normalize_by_vanadium(Cs, Cv; eps=eps))
    end

    # Optional: cache the expensive bin-center kinematics map used by reduction.
    # This avoids re-running Ef_from_tof/Qω_from_pixel in each reduce call.
    map = disk_cache ? TOFtwin.precompute_pixel_tof_Qω_map(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        cache=true, cache_dir=cache_dir, cache_tag=""
    ) : nothing

    Hraw = if map === nothing
        TOFtwin.reduce_pixel_tof_to_Qω_powder(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            C=Cs, Q_edges=Q_edges, ω_edges=ω_edges
        )
    else
        TOFtwin.reduce_pixel_tof_to_Qω_powder(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            C=Cs, Q_edges=Q_edges, ω_edges=ω_edges,
            map=map
        )
    end

    Cnorm = normalize_by_vanadium(Cs, Cv; eps=eps)

    Hsum = if map === nothing
        TOFtwin.reduce_pixel_tof_to_Qω_powder(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            C=Cnorm, Q_edges=Q_edges, ω_edges=ω_edges
        )
    else
        TOFtwin.reduce_pixel_tof_to_Qω_powder(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            C=Cnorm, Q_edges=Q_edges, ω_edges=ω_edges,
            map=map
        )
    end

    W = zeros(size(Cnorm))
    W[Cv .> 0.0] .= 1.0
    Hwt = if map === nothing
        TOFtwin.reduce_pixel_tof_to_Qω_powder(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            C=W, Q_edges=Q_edges, ω_edges=ω_edges
        )
    else
        TOFtwin.reduce_pixel_tof_to_Qω_powder(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            C=W, Q_edges=Q_edges, ω_edges=ω_edges,
            map=map
        )
    end

    Hmean = Hist2D(Q_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)

    return (Hraw=Hraw, Hsum=Hsum, Hwt=Hwt, Hmean=Hmean, Cs=Cs, Cv=Cv, Cnorm=Cnorm)
end

pred = predict_powder_mean_Qω_with_resolution(inst;
    pixels=pix_used,
    Ei_meV=Ei,
    tof_edges_s=tof_edges,
    Q_edges=Q_edges,
    ω_edges=ω_edges,
    model=model,
    resolution=resolution,
    disk_cache=disk_cache,
    cache_dir=cache_dir,
    cache_Cs=cache_Cs,
    cache_Cv=(disk_cache && cache_Cv),
    model_cache_tag=model_cache_tag
)
# -----------------------------
# Concept helper: approximate σt -> energy width at a representative pixel
# -----------------------------
if σt_us > 0
    p_ref = pix_used[clamp(Int(round(length(pix_used)/2)), 1, length(pix_used))]
    L2p = L2(inst, p_ref.id)
    @info "Energy-width estimate from timing resolution (representative pixel id=$(p_ref.id))"
    @info "  Using σ_ω ≈ |dω/dt| σt. (FWHM ≈ 2.355 σ_ω)"

    @info "Q-width estimate from timing resolution (timing-only; no angular/divergence/mosaic broadening)"
    @info "  Using σ_|Q| ≈ |d|Q|/dt| σt at fixed pixel direction (kf changes in magnitude only)."

    ω_test = [0.0, 2.0, 5.0, min(10.0, Ei-0.5)]
    for ω in ω_test
        Ef = Ei - ω
        Ef <= 0 && continue
        t = tof_from_EiEf(inst.L1, L2p, Ei, Ef)
        σω = abs(dω_dt(inst.L1, L2p, Ei, t)) * σt          # meV
        fwhm = 2.354820045 * σω
        @info "  ω=$(round(ω,digits=3)) meV: σ_ω=$(round(σω,digits=5)) meV, FWHM=$(round(fwhm,digits=5)) meV"
        dQ = dQmag_dt(inst.L1, L2p, p_ref.r_L, Ei, t; r_samp_L=inst.r_samp_L, ε=max(σt/20, 1e-7))
        σQ = abs(dQ) * σt
        fwhmQ = 2.354820045 * σQ
        # also report |Q| at this (ω) point for context
        Qvec, _ = Qω_from_pixel(p_ref.r_L, Ei, Ef; r_samp_L=inst.r_samp_L)
        Qmag = norm(Qvec)
        @info "           |Q|=$(round(Qmag,digits=5)) Å⁻¹: σ_|Q|=$(round(σQ,digits=6)) Å⁻¹, FWHM=$(round(fwhmQ,digits=6)) Å⁻¹"
    end
end


# -----------------------------
# Optional: save pixel×TOF arrays (instrument space)
# -----------------------------

if !do_hist
    @info "TOFTWIN_DO_HIST=0: skipping (|Q|,ω) histogramming/plotting."
    if save_pixel_tof
        outjld = joinpath(outdir, "pixel_tof_$(lowercase(String(instr)))_$(res_mode).jld2")
        @save outjld inst_name=String(instr) Ei=Ei tof_edges=tof_edges ψstride=ψstride ηstride=ηstride pixels_used=pix_used Cs=pred.Cs Cv=pred.Cv Cnorm=pred.Cnorm
        @info "Saved pixel×TOF arrays to $outjld"
    else
        @info "Set TOFTWIN_SAVE_PIXEL_TOF=1 to write Cs/Cv/Cnorm to disk (can be large)."
    end
end

if do_hist
    # -----------------------------
    # Plot grids
    # -----------------------------
    Q_cent = 0.5 .* (Q_edges[1:end-1] .+ Q_edges[2:end])
    ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

    kernel_grid = [kern(q, w) for q in Q_cent, w in ω_cent]
    raw_sum     = pred.Hraw.counts
    mean_map    = pred.Hmean.counts
    wt_map      = pred.Hwt.counts
    wt_log      = log10.(wt_map .+ 1.0)


    fig = Figure(size=(1400, 900))

    title_tag = (σt_us <= 0 || res_mode == "none") ? "no resolution" :
                (res_mode == "gh") ? "GH(order=$(gh_order)), σt=$(σt_us) µs" :
                "CDF(nsigma=$(nsigma)), σt=$(σt_us) µs"

    ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="Sunny kernel S(|Q|,ω) on TOFtwin grid")
    heatmap!(ax1, Q_cent, ω_cent, kernel_grid)

    ax2 = Axis(fig[1,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin predicted (raw SUM) — $(String(instr)) IDF ($title_tag)")
    heatmap!(ax2, Q_cent, ω_cent, raw_sum)

    ax3 = Axis(fig[2,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin weights log10(N+1) — $(String(instr)) IDF")
    heatmap!(ax3, Q_cent, ω_cent, wt_log)

    ax4 = Axis(fig[2,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin vanadium-normalized MEAN ($title_tag) + kernel contours")
    heatmap!(ax4, Q_cent, ω_cent, mean_map)
    #contour!(ax4, Q_cent, ω_cent, kernel_grid; color=:white, linewidth=1.0)
    if do_save
        outpng = joinpath(outdir, "demo_powder_resolution_switch_$(lowercase(String(instr)))_$(res_mode).png")
        save(outpng, fig)
        @info "Wrote $outpng"
    end
    display(fig)
end
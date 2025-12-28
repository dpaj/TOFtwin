using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin
using JLD2

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
# Instrument selection
# -----------------------------
function parse_instrument()
    s = uppercase(get(ENV, "TOFTWIN_INSTRUMENT", "SEQUOIA"))
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

# TOF range based on geometry (rough)
L2min = minimum(inst.L2)
L2max = maximum(inst.L2)
Ef_min, Ef_max = 1.0, Ei

tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ef_max)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, Ef_min)

ntof = parse(Int, get(ENV, "TOFTWIN_NTOF", "500"))
tof_edges = collect(range(tmin, tmax; length=ntof+1))

@info "TOF window (ms) = $(1e3*tmin) .. $(1e3*tmax)"
@info "Sunny kernel domain: Q∈($(kern.Q[1]), $(kern.Q[end]))  ω∈($(kern.ω[1]), $(kern.ω[end]))"

# -----------------------------
# Choose comparison grid (Q_edges, ω_edges)
# -----------------------------
nQbins = parse(Int, get(ENV, "TOFTWIN_NQBINS", "220"))
nωbins = parse(Int, get(ENV, "TOFTWIN_NWBINS", "240"))

Q_edges = collect(range(kern.Q[1], kern.Q[end]; length=nQbins+1))
ωlo = min(-2.0, kern.ω[1])
ωhi = max(Ei,  kern.ω[end])
ω_edges = collect(range(ωlo, ωhi; length=nωbins+1))

# -----------------------------
# Resolution: pick mode + parameters
# -----------------------------
res_mode = lowercase(get(ENV, "TOFTWIN_RES_MODE", "cdf"))  # none | gh | cdf
σt_us    = parse(Float64, get(ENV, "TOFTWIN_SIGMA_T_US", "100.0"))
σt       = σt_us * 1e-6

gh_order = parse(Int, get(ENV, "TOFTWIN_GH_ORDER", "3"))
nsigma   = parse(Float64, get(ENV, "TOFTWIN_NSIGMA", "4.0"))

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
    eps::Float64 = 1e-12)

    Cs = predict_pixel_tof(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        model=model, resolution=resolution
    )
    Cv = predict_pixel_tof(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        model=(Q,ω)->1.0, resolution=resolution
    )

    Hraw = reduce_pixel_tof_to_Qω_powder(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=Cs, Q_edges=Q_edges, ω_edges=ω_edges
    )

    Cnorm = normalize_by_vanadium(Cs, Cv; eps=eps)

    Hsum = reduce_pixel_tof_to_Qω_powder(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=Cnorm, Q_edges=Q_edges, ω_edges=ω_edges
    )

    W = zeros(size(Cnorm))
    W[Cv .> 0.0] .= 1.0
    Hwt = reduce_pixel_tof_to_Qω_powder(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=W, Q_edges=Q_edges, ω_edges=ω_edges
    )

    Hmean = Hist2D(Q_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)

    return (Hraw=Hraw, Hsum=Hsum, Hwt=Hwt, Hmean=Hmean)
end

pred = predict_powder_mean_Qω_with_resolution(inst;
    pixels=pix_used,
    Ei_meV=Ei,
    tof_edges_s=tof_edges,
    Q_edges=Q_edges,
    ω_edges=ω_edges,
    model=model,
    resolution=resolution
)

# -----------------------------
# Concept helper: approximate σt -> energy width at a representative pixel
# -----------------------------
if σt_us > 0
    p_ref = pix_used[clamp(Int(round(length(pix_used)/2)), 1, length(pix_used))]
    L2p = L2(inst, p_ref.id)
    @info "Energy-width estimate from timing resolution (representative pixel id=$(p_ref.id))"
    @info "  Using σ_ω ≈ |dω/dt| σt. (FWHM ≈ 2.355 σ_ω)"

    ω_test = [0.0, 2.0, 5.0, min(10.0, Ei-0.5)]
    for ω in ω_test
        Ef = Ei - ω
        Ef <= 0 && continue
        t = tof_from_EiEf(inst.L1, L2p, Ei, Ef)
        σω = abs(dω_dt(inst.L1, L2p, Ei, t)) * σt          # meV
        fwhm = 2.354820045 * σω
        @info "  ω=$(round(ω,digits=3)) meV: σ_ω=$(round(σω,digits=5)) meV, FWHM=$(round(fwhm,digits=5)) meV"
    end
end

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

outdir = joinpath(@__DIR__, "..", "out")
mkpath(outdir)

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
contour!(ax4, Q_cent, ω_cent, kernel_grid; color=:white, linewidth=1.0)

outpng = joinpath(outdir, "demo_powder_resolution_switch_$(lowercase(String(instr)))_$(res_mode).png")
save(outpng, fig)
@info "Wrote $outpng"
display(fig)

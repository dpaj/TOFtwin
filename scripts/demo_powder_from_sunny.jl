using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin
using JLD2

# Backend selection
backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "cairo"))
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

    return GridKernelPowder(Q, W, S; outside=outside)
end

# -----------------------------
# Input Sunny file
# -----------------------------
sunny_path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "sunny_powder_corh2o4.jld2")
@info "Sunny input file: $sunny_path"

kern = load_sunny_powder_jld2(sunny_path; outside=0.0)

# -----------------------------
# Instrument example
# -----------------------------
bank = bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
inst = Instrument(name="demo", L1=36.262, pixels=bank.pixels)

# Speed knob: subsample pixels while iterating
pix_used = sample_pixels(bank, AngularDecimate(3, 2))  # or AllPixels()

Ei = 12.0  # meV

# TOF range based on geometry (same as your demos)
L2min = minimum(inst.L2)
L2max = maximum(inst.L2)
Ef_min, Ef_max = 1.0, Ei

tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ef_max)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, Ef_min)

ntof = 500
tof_edges = collect(range(tmin, tmax; length=ntof+1))

@info "pixels used = $(length(pix_used))"
@info "TOF window (ms) = $(1e3*tmin) .. $(1e3*tmax)"
@info "Sunny kernel domain: Q∈($(kern.Q[1]), $(kern.Q[end]))  ω∈($(kern.ω[1]), $(kern.ω[end]))"

# -----------------------------
# Choose comparison grid (Q_edges, ω_edges)
# Keep it focused to the Sunny domain (and include ω<0 if you want)
# -----------------------------
nQbins = 220
nωbins = 240

Q_edges = collect(range(kern.Q[1], kern.Q[end]; length=nQbins+1))
ωlo = min(-2.0, kern.ω[1])            # include a little ω<0 for plotting
ωhi = max(Ei,  kern.ω[end])           # include up to Ei
ω_edges = collect(range(ωlo, ωhi; length=nωbins+1))

# -----------------------------
# Run TOFtwin forward prediction using the Sunny kernel
# -----------------------------
model = (q, w) -> kern(q, w)

pred = predict_powder_mean_Qω(inst;
    pixels=pix_used,
    Ei_meV=Ei,
    tof_edges_s=tof_edges,
    Q_edges=Q_edges,
    ω_edges=ω_edges,
    model=model
)

# -----------------------------
# Plot grids
# -----------------------------
Q_cent = 0.5 .* (Q_edges[1:end-1] .+ Q_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

kernel_grid = [kern(q, w) for q in Q_cent, w in ω_cent]  # (nQ, nω)
raw_sum     = pred.Hraw.counts
mean_map    = pred.Hmean.counts
wt_map      = pred.Hwt.counts
wt_log      = log10.(wt_map .+ 1.0)

outdir = joinpath(@__DIR__, "..", "out")
mkpath(outdir)

fig = Figure(size=(1400, 900))

ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="Sunny kernel S(|Q|,ω) on TOFtwin grid")
heatmap!(ax1, Q_cent, ω_cent, kernel_grid)

ax2 = Axis(fig[1,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin predicted (raw SUM)")
heatmap!(ax2, Q_cent, ω_cent, raw_sum)

ax3 = Axis(fig[2,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin weights log10(N+1)")
heatmap!(ax3, Q_cent, ω_cent, wt_log)

ax4 = Axis(fig[2,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin vanadium-normalized MEAN + kernel contours")
heatmap!(ax4, Q_cent, ω_cent, mean_map)
#contour!(ax4, Q_cent, ω_cent, kernel_grid)
contour!(ax4, Q_cent, ω_cent, kernel_grid; color=:white, linewidth=1.0)

save(joinpath(outdir, "demo_powder_from_sunny.png"), fig)
@info "Wrote $(joinpath(outdir, "demo_powder_from_sunny.png"))"
display(fig)

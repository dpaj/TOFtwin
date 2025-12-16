using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin

# choose backend like your viewer script
backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
if backend == "cairo"
    using CairoMakie
else
    using GLMakie
end

# ---------------- Geometry / instrument ----------------
bank = bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
inst = Instrument(name="demo", L1=36.262, pixels=bank.pixels)

# optional: subsample pixels for speed while iterating
pix_used = sample_pixels(bank, AngularDecimate(3, 2))   # tweak or use AllPixels()

Ei = 12.0  # meV

# ---------------- Choose TOF edges based on geometry ----------------
L2min = minimum(inst.L2)
L2max = maximum(inst.L2)

# pick an Ef range you care about (for TOF coverage)
Ef_min = 1.0
Ef_max = Ei

tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ef_max)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, Ef_min)

# build TOF edges (seconds)
ntof = 500
tof_edges = collect(range(tmin, tmax; length=ntof+1))

# ---------------- Q, ω binning ----------------
Q_edges = collect(range(0.0, 8.0; length=220))     # Å^-1
ω_edges = collect(range(-2.0, Ei; length=240))     # meV

# ---------------- Toy powder kernel ----------------
model = ToyModePowder(Δ=2.0, v=0.8, σ=0.25, amp=1.0)

# ---------------- Predict histogram ----------------
H = predict_hist_Qω_powder(inst;
    pixels=pix_used,
    Ei_meV=Ei,
    tof_edges_s=tof_edges,
    Q_edges=Q_edges,
    ω_edges=ω_edges,
    model=model
)

@info "pixels used = $(length(pix_used))"
@info "TOF window (ms) = $(1e3*tmin) .. $(1e3*tmax)"

# ---------------- Plot ----------------
Q_cent = 0.5 .* (Q_edges[1:end-1] .+ Q_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

fig = Figure(resolution=(1000, 800))
ax  = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)",
           title="Predicted powder intensity (toy kernel), Ei=$(Ei) meV")

heatmap!(ax, Q_cent, ω_cent, H.counts';)  # transpose for (x,y) orientation
save(get(ENV, "TOFTWIN_OUT", "pred_Qw.png"), fig)
display(fig)

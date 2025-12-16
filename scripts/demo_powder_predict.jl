using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin

backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
if backend == "cairo"
    using CairoMakie
else
    using GLMakie
end

# ---------------- Geometry / instrument ----------------
bank = bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
inst = Instrument(name="demo", L1=36.262, pixels=bank.pixels)

pix_used = sample_pixels(bank, AngularDecimate(3, 2))  # or AllPixels()
Ei = 12.0  # meV

# ---------------- Choose TOF edges based on geometry ----------------
L2min = minimum(inst.L2)
L2max = maximum(inst.L2)

Ef_min = 1.0
Ef_max = Ei

tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ef_max)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, Ef_min)

ntof = 500
tof_edges = collect(range(tmin, tmax; length=ntof+1))

# ---------------- Q, ω binning ----------------
Q_edges = collect(range(0.0, 8.0; length=220))     # Å^-1
ω_edges = collect(range(-2.0, Ei; length=240))     # meV

# ---------------- Toy powder kernel ----------------
model = ToyModePowder(Δ=2.0, v=0.8, σ=0.25, amp=1.0)

@info "pixels used = $(length(pix_used))"
@info "TOF window (ms) = $(1e3*tmin) .. $(1e3*tmax)"

# ---------------- Predict in detector×TOF ----------------
Cs = predict_pixel_tof(inst;
    pixels=pix_used,
    Ei_meV=Ei,
    tof_edges_s=tof_edges,
    model=model
)

Cv = predict_pixel_tof(inst;
    pixels=pix_used,
    Ei_meV=Ei,
    tof_edges_s=tof_edges,
    model=(Q,ω)->1.0   # "vanadium": flat in (Q,ω)
)

Cnorm = normalize_by_vanadium(Cs, Cv; eps=1e-12)

# ---------------- Reduce to (|Q|, ω) ----------------
Hraw = reduce_pixel_tof_to_Qω_powder(inst;
    pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
    C=Cs, Q_edges=Q_edges, ω_edges=ω_edges
)

# Sum of vanadium-normalized values in each (Q,ω) bin
Hvan_sum = reduce_pixel_tof_to_Qω_powder(inst;
    pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
    C=Cnorm, Q_edges=Q_edges, ω_edges=ω_edges
)

# Coverage/weights: count how many detector×TOF samples contribute to each (Q,ω) bin
W = zeros(size(Cnorm))
W[Cv .> 0.0] .= 1.0

Hwt = reduce_pixel_tof_to_Qω_powder(inst;
    pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
    C=W, Q_edges=Q_edges, ω_edges=ω_edges
)

# Mean vanadium-normalized intensity per (Q,ω) bin
ϵ = 1e-12
Hvan_mean = Hvan_sum.counts ./ (Hwt.counts .+ ϵ)

# ---------------- Plot (+ analytic kernel contours) ----------------
Q_cent = 0.5 .* (Q_edges[1:end-1] .+ Q_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])
kernel_grid = [model(q, w) for q in Q_cent, w in ω_cent]

fig = Figure(size=(1200, 800))

ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)",
           title="Predicted (raw detector×TOF → Qω, SUM)")
heatmap!(ax1, Q_cent, ω_cent, Hraw.counts)

ax2 = Axis(fig[1,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)",
           title="Vanadium-normalized (MEAN per Qω bin)")
heatmap!(ax2, Q_cent, ω_cent, Hvan_mean)
contour!(ax2, Q_cent, ω_cent, kernel_grid)

save("pred_Qw_vanadium_norm_mean.png", fig)
display(fig)

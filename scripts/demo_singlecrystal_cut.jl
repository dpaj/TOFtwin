using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin
using LinearAlgebra


backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "cairo"))
#backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
if backend == "gl"
    using GLMakie
else
    using CairoMakie
end

# Instrument
bank = bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
inst = Instrument(name="demo", L1=36.262, pixels=bank.pixels)
pix_used = sample_pixels(bank, AngularDecimate(3, 2))

Ei = 12.0
L2min, L2max = minimum(inst.L2), maximum(inst.L2)
tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ei)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, 1.0)
tof_edges = collect(range(tmin, tmax; length=500+1))

# Sample pose (identity for now; later plug in goniometer angles)
pose = Pose()  # assumes your Pose() default is identity; if not, set accordingly

# Lattice (example: cubic a=8.5031 Å like CoRh2O4 tutorial; swap for your crystal)
lp = LatticeParams(8.5031, 8.5031, 8.5031, 90.0, 90.0, 90.0)
recip = reciprocal_lattice(lp)

# A toy single-crystal kernel (depends on |Q_S|)
Δ, v, σ, amp = 2.0, 0.8, 0.25, 1.0
model = (Q_S, ω) -> amp * exp(-0.5*((ω - (Δ + v*norm(Q_S)))/σ)^2)

# Cut and binning: I(H,ω) with a window in K,L
H_edges = collect(range(-2.0, 2.0; length=260))
ω_edges = collect(range(-2.0, Ei; length=260))

pred = predict_cut_mean_Hω(inst;
    pixels=pix_used,
    Ei_meV=Ei,
    tof_edges_s=tof_edges,
    H_edges=H_edges,
    ω_edges=ω_edges,
    recip=recip,
    pose=pose,
    model=model,
    K_center=0.0, K_halfwidth=0.15,
    L_center=0.0, L_halfwidth=10.0
)

H_cent = 0.5 .* (H_edges[1:end-1] .+ H_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

fig = Figure(size=(1200, 800))
ax1 = Axis(fig[1,1], xlabel="H (r.l.u.)", ylabel="ω (meV)", title="Single-crystal cut: raw SUM")
heatmap!(ax1, H_cent, ω_cent, pred.Hraw.counts)

ax2 = Axis(fig[1,2], xlabel="H (r.l.u.)", ylabel="ω (meV)", title="Single-crystal cut: vanadium-normalized MEAN")
heatmap!(ax2, H_cent, ω_cent, pred.Hmean.counts)

save("out/demo_singlecrystal_cut.png", fig)
display(fig)

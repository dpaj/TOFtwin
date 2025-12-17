using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin

backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "cairo"))
if backend == "gl"
    using GLMakie
else
    using CairoMakie
end

const TVec3 = TOFtwin.Vec3

# Instrument
bank = bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
inst = Instrument(name="demo", L1=36.262, pixels=bank.pixels)
pix_used = sample_pixels(bank, AngularDecimate(3, 2))

Ei = 12.0
L2min, L2max = minimum(inst.L2), maximum(inst.L2)
tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ei)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, 1.0)
tof_edges = collect(range(tmin, tmax; length=600+1))

# Lattice (example cubic)
lp = LatticeParams(8.5031, 8.5031, 8.5031, 90.0, 90.0, 90.0)
recip = reciprocal_lattice(lp)

# HKL kernel (toy)
toy = ToyCosineHKL()#ToyGaussianDispHKL(q0=TVec3(1.0, 0.0, 0.0), Δ=2.0, v=3.0, σE=0.25, σQ=0.18, amp=1.0)
model_hkl = (hkl, ω) -> toy(hkl, ω)

# Goniometer scan: rotate sample about lab +y by angles (degrees)
# IMPORTANT: Q_S = R_SL * Q_L. If the sample is rotated by +θ in lab,
# then R_SL is typically rot_y(-θ) (inverse rotation).
angles_deg = 0:0.5:90
angles = deg2rad.(collect(angles_deg))
R_SL_list = [Ry(-θ) for θ in angles]

# Cut binning
H_edges = collect(range(-2.0, 2.0; length=260))
ω_edges = collect(range(-2.0, Ei; length=260))

pred = predict_cut_mean_Hω_hkl_scan(inst;
    pixels=pix_used,
    Ei_meV=Ei,
    tof_edges_s=tof_edges,
    H_edges=H_edges,
    ω_edges=ω_edges,
    recip=recip,
    R_SL_list=R_SL_list,
    model_hkl=model_hkl,
    K_center=0.0, K_halfwidth=0.1,
    L_center=0.0, L_halfwidth=10.0   # keep wide initially
)

H_cent = 0.5 .* (H_edges[1:end-1] .+ H_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

fig = Figure(size=(900, 700))
ax = Axis(fig[1,1], xlabel="H (r.l.u.)", ylabel="ω (meV)",
          title="Single-crystal I(H,ω) from HKL kernel + goniometer scan")
heatmap!(ax, H_cent, ω_cent, pred.Hmean.counts)

using Statistics

# same cut window you used in pred = ...
Kc, Kh = 0.0, 0.15
Lc, Lh = 0.0, 10.0

# coarse averaging over the K/L window (cheap but very effective)
nK, nL = 9, 9
Kgrid = collect(range(Kc - Kh, Kc + Kh; length=nK))
Lgrid = collect(range(Lc - Lh, Lc + Lh; length=nL))

model_Hω = [mean(model_hkl(TVec3(H, k, l), ω) for k in Kgrid, l in Lgrid)
            for H in H_cent, ω in ω_cent]   # size = (length(H_cent), length(ω_cent))

# scale model to match heatmap dynamic range (optional, just for visibility)
s = maximum(pred.Hmean.counts) / (maximum(model_Hω) + 1e-12)
model_Hω_scaled = s .* model_Hω

# overlay contours (high contrast)
contour!(ax, H_cent, ω_cent, model_Hω_scaled;
         color=:white, linewidth=1.0)


mkpath("out")
save("out/demo_singlecrystal_scan.png", fig)

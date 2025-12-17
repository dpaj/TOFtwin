using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin
using LinearAlgebra
using Statistics
using Base.Threads

backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
if backend == "gl"
    using GLMakie
else
    using CairoMakie
end

const TVec3 = TOFtwin.Vec3

# ---------------- Instrument ----------------
bank = bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
inst = Instrument(name="demo", L1=36.262, pixels=bank.pixels)

# Speed while iterating (use bank.pixels for full coverage)
pix_used = sample_pixels(bank, AngularDecimate(1, 1))
# pix_used = bank.pixels

Ei = 12.0
L2min, L2max = minimum(inst.L2), maximum(inst.L2)
tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ei)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, 1.0)
tof_edges = collect(range(tmin, tmax; length=600+1))

@info "pixels used = $(length(pix_used))"
@info "TOF window (ms) = $(1e3*tmin) .. $(1e3*tmax)"
@info "Threads = $(nthreads())"

# ---------------- Lattice / reciprocal ----------------
lp = LatticeParams(8.5031, 8.5031, 8.5031, 90.0, 90.0, 90.0)
recip = reciprocal_lattice(lp)

# ---------------- Sample alignment (u,v) ----------------
u_hkl = TVec3(0.5, 0.5, 0.1)
v_hkl = TVec3(0.0, 1.0, -0.1)
aln = alignment_from_uv(recip; u_hkl=u_hkl, v_hkl=v_hkl)

# ---------------- Goniometer scan ----------------
angles_deg = 0:5:180
angles = deg2rad.(collect(angles_deg))

zero_offset_deg = 0.0
R_SL_list = goniometer_scan_RSL_y(angles; zero_offset_rad=deg2rad(zero_offset_deg))

# ---------------- Cut definition ----------------
H_edges = collect(range(-2.0, 2.0; length=260))
ω_edges = collect(range(-2.0, Ei; length=260))
H_cent = 0.5 .* (H_edges[1:end-1] .+ H_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

Kc, Kh = 0.0, 0.10
Lc, Lh = 0.0, 0.10

# ---------------- HKL kernel ----------------
q0  = TVec3(1.0, 0.0, 0.0)
Δ   = 2.0
v   = 3.0
σE  = 0.25
σQ  = 0.25
amp = 1.0

model_hkl = function(hkl::TVec3, ω::Float64)
    dq = norm(hkl - q0)
    ω0 = Δ + v*dq
    return amp * exp(-0.5*(dq/σQ)^2) * exp(-0.5*((ω - ω0)/σE)^2)
end

# ---------------- Predict ----------------
@info "Scanning WITH u,v alignment..."
pred = predict_cut_mean_Hω_hkl_scan_aligned(inst;
    pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
    H_edges=H_edges, ω_edges=ω_edges,
    aln=aln, R_SL_list=R_SL_list, model_hkl=model_hkl,
    K_center=Kc, K_halfwidth=Kh, L_center=Lc, L_halfwidth=Lh
)

# ---------------- Optional: overlay kernel contours (K/L-averaged) ----------------
nK, nL = 9, 9
Kgrid = collect(range(Kc - Kh, Kc + Kh; length=nK))
Lgrid = collect(range(Lc - Lh, Lc + Lh; length=nL))
model_Hω = [mean(model_hkl(TVec3(H,k,l), ω) for k in Kgrid, l in Lgrid)
            for H in H_cent, ω in ω_cent]

scaled_for(pred_counts, model_grid) =
    maximum(pred_counts) / (maximum(model_grid) + 1e-12) .* model_grid

# ---------------- Plot ----------------
fig = Figure(size=(1100, 800))

ax1 = Axis(fig[1,1], xlabel="H (r.l.u.)", ylabel="ω (meV)",
           title="Aligned (u,v) mean-per-bin   (zero_offset=$(zero_offset_deg)°)")
heatmap!(ax1, H_cent, ω_cent, pred.Hmean.counts)
contour!(ax1, H_cent, ω_cent, scaled_for(pred.Hmean.counts, model_Hω);
         color=:white, linewidth=1.0)

ax2 = Axis(fig[1,2], xlabel="H (r.l.u.)", ylabel="ω (meV)",
           title="Coverage weights (aligned)  log10(w+1)")
heatmap!(ax2, H_cent, ω_cent, log10.(pred.Hwt.counts .+ 1.0))

mkpath("out")
save("out/demo_singlecrystal_alignment_scan.png", fig)
display(fig)

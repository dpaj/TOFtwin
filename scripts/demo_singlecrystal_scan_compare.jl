using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin
using LinearAlgebra
using Statistics

backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "cairo"))
if backend == "gl"
    using GLMakie
else
    using CairoMakie
end

const TVec3 = TOFtwin.Vec3

# ---------------- Instrument ----------------
bank = bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
inst = Instrument(name="demo", L1=36.262, pixels=bank.pixels)

# Start with more pixels if you’re debugging striping:
# pix_used = bank.pixels
pix_used = sample_pixels(bank, AngularDecimate(3, 2))

Ei = 12.0
L2min, L2max = minimum(inst.L2), maximum(inst.L2)
tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ei)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, 1.0)
tof_edges = collect(range(tmin, tmax; length=600+1))

# ---------------- Lattice ----------------
lp = LatticeParams(8.5031, 8.5031, 8.5031, 90.0, 90.0, 90.0)
recip = reciprocal_lattice(lp)

# ---------------- HKL kernel ----------------
toy = ToyCosineHKL()
# toy = ToyGaussianDispHKL(q0=TVec3(1.0,0.0,0.0), Δ=2.0, v=3.0, σE=0.25, σQ=0.18, amp=1.0)
model_hkl = (hkl, ω) -> toy(hkl, ω)

# ---------------- Goniometer scan ----------------
angles_deg = 0:0.5:90
angles = deg2rad.(collect(angles_deg))
R_SL_list = [Ry(-θ) for θ in angles]   # Q_S = R_SL * Q_L

# ---------------- Cut definition ----------------
H_edges = collect(range(-2.0, 2.0; length=260))
ω_edges = collect(range(-2.0, Ei; length=260))
H_cent = 0.5 .* (H_edges[1:end-1] .+ H_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

Kc, Kh = 0.0, 0.10
Lc, Lh = 0.0, 0.10

# ---------------- Helper: reduce detector×TOF with a given R_SL ----------------
function reduce_pixel_tof_to_Hω_cut_RSL(inst::Instrument;
    pixels::Vector{TOFtwin.DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    C::AbstractMatrix,
    recip::TOFtwin.ReciprocalLattice,
    R_SL,
    H_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    K_center::Float64,
    K_halfwidth::Float64,
    L_center::Float64,
    L_halfwidth::Float64)

    H = TOFtwin.Hist2D(H_edges, ω_edges)
    n_tof = length(tof_edges_s) - 1

    for p in pixels
        L2p = TOFtwin.L2(inst, p.id)
        for it in 1:n_tof
            w = C[p.id, it]
            w == 0.0 && continue

            t  = 0.5*(tof_edges_s[it] + tof_edges_s[it+1])
            Ef = try
                TOFtwin.Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Q_L, ω = TOFtwin.Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            Q_S = R_SL * Q_L
            hkl = TOFtwin.hkl_from_Q(recip, Q_S)

            Hh, Kk, Ll = hkl
            (abs(Kk - K_center) > K_halfwidth) && continue
            (abs(Ll - L_center) > L_halfwidth) && continue

            TOFtwin.deposit_bilinear!(H, Hh, ω, w)
        end
    end
    return H
end

# ---------------- Method (3): explicit detector×TOF vanadium then mean-per-bin ----------------
function predict_cut_detvan_mean_Hω_hkl_scan(inst::Instrument;
    pixels::Vector{TOFtwin.DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    H_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    recip::TOFtwin.ReciprocalLattice,
    R_SL_list::Vector,
    model_hkl,
    eps::Float64=1e-12,
    K_center::Float64=0.0,
    K_halfwidth::Float64=0.1,
    L_center::Float64=0.0,
    L_halfwidth::Float64=0.1)

    Hsum = TOFtwin.Hist2D(H_edges, ω_edges)
    Hwt  = TOFtwin.Hist2D(H_edges, ω_edges)

    n_pix = length(inst.pixels)
    n_tof = length(tof_edges_s) - 1

    for R_SL in R_SL_list
        Cs = zeros(Float64, n_pix, n_tof)
        Cv = zeros(Float64, n_pix, n_tof)

        for p in pixels
            L2p = TOFtwin.L2(inst, p.id)
            for it in 1:n_tof
                t0 = tof_edges_s[it]; t1 = tof_edges_s[it+1]
                t  = 0.5*(t0 + t1)
                dt = t1 - t0

                Ef = try
                    TOFtwin.Ef_from_tof(inst.L1, L2p, Ei_meV, t)
                catch
                    continue
                end
                (Ef <= 0 || Ef > Ei_meV) && continue

                Q_L, ω = TOFtwin.Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
                Q_S = R_SL * Q_L
                hkl = TOFtwin.hkl_from_Q(recip, Q_S)

                jac = abs(TOFtwin.dω_dt(inst.L1, L2p, Ei_meV, t)) * dt
                Cs[p.id, it] += model_hkl(hkl, ω) * jac
                Cv[p.id, it] += 1.0 * jac
            end
        end

        Cnorm = Cs ./ (Cv .+ eps)
        W = zeros(Float64, n_pix, n_tof)
        W[Cv .> 0.0] .= 1.0

        Hsum_pose = reduce_pixel_tof_to_Hω_cut_RSL(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            C=Cnorm, recip=recip, R_SL=R_SL,
            H_edges=H_edges, ω_edges=ω_edges,
            K_center=K_center, K_halfwidth=K_halfwidth,
            L_center=L_center, L_halfwidth=L_halfwidth
        )
        Hwt_pose = reduce_pixel_tof_to_Hω_cut_RSL(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            C=W, recip=recip, R_SL=R_SL,
            H_edges=H_edges, ω_edges=ω_edges,
            K_center=K_center, K_halfwidth=K_halfwidth,
            L_center=L_center, L_halfwidth=L_halfwidth
        )

        Hsum.counts .+= Hsum_pose.counts
        Hwt.counts  .+= Hwt_pose.counts
    end

    Hmean = TOFtwin.Hist2D(H_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)
    return (Hsum=Hsum, Hwt=Hwt, Hmean=Hmean)
end

# ---------------- Compute all 3 methods ----------------

# (1) ratio-of-sums / Jacobian weighted
pred_ratio = TOFtwin.predict_cut_weightedmean_Hω_hkl_scan(inst;
    pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
    H_edges=H_edges, ω_edges=ω_edges,
    recip=recip, R_SL_list=R_SL_list, model_hkl=model_hkl,
    K_center=Kc, K_halfwidth=Kh, L_center=Lc, L_halfwidth=Lh
)

# (2) mean-per-bin directly in (H,ω)
pred_mean = TOFtwin.predict_cut_mean_Hω_hkl_scan(inst;
    pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
    H_edges=H_edges, ω_edges=ω_edges,
    recip=recip, R_SL_list=R_SL_list, model_hkl=model_hkl,
    K_center=Kc, K_halfwidth=Kh, L_center=Lc, L_halfwidth=Lh
)

# (3) explicit detector×TOF vanadium then reduce + mean-per-bin
pred_detvan = predict_cut_detvan_mean_Hω_hkl_scan(inst;
    pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
    H_edges=H_edges, ω_edges=ω_edges,
    recip=recip, R_SL_list=R_SL_list, model_hkl=model_hkl,
    K_center=Kc, K_halfwidth=Kh, L_center=Lc, L_halfwidth=Lh
)

# ---------------- Optional: analytic overlay (K/L-averaged model) ----------------
nK, nL = 9, 9
Kgrid = collect(range(Kc - Kh, Kc + Kh; length=nK))
Lgrid = collect(range(Lc - Lh, Lc + Lh; length=nL))
model_Hω = [mean(model_hkl(TVec3(H, k, l), ω) for k in Kgrid, l in Lgrid)
            for H in H_cent, ω in ω_cent]

# scale for contours so they’re visible on each panel
function scaled_for(pred_counts, model_grid)
    s = maximum(pred_counts) / (maximum(model_grid) + 1e-12)
    return s .* model_grid
end

# ---------------- Plot ----------------
fig = Figure(size=(1200, 900))

ax1 = Axis(fig[1,1], xlabel="H (r.l.u.)", ylabel="ω (meV)", title="(1) Ratio-of-sums: Hnum/Hden")
heatmap!(ax1, H_cent, ω_cent, pred_ratio.Hmean.counts)
contour!(ax1, H_cent, ω_cent, scaled_for(pred_ratio.Hmean.counts, model_Hω); color=:white, linewidth=1.0)

ax2 = Axis(fig[1,2], xlabel="H (r.l.u.)", ylabel="ω (meV)", title="(2) Mean-per-bin: Hsum/Hwt")
heatmap!(ax2, H_cent, ω_cent, pred_mean.Hmean.counts)
contour!(ax2, H_cent, ω_cent, scaled_for(pred_mean.Hmean.counts, model_Hω); color=:white, linewidth=1.0)

ax3 = Axis(fig[2,1], xlabel="H (r.l.u.)", ylabel="ω (meV)", title="(3) Detector×TOF vanadium → reduce → mean")
heatmap!(ax3, H_cent, ω_cent, pred_detvan.Hmean.counts)
contour!(ax3, H_cent, ω_cent, scaled_for(pred_detvan.Hmean.counts, model_Hω); color=:white, linewidth=1.0)

ax4 = Axis(fig[2,2], xlabel="H (r.l.u.)", ylabel="ω (meV)", title="Coverage (weights) for mean-per-bin")
heatmap!(ax4, H_cent, ω_cent, log10.(pred_mean.Hwt.counts .+ 1.0))

mkpath("out")
save("out/demo_singlecrystal_scan_compare.png", fig)

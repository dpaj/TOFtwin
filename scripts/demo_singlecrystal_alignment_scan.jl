using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin
using LinearAlgebra
using Statistics
using Base.Threads

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

pix_used = sample_pixels(bank, AngularDecimate(3, 2))   # speed while iterating
Ei = 12.0

L2min, L2max = minimum(inst.L2), maximum(inst.L2)
tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ei)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, 1.0)
tof_edges = collect(range(tmin, tmax; length=600+1))
n_tof = length(tof_edges) - 1

@info "pixels used = $(length(pix_used))"
@info "TOF window (ms) = $(1e3*tmin) .. $(1e3*tmax)"
@info "Threads = $(nthreads())   (set JULIA_NUM_THREADS for more)"

# ---------------- Lattice / reciprocal ----------------
lp = LatticeParams(8.5031, 8.5031, 8.5031, 90.0, 90.0, 90.0)
recip = reciprocal_lattice(lp)

# ---------------- Sample alignment (u,v) ----------------
# u_hkl and v_hkl are in r.l.u. (crystal directions defining the scattering plane).
# Defaults in alignment_from_uv map u -> +x_S, v -> +z_S (horizontal scattering plane).
u_hkl = TVec3(0.5, 0.5, 0.1)
v_hkl = TVec3(0.0, 1.0, -0.1)

aln = alignment_from_uv(recip; u_hkl=u_hkl, v_hkl=v_hkl)

# A "no-alignment" reference: UB_S = B (assumes sample frame == crystal reciprocal frame)
B = B_matrix(recip)
aln_identity = SampleAlignment_from_UB(B)

# ---------------- Goniometer scan (+ variable zero) ----------------
angles_deg = 0:5:180
angles = deg2rad.(collect(angles_deg))

# encoder zero offset (degrees): change this to match Mantid/SNS "zero" choices
zero_offset_deg = 0.0
R_SL_list = goniometer_scan_RSL_y(angles; zero_offset_rad=deg2rad(zero_offset_deg))

# ---------------- Cut definition ----------------
H_edges = collect(range(-2.0, 2.0; length=260))
ω_edges = collect(range(-2.0, Ei; length=260))
H_cent = 0.5 .* (H_edges[1:end-1] .+ H_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

Kc, Kh = 0.0, 0.10
Lc, Lh = 0.0, 0.10

# ---------------- A simple HKL kernel (local to this demo) ----------------
# A localized dispersing blob around q0 in HKL space.
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

# ---------------- Precompute (pixel,TOF) kinematics once ----------------
np = length(pix_used)
QL    = Matrix{TVec3}(undef, np, n_tof)
ωL    = Matrix{Float64}(undef, np, n_tof)
jacdt = Matrix{Float64}(undef, np, n_tof)
valid = BitMatrix(undef, np, n_tof)

L2_used = [TOFtwin.L2(inst, p.id) for p in pix_used]
rL_used = [p.r_L for p in pix_used]

@info "Precomputing kinematics: np=$np n_tof=$n_tof"
for ip in 1:np
    L2p = L2_used[ip]
    rL  = rL_used[ip]
    for it in 1:n_tof
        t0 = tof_edges[it]; t1 = tof_edges[it+1]
        t  = 0.5*(t0 + t1)
        dt = t1 - t0

        Ef = try
            TOFtwin.Ef_from_tof(inst.L1, L2p, Ei, t)
        catch
            valid[ip,it] = false
            continue
        end
        if !(Ef > 0 && Ef <= Ei)
            valid[ip,it] = false
            continue
        end

        Q, ω = TOFtwin.Qω_from_pixel(rL, Ei, Ef; r_samp_L=inst.r_samp_L)
        QL[ip,it] = Q
        ωL[ip,it] = ω
        jacdt[ip,it] = abs(TOFtwin.dω_dt(inst.L1, L2p, Ei, t)) * dt
        valid[ip,it] = true
    end
end

# ---------------- Scan + reduce using alignment ----------------
function scan_mean_Hω_with_alignment(aln::TOFtwin.SampleAlignment)
    Hsum_thr = [TOFtwin.Hist2D(H_edges, ω_edges) for _ in 1:nthreads()]
    Hwt_thr  = [TOFtwin.Hist2D(H_edges, ω_edges) for _ in 1:nthreads()]

    @threads for ia in eachindex(R_SL_list)
        tid = threadid()
        R = R_SL_list[ia]
        Hsum = Hsum_thr[tid]
        Hwt  = Hwt_thr[tid]

        for ip in 1:np, it in 1:n_tof
            valid[ip,it] || continue

            Q_S = R * QL[ip,it]
            ω   = ωL[ip,it]

            hkl = TOFtwin.hkl_from_Q_S(aln, Q_S)
            Hh, Kk, Ll = hkl

            (abs(Kk - Kc) > Kh) && continue
            (abs(Ll - Lc) > Lh) && continue

            TOFtwin.deposit_bilinear!(Hsum, Hh, ω, model_hkl(hkl, ω))
            TOFtwin.deposit_bilinear!(Hwt,  Hh, ω, 1.0)
        end
    end

    Hsum = TOFtwin.Hist2D(H_edges, ω_edges)
    Hwt  = TOFtwin.Hist2D(H_edges, ω_edges)
    for k in 1:nthreads()
        Hsum.counts .+= Hsum_thr[k].counts
        Hwt.counts  .+= Hwt_thr[k].counts
    end

    Hmean = TOFtwin.Hist2D(H_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ 1e-12)
    return (Hsum=Hsum, Hwt=Hwt, Hmean=Hmean)
end

@info "Scanning WITH u,v alignment..."
pred_aln = scan_mean_Hω_with_alignment(aln)

@info "Scanning with IDENTITY mapping (no alignment, for comparison)..."
pred_id  = scan_mean_Hω_with_alignment(aln_identity)

# ---------------- Optional: overlay model contours (K/L-averaged) ----------------
nK, nL = 9, 9
Kgrid = collect(range(Kc - Kh, Kc + Kh; length=nK))
Lgrid = collect(range(Lc - Lh, Lc + Lh; length=nL))
model_Hω = [mean(model_hkl(TVec3(H,k,l), ω) for k in Kgrid, l in Lgrid)
            for H in H_cent, ω in ω_cent]

function scaled_for(pred_counts, model_grid)
    maximum(pred_counts) / (maximum(model_grid) + 1e-12) .* model_grid
end

# ---------------- Plot ----------------
fig = Figure(size=(1200, 900))

ax1 = Axis(fig[1,1], xlabel="H (r.l.u.)", ylabel="ω (meV)",
           title="Aligned (u,v) mean-per-bin   (zero_offset=$(zero_offset_deg)°)")
heatmap!(ax1, H_cent, ω_cent, pred_aln.Hmean.counts)
contour!(ax1, H_cent, ω_cent, scaled_for(pred_aln.Hmean.counts, model_Hω);
         color=:white, linewidth=1.0)

ax2 = Axis(fig[1,2], xlabel="H (r.l.u.)", ylabel="ω (meV)",
           title="Identity mapping (no alignment) mean-per-bin")
heatmap!(ax2, H_cent, ω_cent, pred_id.Hmean.counts)
contour!(ax2, H_cent, ω_cent, scaled_for(pred_id.Hmean.counts, model_Hω);
         color=:white, linewidth=1.0)

ax3 = Axis(fig[2,1], xlabel="H (r.l.u.)", ylabel="ω (meV)",
           title="Coverage weights (aligned)  log10(w+1)")
heatmap!(ax3, H_cent, ω_cent, log10.(pred_aln.Hwt.counts .+ 1.0))

ax4 = Axis(fig[2,2], xlabel="H (r.l.u.)", ylabel="ω (meV)",
           title="Coverage weights (identity) log10(w+1)")
heatmap!(ax4, H_cent, ω_cent, log10.(pred_id.Hwt.counts .+ 1.0))

mkpath("out")
save("out/demo_singlecrystal_alignment_scan.png", fig)
display(fig)

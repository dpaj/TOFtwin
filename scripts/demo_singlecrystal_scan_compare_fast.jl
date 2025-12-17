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
pix_used = sample_pixels(bank, AngularDecimate(6, 4))   # debug speed

#pix_used = bank.pixels

Ei = 12.0
L2min, L2max = minimum(inst.L2), maximum(inst.L2)
tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ei)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, 1.0)
tof_edges = collect(range(tmin, tmax; length=600+1))
n_tof = length(tof_edges) - 1

# ---------------- Lattice ----------------
lp = LatticeParams(8.5031, 8.5031, 8.5031, 90.0, 90.0, 90.0)
recip = reciprocal_lattice(lp)

# ---------------- HKL kernel ----------------
toy = ToyCosineHKL()
model_hkl = (hkl, ω) -> toy(hkl, ω)

# ---------------- Goniometer scan ----------------
angles_deg = 0:0.5:90
angles = deg2rad.(collect(angles_deg))
R_SL_list = [Ry(-θ) for θ in angles]

# ---------------- Cut definition ----------------
H_edges = collect(range(-2.0, 2.0; length=260))
ω_edges = collect(range(-2.0, Ei; length=260))
H_cent = 0.5 .* (H_edges[1:end-1] .+ H_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

Kc, Kh = 0.0, 0.10
Lc, Lh = 0.0, 0.10

# ---------------- Precompute pixel×TOF kinematics (Q_L, ω, jac*dt) ----------------
np = length(pix_used)
QL = Matrix{TVec3}(undef, np, n_tof)
ωL = Matrix{Float64}(undef, np, n_tof)
jacdt = Matrix{Float64}(undef, np, n_tof)
valid = BitMatrix(undef, np, n_tof)

L2_used = [TOFtwin.L2(inst, p.id) for p in pix_used]
rL_used = [p.r_L for p in pix_used]

@info "Precomputing kinematics: np=$np, n_tof=$n_tof"
for ip in 1:np
    L2p = L2_used[ip]
    rL = rL_used[ip]
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

# ---------------- Threaded scan: build histograms per thread, then reduce ----------------
function scan_ratio_of_sums()
    Hnum_thr = [TOFtwin.Hist2D(H_edges, ω_edges) for _ in 1:nthreads()]
    Hden_thr = [TOFtwin.Hist2D(H_edges, ω_edges) for _ in 1:nthreads()]

    @threads for ia in eachindex(R_SL_list)
        tid = threadid()
        R = R_SL_list[ia]
        Hnum = Hnum_thr[tid]
        Hden = Hden_thr[tid]

        for ip in 1:np, it in 1:n_tof
            valid[ip,it] || continue
            Q_S = R * QL[ip,it]
            ω   = ωL[ip,it]
            hkl = TOFtwin.hkl_from_Q(recip, Q_S)
            Hh, Kk, Ll = hkl
            (abs(Kk - Kc) > Kh) && continue
            (abs(Ll - Lc) > Lh) && continue
            w = jacdt[ip,it]
            TOFtwin.deposit_bilinear!(Hnum, Hh, ω, model_hkl(hkl, ω) * w)
            TOFtwin.deposit_bilinear!(Hden, Hh, ω, w)
        end
    end

    Hnum = TOFtwin.Hist2D(H_edges, ω_edges)
    Hden = TOFtwin.Hist2D(H_edges, ω_edges)
    for k in 1:nthreads()
        Hnum.counts .+= Hnum_thr[k].counts
        Hden.counts .+= Hden_thr[k].counts
    end
    Hmean = TOFtwin.Hist2D(H_edges, ω_edges)
    Hmean.counts .= Hnum.counts ./ (Hden.counts .+ 1e-12)
    return (Hnum=Hnum, Hden=Hden, Hmean=Hmean)
end

function scan_mean_per_bin()
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
            hkl = TOFtwin.hkl_from_Q(recip, Q_S)
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

@info "Running ratio-of-sums..."
pred_ratio = scan_ratio_of_sums()

@info "Running mean-per-bin..."
pred_mean = scan_mean_per_bin()

# ---------------- Plot ----------------
fig = Figure(size=(1200, 900))

ax1 = Axis(fig[1,1], xlabel="H (r.l.u.)", ylabel="ω (meV)", title="(1) Ratio-of-sums: Hnum/Hden")
heatmap!(ax1, H_cent, ω_cent, pred_ratio.Hmean.counts)

ax2 = Axis(fig[1,2], xlabel="H (r.l.u.)", ylabel="ω (meV)", title="(2) Mean-per-bin: Hsum/Hwt")
heatmap!(ax2, H_cent, ω_cent, pred_mean.Hmean.counts)

ax3 = Axis(fig[2,2], xlabel="H (r.l.u.)", ylabel="ω (meV)", title="Coverage weights (log10)")
heatmap!(ax3, H_cent, ω_cent, log10.(pred_mean.Hwt.counts .+ 1.0))

mkpath("out")
save("out/demo_singlecrystal_scan_compare_fast.png", fig)

using LinearAlgebra

"""
Reduce detector×TOF matrix into a single-crystal cut I(H, ω) by:
- mapping pixel×TOF -> (Q_L, ω)
- rotating Q_L -> Q_S (sample frame) using pose
- converting Q_S -> (H,K,L) using recip
- selecting a window in K and L
- depositing into (H, ω) histogram
"""
function reduce_pixel_tof_to_Hω_cut(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    C::AbstractMatrix,
    recip::ReciprocalLattice,
    pose::Pose,
    H_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    K_center::Float64=0.0,
    K_halfwidth::Float64=0.1,
    L_center::Float64=0.0,
    L_halfwidth::Float64=0.1)

    H = Hist2D(H_edges, ω_edges)
    n_tof = length(tof_edges_s) - 1

    # lab -> sample rotation (vectors: rotation only)
    R_SL = T_SL(pose).R

    for p in pixels
        L2p = L2(inst, p.id)
        for it in 1:n_tof
            w = C[p.id, it]
            w == 0.0 && continue

            t0 = tof_edges_s[it]
            t1 = tof_edges_s[it+1]
            t  = 0.5*(t0 + t1)

            Ef = try
                Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Q_L, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            Q_S = R_SL * Q_L

            hkl = hkl_from_Q(recip, Q_S)
            Hh, Kk, Ll = hkl[1], hkl[2], hkl[3]

            (abs(Kk - K_center) > K_halfwidth) && continue
            (abs(Ll - L_center) > L_halfwidth) && continue

            deposit_bilinear!(H, Hh, ω, w)
        end
    end

    return H
end

"""
Full conventional single-crystal cut workflow:
model(Q_S, ω) -> detector×TOF -> vanadium normalize -> cut to (H,ω)
Return (Hraw, Hsum, Hwt, Hmean).
"""
function predict_cut_mean_Hω(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    H_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    recip::ReciprocalLattice,
    pose::Pose,
    model,
    eps::Float64=1e-12,
    K_center::Float64=0.0,
    K_halfwidth::Float64=0.1,
    L_center::Float64=0.0,
    L_halfwidth::Float64=0.1)

    # Predict in detector×TOF (note: model expects Q_S,ω — we compute Q_S inside)
    # We do this by building Cs,Cv directly with a loop here (so we can use Q_S).
    n_pix = length(inst.pixels)
    n_tof = length(tof_edges_s) - 1
    Cs = zeros(Float64, n_pix, n_tof)
    Cv = zeros(Float64, n_pix, n_tof)

    R_SL = T_SL(pose).R

    for p in pixels
        L2p = L2(inst, p.id)
        for it in 1:n_tof
            t0 = tof_edges_s[it]
            t1 = tof_edges_s[it+1]
            t  = 0.5*(t0 + t1)
            dt = t1 - t0

            Ef = try
                Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Q_L, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            Q_S = R_SL * Q_L

            # detector×TOF expected counts ~ S * dω = S * |dω/dt| dt
            jac = abs(dω_dt(inst.L1, L2p, Ei_meV, t)) * dt
            Cs[p.id, it] += model(Q_S, ω) * jac
            Cv[p.id, it] += 1.0 * jac
        end
    end

    Cnorm = normalize_by_vanadium(Cs, Cv; eps=eps)

    # Raw cut (sum)
    Hraw = reduce_pixel_tof_to_Hω_cut(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=Cs, recip=recip, pose=pose,
        H_edges=H_edges, ω_edges=ω_edges,
        K_center=K_center, K_halfwidth=K_halfwidth,
        L_center=L_center, L_halfwidth=L_halfwidth
    )

    # Vanadium-normalized (sum)
    Hsum = reduce_pixel_tof_to_Hω_cut(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=Cnorm, recip=recip, pose=pose,
        H_edges=H_edges, ω_edges=ω_edges,
        K_center=K_center, K_halfwidth=K_halfwidth,
        L_center=L_center, L_halfwidth=L_halfwidth
    )

    # Weights/coverage: 1 where Cv>0
    W = zeros(size(Cnorm))
    W[Cv .> 0.0] .= 1.0
    Hwt = reduce_pixel_tof_to_Hω_cut(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=W, recip=recip, pose=pose,
        H_edges=H_edges, ω_edges=ω_edges,
        K_center=K_center, K_halfwidth=K_halfwidth,
        L_center=L_center, L_halfwidth=L_halfwidth
    )

    # Mean per (H,ω) bin
    Hmean = Hist2D(H_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)

    return (Hraw=Hraw, Hsum=Hsum, Hwt=Hwt, Hmean=Hmean)
end

"""
Predict a single-crystal cut I(H,ω) using an HKL-based kernel model_hkl(hkl,ω),
for one orientation R_SL (lab->sample rotation).

Returns numerator/denominator and their ratio (weighted mean).
"""
function predict_cut_weightedmean_Hω_hkl(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    H_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    recip::ReciprocalLattice,
    R_SL,                    # 3×3 rotation matrix mapping Q_L -> Q_S
    model_hkl,
    eps::Float64=1e-12,
    K_center::Float64=0.0,
    K_halfwidth::Float64=0.15,
    L_center::Float64=0.0,
    L_halfwidth::Float64=10.0)

    Hnum = Hist2D(H_edges, ω_edges)  # sum(model * jac*dt)
    Hden = Hist2D(H_edges, ω_edges)  # sum(jac*dt)

    n_tof = length(tof_edges_s) - 1

    for p in pixels
        L2p = L2(inst, p.id)
        for it in 1:n_tof
            t0 = tof_edges_s[it]
            t1 = tof_edges_s[it+1]
            t  = 0.5*(t0 + t1)
            dt = (t1 - t0)

            Ef = try
                Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Q_L, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            Q_S = R_SL * Q_L
            hkl = hkl_from_Q(recip, Q_S)

            Hh, Kk, Ll = hkl[1], hkl[2], hkl[3]
            (abs(Kk - K_center) > K_halfwidth) && continue
            (abs(Ll - L_center) > L_halfwidth) && continue

            jac = abs(dω_dt(inst.L1, L2p, Ei_meV, t)) * dt

            deposit_bilinear!(Hnum, Hh, ω, model_hkl(hkl, ω) * jac)
            deposit_bilinear!(Hden, Hh, ω, jac)
        end
    end

    Hmean = Hist2D(H_edges, ω_edges)
    Hmean.counts .= Hnum.counts ./ (Hden.counts .+ eps)

    return (Hnum=Hnum, Hden=Hden, Hmean=Hmean)
end

"""
Same as above, but for a scan over many orientations (goniometer angles).
Accumulates numerator/denominator across poses, then forms the weighted mean.
"""
function predict_cut_weightedmean_Hω_hkl_scan(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    H_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    recip::ReciprocalLattice,
    R_SL_list::Vector,
    model_hkl,
    eps::Float64=1e-12,
    K_center::Float64=0.0,
    K_halfwidth::Float64=0.15,
    L_center::Float64=0.0,
    L_halfwidth::Float64=10.0)

    Hnum = Hist2D(H_edges, ω_edges)
    Hden = Hist2D(H_edges, ω_edges)

    for R_SL in R_SL_list
        tmp = predict_cut_weightedmean_Hω_hkl(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            H_edges=H_edges, ω_edges=ω_edges,
            recip=recip, R_SL=R_SL, model_hkl=model_hkl,
            eps=eps,
            K_center=K_center, K_halfwidth=K_halfwidth,
            L_center=L_center, L_halfwidth=L_halfwidth
        )
        Hnum.counts .+= tmp.Hnum.counts
        Hden.counts .+= tmp.Hden.counts
    end

    Hmean = Hist2D(H_edges, ω_edges)
    Hmean.counts .= Hnum.counts ./ (Hden.counts .+ eps)

    return (Hnum=Hnum, Hden=Hden, Hmean=Hmean)
end

function predict_cut_mean_Hω_hkl(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    H_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    recip::ReciprocalLattice,
    R_SL,                    # Q_S = R_SL * Q_L
    model_hkl,
    eps::Float64=1e-12,
    K_center::Float64=0.0,
    K_halfwidth::Float64=0.15,
    L_center::Float64=0.0,
    L_halfwidth::Float64=10.0)

    Hsum = Hist2D(H_edges, ω_edges)   # sum of normalized samples
    Hwt  = Hist2D(H_edges, ω_edges)   # number of samples (weights)
    n_tof = length(tof_edges_s) - 1

    for p in pixels
        L2p = L2(inst, p.id)
        for it in 1:n_tof
            t = 0.5*(tof_edges_s[it] + tof_edges_s[it+1])

            Ef = try
                Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Q_L, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            Q_S = R_SL * Q_L
            hkl = hkl_from_Q(recip, Q_S)

            Hh, Kk, Ll = hkl
            (abs(Kk - K_center) > K_halfwidth) && continue
            (abs(Ll - L_center) > L_halfwidth) && continue

            val = model_hkl(hkl, ω)  # already "vanadium-normalized" in spirit
            deposit_bilinear!(Hsum, Hh, ω, val)
            deposit_bilinear!(Hwt,  Hh, ω, 1.0)
        end
    end

    Hmean = Hist2D(H_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)
    return (Hsum=Hsum, Hwt=Hwt, Hmean=Hmean)
end

function predict_cut_mean_Hω_hkl_scan(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    H_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    recip::ReciprocalLattice,
    R_SL_list::Vector,
    model_hkl,
    eps::Float64=1e-12,
    K_center::Float64=0.0,
    K_halfwidth::Float64=0.15,
    L_center::Float64=0.0,
    L_halfwidth::Float64=10.0)

    Hsum = Hist2D(H_edges, ω_edges)
    Hwt  = Hist2D(H_edges, ω_edges)

    for R_SL in R_SL_list
        tmp = predict_cut_mean_Hω_hkl(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            H_edges=H_edges, ω_edges=ω_edges,
            recip=recip, R_SL=R_SL, model_hkl=model_hkl,
            eps=eps,
            K_center=K_center, K_halfwidth=K_halfwidth,
            L_center=L_center, L_halfwidth=L_halfwidth
        )
        Hsum.counts .+= tmp.Hsum.counts
        Hwt.counts  .+= tmp.Hwt.counts
    end

    Hmean = Hist2D(H_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)
    return (Hsum=Hsum, Hwt=Hwt, Hmean=Hmean)
end

using LinearAlgebra
using Base.Threads

"""
Precompute pixel×TOF kinematics once for a given instrument and Ei.

Returns:
  QLx,QLy,QLz : Float64 matrices (np × n_tof) storing Q_L components (Å⁻¹)
  ωL          : Float64 matrix  (np × n_tof) storing ω (meV)
  jacdt       : Float64 matrix  (np × n_tof) storing |dω/dt| * dt
  itmin,itmax : Int vectors (length np) with the valid TOF-bin range for each pixel
"""
function precompute_pixel_tof_kinematics(inst::Instrument,
    pixels::AbstractVector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::AbstractVector{<:Real})

    np = length(pixels)
    n_tof = length(tof_edges_s) - 1

    QLx = Matrix{Float64}(undef, np, n_tof)
    QLy = Matrix{Float64}(undef, np, n_tof)
    QLz = Matrix{Float64}(undef, np, n_tof)
    ωL  = Matrix{Float64}(undef, np, n_tof)
    jacdt = Matrix{Float64}(undef, np, n_tof)

    itmin = fill(n_tof + 1, np)
    itmax = fill(0, np)

    @inbounds for ip in 1:np
        p = pixels[ip]
        L2p = L2(inst, p.id)

        for it in 1:n_tof
            t0 = Float64(tof_edges_s[it])
            t1 = Float64(tof_edges_s[it+1])
            t  = 0.5*(t0 + t1)
            dt = (t1 - t0)

            Ef = try
                Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Q, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)

            QLx[ip,it] = Q[1]
            QLy[ip,it] = Q[2]
            QLz[ip,it] = Q[3]
            ωL[ip,it]  = ω

            jacdt[ip,it] = abs(dω_dt(inst.L1, L2p, Ei_meV, t)) * dt

            itmin[ip] = min(itmin[ip], it)
            itmax[ip] = max(itmax[ip], it)
        end
    end

    return (QLx=QLx, QLy=QLy, QLz=QLz, ωL=ωL, jacdt=jacdt, itmin=itmin, itmax=itmax)
end


"""
Alignment-aware single-crystal scan into a cut I(H,ω), using a kernel model_hkl(hkl, ω).

- Uses SampleAlignment (UB / u,v) via hkl_from_Q_S(aln, Q_S)
- Scans over orientations R_SL_list (each maps Q_L -> Q_S)
- Deposits weighted mean per (H,ω) bin:
    Hsum += model * (|dω/dt| dt)
    Hwt  += (|dω/dt| dt)
    Hmean = Hsum / (Hwt + eps)

Returns (Hsum, Hwt, Hmean).
"""
function predict_cut_mean_Hω_hkl_scan_aligned(inst::Instrument;
    pixels::AbstractVector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::AbstractVector{<:Real},
    H_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    aln::SampleAlignment,
    R_SL_list::AbstractVector,
    model_hkl,
    eps::Float64=1e-12,
    K_center::Float64=0.0,
    K_halfwidth::Float64=0.1,
    L_center::Float64=0.0,
    L_halfwidth::Float64=0.1,
    threaded::Bool=true)

    kin = precompute_pixel_tof_kinematics(inst, pixels, Ei_meV, tof_edges_s)

    QLx, QLy, QLz = kin.QLx, kin.QLy, kin.QLz
    ωL, jacdt = kin.ωL, kin.jacdt
    itmin, itmax = kin.itmin, kin.itmax

    np = length(pixels)
    n_tof = length(tof_edges_s) - 1

    # per-thread accumulators
    Hsum_thr = [Hist2D(H_edges, ω_edges) for _ in 1:nthreads()]
    Hwt_thr  = [Hist2D(H_edges, ω_edges) for _ in 1:nthreads()]

    function do_one_orientation!(R_SL)
        tid = threadid()
        Hsum = Hsum_thr[tid]
        Hwt  = Hwt_thr[tid]

        @inbounds for ip in 1:np
            hi = itmax[ip]
            hi == 0 && continue
            lo = itmin[ip]

            for it in lo:hi
                # Weight is the measure in ω-space induced by TOF binning
                w = jacdt[ip,it]
                w == 0.0 && continue

                # Q_S = R_SL * Q_L (manual multiply to avoid extra temporaries)
                qlx = QLx[ip,it]; qly = QLy[ip,it]; qlz = QLz[ip,it]
                qsx = R_SL[1,1]*qlx + R_SL[1,2]*qly + R_SL[1,3]*qlz
                qsy = R_SL[2,1]*qlx + R_SL[2,2]*qly + R_SL[2,3]*qlz
                qsz = R_SL[3,1]*qlx + R_SL[3,2]*qly + R_SL[3,3]*qlz
                Q_S = Vec3(qsx, qsy, qsz)

                ω = ωL[ip,it]
                hkl = hkl_from_Q_S(aln, Q_S)
                Hh, Kk, Ll = hkl

                (abs(Kk - K_center) > K_halfwidth) && continue
                (abs(Ll - L_center) > L_halfwidth) && continue

                deposit_bilinear!(Hsum, Hh, ω, model_hkl(hkl, ω) * w)
                deposit_bilinear!(Hwt,  Hh, ω, w)
            end
        end
    end

    if threaded && nthreads() > 1
        @threads for i in eachindex(R_SL_list)
            do_one_orientation!(R_SL_list[i])
        end
    else
        for R in R_SL_list
            do_one_orientation!(R)
        end
    end

    # reduce thread accumulators
    Hsum = Hist2D(H_edges, ω_edges)
    Hwt  = Hist2D(H_edges, ω_edges)
    for k in 1:nthreads()
        Hsum.counts .+= Hsum_thr[k].counts
        Hwt.counts  .+= Hwt_thr[k].counts
    end

    Hmean = Hist2D(H_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)

    return (Hsum=Hsum, Hwt=Hwt, Hmean=Hmean)
end

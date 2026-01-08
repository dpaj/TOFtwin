using LinearAlgebra
using Base.Threads


# ----------------------------
# Utility: safe solid angle factor
# (assumes DetectorPixel has ΔΩ; if you ever revert, set ΔΩ=1 in pixels)
# ----------------------------
@inline pixel_Ω(p::DetectorPixel) = p.ΔΩ

"""
Reduce detector×TOF matrix into a single-crystal cut I(H, ω) by:
- mapping pixel×TOF -> (Q_L, ω)
- rotating Q_L -> Q_S (sample frame)
- converting Q_S -> (H,K,L)
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
            Hh, Kk, Ll = hkl

            (abs(Kk - K_center) > K_halfwidth) && continue
            (abs(Ll - L_center) > L_halfwidth) && continue

            deposit_bilinear!(H, Hh, ω, w)
        end
    end

    return H
end


"""
Full conventional single-crystal cut workflow (pose-based):
model(Q_S, ω) -> detector×TOF -> vanadium normalize -> cut to (H,ω)
Return (Hraw, Hsum, Hwt, Hmean).

Weights include: |dω/dt| dt ΔΩ
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
    resolution::AbstractResolutionModel = NoResolution(),
    eps::Float64=1e-12,
    K_center::Float64=0.0,
    K_halfwidth::Float64=0.1,
    L_center::Float64=0.0,
    L_halfwidth::Float64=0.1)

    n_pix = length(inst.pixels)
    n_tof = length(tof_edges_s) - 1
    Cs = zeros(Float64, n_pix, n_tof)
    Cv = zeros(Float64, n_pix, n_tof)

    R_SL = T_SL(pose).R

    for p in pixels
        Ω = pixel_Ω(p)
        L2p = L2(inst, p.id)
        for it in 1:n_tof
            t0 = tof_edges_s[it]
            t1 = tof_edges_s[it+1]
            t  = 0.5*(t0 + t1)
            dt = t1 - t0

            δts, ws = time_nodes_weights(resolution, inst, p, Ei_meV, t)
            @inbounds for j in 1:length(δts)
                tj = t + δts[j]

                Ef = try
                    Ef_from_tof(inst.L1, L2p, Ei_meV, tj)
                catch
                    continue
                end
                (Ef <= 0 || Ef > Ei_meV) && continue

                Q_L, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
                Q_S = R_SL * Q_L

                jac = ws[j] * abs(dω_dt(inst.L1, L2p, Ei_meV, tj)) * dt * Ω
                Cs[p.id, it] += model(Q_S, ω) * jac
                Cv[p.id, it] += 1.0 * jac
            end
        end
    end

    Cnorm = normalize_by_vanadium(Cs, Cv; eps=eps)

    Hraw = reduce_pixel_tof_to_Hω_cut(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=Cs, recip=recip, pose=pose,
        H_edges=H_edges, ω_edges=ω_edges,
        K_center=K_center, K_halfwidth=K_halfwidth,
        L_center=L_center, L_halfwidth=L_halfwidth
    )

    Hsum = reduce_pixel_tof_to_Hω_cut(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=Cnorm, recip=recip, pose=pose,
        H_edges=H_edges, ω_edges=ω_edges,
        K_center=K_center, K_halfwidth=K_halfwidth,
        L_center=L_center, L_halfwidth=L_halfwidth
    )

    W = zeros(size(Cnorm))
    W[Cv .> 0.0] .= 1.0
    Hwt = reduce_pixel_tof_to_Hω_cut(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=W, recip=recip, pose=pose,
        H_edges=H_edges, ω_edges=ω_edges,
        K_center=K_center, K_halfwidth=K_halfwidth,
        L_center=L_center, L_halfwidth=L_halfwidth
    )

    Hmean = Hist2D(H_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)

    return (Hraw=Hraw, Hsum=Hsum, Hwt=Hwt, Hmean=Hmean)
end


"""
Predict a single-crystal cut I(H,ω) using an HKL-based kernel model_hkl(hkl,ω),
for one orientation R_SL (lab->sample rotation).

Uses weights w = |dω/dt| dt ΔΩ.
Returns numerator/denominator and their ratio (weighted mean).
"""
function predict_cut_weightedmean_Hω_hkl(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    H_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    recip::ReciprocalLattice,
    R_SL,
    model_hkl,
    eps::Float64=1e-12,
    K_center::Float64=0.0,
    K_halfwidth::Float64=0.15,
    L_center::Float64=0.0,
    L_halfwidth::Float64=10.0)

    Hnum = Hist2D(H_edges, ω_edges)  # sum(model * w)
    Hden = Hist2D(H_edges, ω_edges)  # sum(w)

    n_tof = length(tof_edges_s) - 1

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
            hkl = hkl_from_Q(recip, Q_S)

            Hh, Kk, Ll = hkl
            (abs(Kk - K_center) > K_halfwidth) && continue
            (abs(Ll - L_center) > L_halfwidth) && continue

            w = abs(dω_dt(inst.L1, L2p, Ei_meV, t)) * dt * p.ΔΩ

            deposit_bilinear!(Hnum, Hh, ω, model_hkl(hkl, ω) * w)
            deposit_bilinear!(Hden, Hh, ω, w)
        end
    end

    Hmean = Hist2D(H_edges, ω_edges)
    Hmean.counts .= Hnum.counts ./ (Hden.counts .+ eps)

    return (Hnum=Hnum, Hden=Hden, Hmean=Hmean)
end


"""
Scan version of predict_cut_weightedmean_Hω_hkl over orientations R_SL_list.
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


"""
Mean-per-bin version for HKL kernels:
Hsum += model * w
Hwt  += w
Hmean = Hsum / (Hwt + eps)

Uses w = |dω/dt| dt ΔΩ.
"""
function predict_cut_mean_Hω_hkl(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    H_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    recip::ReciprocalLattice,
    R_SL,
    model_hkl,
    eps::Float64=1e-12,
    K_center::Float64=0.0,
    K_halfwidth::Float64=0.15,
    L_center::Float64=0.0,
    L_halfwidth::Float64=10.0)

    Hsum = Hist2D(H_edges, ω_edges)
    Hwt  = Hist2D(H_edges, ω_edges)
    n_tof = length(tof_edges_s) - 1

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
            hkl = hkl_from_Q(recip, Q_S)

            Hh, Kk, Ll = hkl
            (abs(Kk - K_center) > K_halfwidth) && continue
            (abs(Ll - L_center) > L_halfwidth) && continue

            w = abs(dω_dt(inst.L1, L2p, Ei_meV, t)) * dt * p.ΔΩ
            val = model_hkl(hkl, ω)

            deposit_bilinear!(Hsum, Hh, ω, val * w)
            deposit_bilinear!(Hwt,  Hh, ω, w)
        end
    end

    Hmean = Hist2D(H_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)
    return (Hsum=Hsum, Hwt=Hwt, Hmean=Hmean)
end


"""
Scan version of predict_cut_mean_Hω_hkl over orientations R_SL_list.
"""
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


# ----------------------------
# FAST precompute + alignment-aware scan
# ----------------------------

"""
Precompute pixel×TOF kinematics once for a given instrument and Ei.

Returns:
  QLx,QLy,QLz : Float64 matrices (np × n_tof) storing Q_L components (Å⁻¹)
  ωL          : Float64 matrix  (np × n_tof) storing ω (meV)
  jacdt       : Float64 matrix  (np × n_tof) storing |dω/dt| * dt * ΔΩ
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
        Ω = pixel_Ω(p)
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

            jacdt[ip,it] = abs(dω_dt(inst.L1, L2p, Ei_meV, t)) * dt * Ω

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
    Hsum += model * (|dω/dt| dt ΔΩ)
    Hwt  += (|dω/dt| dt ΔΩ)
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
    resolution::AbstractResolutionModel = NoResolution(),
    eps::Float64=1e-12,
    K_center::Float64=0.0,
    K_halfwidth::Float64=0.1,
    L_center::Float64=0.0,
    L_halfwidth::Float64=0.1,
    threaded::Bool=true)

    # Fast path uses precomputed (pixel,TOF)->(Q_L,ω,jacdt) with *no* resolution.
    # If a nontrivial resolution model is provided, fall back to on-the-fly kinematics
    # with a small timing quadrature at each TOF bin.
    kin = (resolution isa NoResolution) ?
        precompute_pixel_tof_kinematics(inst, pixels, Ei_meV, tof_edges_s) : nothing

    if kin !== nothing
        QLx, QLy, QLz = kin.QLx, kin.QLy, kin.QLz
        ωL, jacdt = kin.ωL, kin.jacdt
        itmin, itmax = kin.itmin, kin.itmax
    end

    np = length(pixels)

    Hsum_thr = [Hist2D(H_edges, ω_edges) for _ in 1:nthreads()]
    Hwt_thr  = [Hist2D(H_edges, ω_edges) for _ in 1:nthreads()]

    function do_one_orientation!(R_SL)
        tid = threadid()
        Hsum = Hsum_thr[tid]
        Hwt  = Hwt_thr[tid]

        if kin !== nothing
            @inbounds for ip in 1:np
                hi = itmax[ip]
                hi == 0 && continue
                lo = itmin[ip]

                for it in lo:hi
                    w = jacdt[ip,it]
                    w == 0.0 && continue

                    # Q_S = R_SL * Q_L (manual multiply to avoid temporaries)
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
        else
            n_tof = length(tof_edges_s) - 1
            @inbounds for ip in 1:np
                p = pixels[ip]
                Ω = pixel_Ω(p)
                L2p = L2(inst, p.id)

                for it in 1:n_tof
                    t0 = Float64(tof_edges_s[it])
                    t1 = Float64(tof_edges_s[it+1])
                    t  = 0.5*(t0 + t1)
                    dt = (t1 - t0)

                    δts, ws = time_nodes_weights(resolution, inst, p, Ei_meV, t)
                    for j in 1:length(δts)
                        tj = t + δts[j]
                        Ef = try
                            Ef_from_tof(inst.L1, L2p, Ei_meV, tj)
                        catch
                            continue
                        end
                        (Ef <= 0 || Ef > Ei_meV) && continue

                        Q_L, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
                        # Q_S = R_SL*Q_L
                        qsx = R_SL[1,1]*Q_L[1] + R_SL[1,2]*Q_L[2] + R_SL[1,3]*Q_L[3]
                        qsy = R_SL[2,1]*Q_L[1] + R_SL[2,2]*Q_L[2] + R_SL[2,3]*Q_L[3]
                        qsz = R_SL[3,1]*Q_L[1] + R_SL[3,2]*Q_L[2] + R_SL[3,3]*Q_L[3]
                        Q_S = Vec3(qsx, qsy, qsz)

                        hkl = hkl_from_Q_S(aln, Q_S)
                        Hh, Kk, Ll = hkl
                        (abs(Kk - K_center) > K_halfwidth) && continue
                        (abs(Ll - L_center) > L_halfwidth) && continue

                        w = ws[j] * abs(dω_dt(inst.L1, L2p, Ei_meV, tj)) * dt * Ω
                        deposit_bilinear!(Hsum, Hh, ω, model_hkl(hkl, ω) * w)
                        deposit_bilinear!(Hwt,  Hh, ω, w)
                    end
                end
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

# =============================================================================
# Workflow-style single-crystal wrapper (mirrors PowderCtx / eval_powder pattern)
# =============================================================================

"""
`SingleCrystalCtx` is the single-crystal analogue of `PowderCtx`.

**Intent:** pay one-time costs once (pixel geometry already in `inst`, and optionally
(pixel,TOF)->(Q_L,ω,jacdt) precompute), then evaluate many kernel models cheaply.

Typical usage:
```julia
ctx = setup_singlecrystal_ctx(; inst, pixels, Ei_meV, tof_edges_s, H_edges, ω_edges,
                              aln, R_SL_list, K_center=0, K_halfwidth=0.1,
                              L_center=0, L_halfwidth=0.1)

pred = eval_singlecrystal!(ctx, model_hkl)  # overwrites ctx output hists
pred2 = eval_singlecrystal!(ctx, model_hkl2)
```
"""
mutable struct SingleCrystalCtx{TRes,TKin,TR}
    inst::Instrument
    pixels::Vector{DetectorPixel}
    Ei_meV::Float64
    tof_edges_s::Vector{Float64}

    # Cut definition
    H_edges::Vector{Float64}
    ω_edges::Vector{Float64}
    aln::SampleAlignment
    R_SL_list::Vector{TR}

    # Selection window in K/L (r.l.u.)
    K_center::Float64
    K_halfwidth::Float64
    L_center::Float64
    L_halfwidth::Float64

    # Numerical controls
    resolution::TRes
    eps::Float64

    # Cached kinematics for fast path (only when resolution isa NoResolution)
    kin::TKin

    # Preallocated accumulators (thread-local + reduced)
    Hsum_thr::Vector{Hist2D}
    Hwt_thr::Vector{Hist2D}
    Hsum::Hist2D
    Hwt::Hist2D
    Hmean::Hist2D
end

"""Create a `SingleCrystalCtx`. This does **not** load geometry; pass an `inst` whose pixels are already loaded."""
function setup_singlecrystal_ctx(; 
    inst::Instrument,
    pixels::AbstractVector{DetectorPixel},
    Ei_meV::Real,
    tof_edges_s::AbstractVector{<:Real},
    H_edges::AbstractVector{<:Real},
    ω_edges::AbstractVector{<:Real},
    aln::SampleAlignment,
    R_SL_list::AbstractVector,
    resolution::AbstractResolutionModel = NoResolution(),
    eps::Real = 1e-12,
    K_center::Real = 0.0,
    K_halfwidth::Real = 0.1,
    L_center::Real = 0.0,
    L_halfwidth::Real = 0.1,
)
    pix = Vector{DetectorPixel}(pixels)
    tof = collect(Float64, tof_edges_s)
    Hed = collect(Float64, H_edges)
    wed = collect(Float64, ω_edges)

    # Fast-path cache (only valid if there is no additional timing quadrature).
    kin = (resolution isa NoResolution) ? precompute_pixel_tof_kinematics(inst, pix, Float64(Ei_meV), tof) : nothing

    TR = eltype(collect(R_SL_list))
    Rlist = Vector{TR}(R_SL_list)

    Hsum_thr = [Hist2D(Hed, wed) for _ in 1:nthreads()]
    Hwt_thr  = [Hist2D(Hed, wed) for _ in 1:nthreads()]
    Hsum = Hist2D(Hed, wed)
    Hwt  = Hist2D(Hed, wed)
    Hmean = Hist2D(Hed, wed)

    return SingleCrystalCtx(inst, pix, Float64(Ei_meV), tof,
                            Hed, wed, aln, Rlist,
                            Float64(K_center), Float64(K_halfwidth),
                            Float64(L_center), Float64(L_halfwidth),
                            resolution, Float64(eps),
                            kin,
                            Hsum_thr, Hwt_thr, Hsum, Hwt, Hmean)
end

@inline function _zero_hist!(H::Hist2D)
    fill!(H.counts, 0.0)
    return H
end

"""Evaluate a single-crystal kernel model into the preallocated histograms stored in `ctx` (mutating)."""
function eval_singlecrystal!(ctx::SingleCrystalCtx, model_hkl; threaded::Bool=true)
    # zero thread-local accumulators
    @inbounds for k in 1:length(ctx.Hsum_thr)
        _zero_hist!(ctx.Hsum_thr[k])
        _zero_hist!(ctx.Hwt_thr[k])
    end

    inst = ctx.inst
    pixels = ctx.pixels
    Ei_meV = ctx.Ei_meV
    tof_edges_s = ctx.tof_edges_s
    aln = ctx.aln

    K_center = ctx.K_center
    K_halfwidth = ctx.K_halfwidth
    L_center = ctx.L_center
    L_halfwidth = ctx.L_halfwidth

    kin = ctx.kin
    if kin !== nothing
        QLx, QLy, QLz = kin.QLx, kin.QLy, kin.QLz
        ωL, jacdt = kin.ωL, kin.jacdt
        itmin, itmax = kin.itmin, kin.itmax
    end

    np = length(pixels)
    resolution = ctx.resolution

    function do_one_orientation!(R_SL)
        tid = threadid()
        Hsum = ctx.Hsum_thr[tid]
        Hwt  = ctx.Hwt_thr[tid]

        if kin !== nothing
            @inbounds for ip in 1:np
                hi = itmax[ip]
                hi == 0 && continue
                lo = itmin[ip]

                for it in lo:hi
                    w = jacdt[ip,it]
                    w == 0.0 && continue

                    # Q_S = R_SL * Q_L
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
        else
            n_tof = length(tof_edges_s) - 1
            @inbounds for ip in 1:np
                p = pixels[ip]
                Ω = pixel_Ω(p)
                L2p = L2(inst, p.id)

                for it in 1:n_tof
                    t0 = Float64(tof_edges_s[it])
                    t1 = Float64(tof_edges_s[it+1])
                    t  = 0.5*(t0 + t1)
                    dt = (t1 - t0)

                    δts, ws = time_nodes_weights(resolution, inst, p, Ei_meV, t)
                    for j in 1:length(δts)
                        tj = t + δts[j]
                        Ef = try
                            Ef_from_tof(inst.L1, L2p, Ei_meV, tj)
                        catch
                            continue
                        end
                        (Ef <= 0 || Ef > Ei_meV) && continue

                        Q_L, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)

                        # Q_S = R_SL * Q_L
                        qsx = R_SL[1,1]*Q_L[1] + R_SL[1,2]*Q_L[2] + R_SL[1,3]*Q_L[3]
                        qsy = R_SL[2,1]*Q_L[1] + R_SL[2,2]*Q_L[2] + R_SL[2,3]*Q_L[3]
                        qsz = R_SL[3,1]*Q_L[1] + R_SL[3,2]*Q_L[2] + R_SL[3,3]*Q_L[3]
                        Q_S = Vec3(qsx, qsy, qsz)

                        hkl = hkl_from_Q_S(aln, Q_S)
                        Hh, Kk, Ll = hkl
                        (abs(Kk - K_center) > K_halfwidth) && continue
                        (abs(Ll - L_center) > L_halfwidth) && continue

                        w = ws[j] * abs(dω_dt(inst.L1, L2p, Ei_meV, tj)) * dt * Ω
                        deposit_bilinear!(Hsum, Hh, ω, model_hkl(hkl, ω) * w)
                        deposit_bilinear!(Hwt,  Hh, ω, w)
                    end
                end
            end
        end
        return nothing
    end

    R_SL_list = ctx.R_SL_list
    if threaded && nthreads() > 1
        @threads for i in eachindex(R_SL_list)
            do_one_orientation!(R_SL_list[i])
        end
    else
        for R in R_SL_list
            do_one_orientation!(R)
        end
    end

    _zero_hist!(ctx.Hsum)
    _zero_hist!(ctx.Hwt)
    @inbounds for k in 1:length(ctx.Hsum_thr)
        ctx.Hsum.counts .+= ctx.Hsum_thr[k].counts
        ctx.Hwt.counts  .+= ctx.Hwt_thr[k].counts
    end

    ctx.Hmean.counts .= ctx.Hsum.counts ./ (ctx.Hwt.counts .+ ctx.eps)

    return (Hsum=ctx.Hsum, Hwt=ctx.Hwt, Hmean=ctx.Hmean)
end

"""Non-mutating convenience wrapper: returns a deep copy of the results."""
function eval_singlecrystal(ctx::SingleCrystalCtx, model_hkl; threaded::Bool=true)
    pred = eval_singlecrystal!(ctx, model_hkl; threaded=threaded)
    return (Hsum=deepcopy(pred.Hsum), Hwt=deepcopy(pred.Hwt), Hmean=deepcopy(pred.Hmean))
end

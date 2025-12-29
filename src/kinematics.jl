using StaticArrays
using LinearAlgebra

# Physical constants specialized for neutrons (precomputed)
# E(meV) = 2.07212485 * k^2(Å^-2)
const E_PER_K2 = 2.0721248525676734
const V_PER_SQRT_E = 437.3933626042086  # v(m/s) = V_PER_SQRT_E*sqrt(E_meV)

k_from_EmeV(EmeV::Float64) = sqrt(EmeV / E_PER_K2)          # Å^-1
EmeV_from_k(kAinv::Float64) = E_PER_K2 * kAinv^2            # meV
v_from_EmeV(EmeV::Float64) = V_PER_SQRT_E * sqrt(EmeV)      # m/s

"Total TOF (seconds) from Ei and Ef, with L1 and L2 in meters."
function tof_from_EiEf(L1::Float64, L2::Float64, Ei::Float64, Ef::Float64)
    vi = v_from_EmeV(Ei)
    vf = v_from_EmeV(Ef)
    return L1/vi + L2/vf
end

"Given total TOF (seconds), solve for Ef (meV)."
function Ef_from_tof(L1::Float64, L2::Float64, Ei::Float64, t::Float64)
    vi = v_from_EmeV(Ei)
    tmin = L1/vi
    t <= tmin && throw(ArgumentError("t must be > L1/vi (got t=$(t), L1/vi=$(tmin))"))
    vf = L2 / (t - tmin)
    # Ef = (1/2)m v^2, converted to meV -> invert v_from_EmeV
    return (vf / V_PER_SQRT_E)^2
end

"Unit direction from sample to pixel (pixel position in meters, lab frame)."
function direction_to_pixel(r_pix_L::SVector{3,Float64}, r_samp_L::SVector{3,Float64}=SVector{3,Float64}(0,0,0))
    u = r_pix_L - r_samp_L
    return u / norm(u)
end

"""
Compute (Q_L, ω) for a pixel given Ei and Ef.

Conventions:
- Lab frame is right-handed
- incident beam along +z => k_i = (0,0,k_i)
- pixel direction defines k_f direction
Returns:
- Q_L (Å^-1, lab)
- ω = Ei - Ef (meV)
"""
function Qω_from_pixel(r_pix_L::SVector{3,Float64}, Ei::Float64, Ef::Float64;
                       r_samp_L::SVector{3,Float64}=SVector{3,Float64}(0,0,0))
    u = direction_to_pixel(r_pix_L, r_samp_L)
    ki = k_from_EmeV(Ei)
    kf = k_from_EmeV(Ef)
    k_i_vec = SVector{3,Float64}(0.0, 0.0, ki)
    k_f_vec = kf * u
    Q = k_i_vec - k_f_vec
    ω = Ei - Ef
    return Q, ω
end

"""
Compute Q in lab frame from a pixel and Ei/Ef (same as Qω_from_pixel but Q only).
"""
function Q_L_from_pixel(r_pix_L::Vec3, Ei::Float64, Ef::Float64; r_samp_L::Vec3=Vec3(0.0,0.0,0.0))
    Q, _ = Qω_from_pixel(r_pix_L, Ei, Ef; r_samp_L=r_samp_L)
    return Q
end

"Sample-frame Q using pose rotation (vector => rotation only)."
function Q_S_from_pixel(r_pix_L::Vec3, Ei_meV::Float64, Ef_meV::Float64, pose::Pose; r_samp_L::Vec3=Vec3(0,0,0))
    Q_L = Q_L_from_pixel(r_pix_L, Ei_meV, Ef_meV; r_samp_L=r_samp_L)
    R_SL = T_SL(pose).R
    return R_SL * Q_L
end

"Derivative dω/dt (meV/s) for fixed Ei, L1, L2 at time t."
function dω_dt(L1::Float64, L2::Float64, Ei::Float64, t::Float64)
    vi = v_from_EmeV(Ei)
    t0 = L1 / vi
    τ  = t - t0
    τ <= 0 && throw(ArgumentError("t must be > L1/vi"))
    vf = L2 / τ
    Ef = (vf / V_PER_SQRT_E)^2
    # ω = Ei - Ef  => dω/dt = -dEf/dt = 2 Ef / τ
    return 2.0 * Ef / τ
end

"""
Approximate derivative d|Q|/dt (Å⁻¹/s) at fixed pixel direction.

This captures *timing-only* broadening of |Q| (via changing k_f magnitude) for a fixed pixel.
It does NOT include angular broadening (beam divergence, pixel size, mosaic, etc.).

Implementation: finite difference in total TOF t, using Ef_from_tof + Qω_from_pixel.
"""
function dQmag_dt(L1::Float64, L2::Float64, r_pix_L::SVector{3,Float64}, Ei::Float64, t::Float64;
                  r_samp_L::SVector{3,Float64}=SVector{3,Float64}(0.0,0.0,0.0),
                  ε::Float64=1e-7)

    ε = abs(ε)
    ε == 0 && throw(ArgumentError("ε must be nonzero"))
    vi = v_from_EmeV(Ei)
    t0 = L1 / vi

    Qmag_at(tp) = begin
        Ef = Ef_from_tof(L1, L2, Ei, tp)
        Q, _ = Qω_from_pixel(r_pix_L, Ei, Ef; r_samp_L=r_samp_L)
        return norm(Q)
    end

    tp = t + ε
    tm = t - ε

    # Use central difference when possible; otherwise fall back to forward difference.
    if tm > t0
        return (Qmag_at(tp) - Qmag_at(tm)) / (2ε)
    else
        t > t0 || throw(ArgumentError("t must be > L1/vi (got t=$(t), L1/vi=$(t0))"))
        return (Qmag_at(tp) - Qmag_at(t)) / ε
    end
end


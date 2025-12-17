using LinearAlgebra
using StaticArrays

# uses your package-wide aliases
# const Vec3 = SVector{3,Float64}
# const Mat3 = SMatrix{3,3,Float64,9}

"""
SampleAlignment holds UB in the *sample frame at goniometer zero*.

Convention:
  Q_S (Å⁻¹) = UB_S * HKL  (r.l.u.)
  HKL       = UBinv_S * Q_S
"""
struct SampleAlignment
    UB_S::Mat3
    UBinv_S::Mat3
end

UB_matrix(aln::SampleAlignment) = aln.UB_S

Q_S_from_hkl(aln::SampleAlignment, hkl::Vec3) = aln.UB_S * hkl
hkl_from_Q_S(aln::SampleAlignment, Q_S::Vec3) = aln.UBinv_S * Q_S

"Construct from a user-provided UB matrix in the sample frame."
function SampleAlignment_from_UB(UB_S::AbstractMatrix)
    UB = Mat3(UB_S)
    UBi = Mat3(inv(Matrix(UB)))
    return SampleAlignment(UB, UBi)
end

@inline function _unit(v::Vec3)
    n = norm(v)
    n == 0 && error("Zero-length vector in alignment.")
    return v / n
end

"""
Right-handed orthonormal triad (e1,e2,e3) from two non-collinear vectors.
e1 || a, e3 ⟂ plane(a,b), e2 completes RH basis.
"""
function _triad_from_two(a::Vec3, b::Vec3)
    e1 = _unit(a)
    e3 = cross(e1, b)
    n3 = norm(e3)
    n3 == 0 && error("u and v are collinear; cannot define a plane.")
    e3 = e3 / n3
    e2 = cross(e3, e1)
    return (e1, e2, e3)
end

"""
Build SampleAlignment from experimental u,v (in HKL) and a reciprocal lattice.

Inputs:
  recip: ReciprocalLattice (defines a*,b*,c* including 2π)
  u_hkl, v_hkl: directions in r.l.u. defining the scattering plane (Mantid-style idea)

You must also define where those directions point in the *sample frame* at goniometer zero.
Defaults match your SNS lab axes (z along ki, x right, y up):
  u_dir_S = +x, v_dir_S = +z   (so u–v plane is the horizontal scattering plane)
"""
function alignment_from_uv(recip::ReciprocalLattice;
    u_hkl::Vec3,
    v_hkl::Vec3,
    u_dir_S::Vec3=Vec3(1.0, 0.0, 0.0),
    v_dir_S::Vec3=Vec3(0.0, 0.0, 1.0))

    B = B_matrix(recip)      # HKL -> Q (Å⁻¹) in a crystal-Cartesian basis

    # Convert u,v into reciprocal vectors (Å⁻¹)
    uC = B * u_hkl
    vC = B * v_hkl

    (e1C, e2C, e3C) = _triad_from_two(uC, vC)
    (e1S, e2S, e3S) = _triad_from_two(u_dir_S, v_dir_S)

    M_C = Mat3(hcat(e1C, e2C, e3C))
    M_S = Mat3(hcat(e1S, e2S, e3S))

    # Rotation U (crystal reciprocal Cartesian -> sample frame)
    U_SC = M_S * transpose(M_C)

    UB_S = U_SC * B
    UBinv_S = Mat3(inv(Matrix(UB_S)))
    return SampleAlignment(UB_S, UBinv_S)
end

"""
Helper: y-axis goniometer scan (SNS-style lab axes).
If the sample is rotated by +θ in the lab, then vectors transform with R_SL = Ry(-(θ + θ0)).

zero_offset_rad lets you shift the goniometer "zero" without changing UB/alignment.
"""
goniometer_scan_RSL_y(angles_rad; zero_offset_rad=0.0) =
    [Ry(-(θ + zero_offset_rad)) for θ in angles_rad]

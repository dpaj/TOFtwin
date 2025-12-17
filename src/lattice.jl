using LinearAlgebra
using StaticArrays

const Vec3 = SVector{3,Float64}

"""
Crystal lattice parameters (Å, degrees).
"""
struct LatticeParams
    a::Float64
    b::Float64
    c::Float64
    α::Float64
    β::Float64
    γ::Float64
end

"""
Reciprocal lattice basis vectors (Å⁻¹) including 2π.
Q = h*a* + k*b* + l*c*
"""
struct ReciprocalLattice
    astar::Vec3
    bstar::Vec3
    cstar::Vec3
end

deg2radf(x) = (π/180.0) * x

"Build reciprocal basis (with 2π) from lattice parameters."
function reciprocal_lattice(lp::LatticeParams)
    a,b,c = lp.a, lp.b, lp.c
    α = deg2radf(lp.α)
    β = deg2radf(lp.β)
    γ = deg2radf(lp.γ)

    # Real-space basis in lab-like Cartesian coords
    a1 = Vec3(a, 0.0, 0.0)
    a2 = Vec3(b*cos(γ), b*sin(γ), 0.0)

    cx = c*cos(β)
    cy = c*(cos(α) - cos(β)*cos(γ)) / sin(γ)
    cz2 = c^2 - cx^2 - cy^2
    cz = cz2 > 0 ? sqrt(cz2) : 0.0
    a3 = Vec3(cx, cy, cz)

    V = dot(a1, cross(a2, a3))
    V == 0 && error("Degenerate lattice volume (check α,β,γ).")

    astar = 2π * cross(a2, a3) / V
    bstar = 2π * cross(a3, a1) / V
    cstar = 2π * cross(a1, a2) / V

    return ReciprocalLattice(astar, bstar, cstar)
end

"Convert Q (Å⁻¹) to (H,K,L) using reciprocal basis."
function hkl_from_Q(recip::ReciprocalLattice, Q::Vec3)
    M = hcat(recip.astar, recip.bstar, recip.cstar)  # 3×3
    hkl = M \ collect(Q)
    return Vec3(hkl[1], hkl[2], hkl[3])
end

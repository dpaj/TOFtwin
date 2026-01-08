using LinearAlgebra
using StaticArrays

const Vec3 = SVector{3,Float64}
const Mat3 = SMatrix{3,3,Float64,9}

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

# Convenience keyword constructor (handy for TOML-driven configs)
LatticeParams(; a, b, c, α, β, γ) = LatticeParams(Float64(a), Float64(b), Float64(c),
                                                Float64(α), Float64(β), Float64(γ))

# ASCII aliases (easier to type in configs): alpha/beta/gamma
LatticeParams(; a, b, c, alpha, beta, gamma) = LatticeParams(; a=a, b=b, c=c, α=alpha, β=beta, γ=gamma)

"""
Reciprocal lattice basis vectors (Å⁻¹) including 2π.
Q = h*a* + k*b* + l*c*
"""
struct ReciprocalLattice
    astar::Vec3
    bstar::Vec3
    cstar::Vec3
end

"HKL -> Q matrix with columns [a* b* c*] (Å⁻¹)."
B_matrix(recip::ReciprocalLattice) = Mat3(hcat(recip.astar, recip.bstar, recip.cstar))

"Convert (H,K,L) to Q (Å⁻¹)."
Q_from_hkl(recip::ReciprocalLattice, hkl::Vec3) = B_matrix(recip) * hkl


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

# Convenience keyword constructor: build reciprocal basis from lattice parameters
function ReciprocalLattice(; a::Real, b::Real, c::Real, α::Real, β::Real, γ::Real)
    return reciprocal_lattice(LatticeParams(; a=a, b=b, c=c, α=α, β=β, γ=γ))
end

function ReciprocalLattice(; a::Real, b::Real, c::Real, alpha::Real, beta::Real, gamma::Real)
    return ReciprocalLattice(; a=a, b=b, c=c, α=alpha, β=beta, γ=gamma)
end

"Convert Q (Å⁻¹) to (H,K,L) using reciprocal basis."
function hkl_from_Q(recip::ReciprocalLattice, Q::Vec3)
    B = B_matrix(recip)
    # B is a StaticArrays Mat3, but "\" is happier with Matrix sometimes:
    h = (Matrix(B) \ collect(Q))
    return Vec3(h[1], h[2], h[3])
end

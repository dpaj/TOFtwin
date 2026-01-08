using StaticArrays
using LinearAlgebra

"""
Goniometer model: provides lab->sample rotation R_SL given motor angles.
Store a fixed zero-offset rotation R0_SL to represent "encoder zero" or mounting offset.
"""
struct Goniometer
    R0_SL::Mat3
    axis::Symbol  # :x, :y, :z for now
end

Goniometer(; axis::Symbol=:y, zero_offset_deg::Float64=0.0) =
    Goniometer(axis === :x ? Rx(deg2rad(zero_offset_deg)) :
               axis === :y ? Ry(deg2rad(zero_offset_deg)) :
               axis === :z ? Rz(deg2rad(zero_offset_deg)) :
               error("axis must be :x, :y, or :z"),
               axis)

"Return R_SL for an angle (radians), using SNS sign convention: R_SL = R0 * Raxis(-θ)."
function R_SL(g::Goniometer, θ::Float64)
    Rscan = g.axis === :x ? Rx(-θ) :
            g.axis === :y ? Ry(-θ) :
            g.axis === :z ? Rz(-θ) :
            error("axis must be :x, :y, or :z")
    return g.R0_SL * Rscan
end

"""Return a vector of lab->sample rotations for a scan over angles in degrees.

Angles are interpreted as the motor angle θ (deg). Internally we use `R_SL(g, deg2rad(θ))`.
"""
function R_SL_scan(g::Goniometer, angles_deg::AbstractVector{<:Real})
    return [R_SL(g, deg2rad(Float64(a))) for a in angles_deg]
end

"""Convenience: build a uniform angle grid [start_deg, stop_deg] inclusive (like `start:step:stop`)
and return the corresponding R_SL list.

Returns a NamedTuple `(angles_deg, R_SL_list)`.
"""
function R_SL_scan(g::Goniometer; start_deg::Real, stop_deg::Real, step_deg::Real)
    ang = collect(Float64(start_deg):Float64(step_deg):Float64(stop_deg))
    return (angles_deg=ang, R_SL_list=R_SL_scan(g, ang))
end

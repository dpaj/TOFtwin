using StaticArrays
using LinearAlgebra

"""
Experiment pose describing transforms between sample, goniometer, and lab.

Transforms are right-handed rigid transforms:
- T_LG : goniometer -> lab
- T_GS : sample -> goniometer
So a point in sample coords maps to lab as: p_L = T_LG( T_GS(p_S) )
"""
struct Pose
    T_LG::Rigid
    T_GS::Rigid
end

Pose() = Pose(Rigid(), Rigid())

"Total transform from sample -> lab."
T_LS(p::Pose) = p.T_LG(p.T_GS)

"Inverse: lab -> sample."
T_SL(p::Pose) = inv(T_LS(p))

"""
Simple Euler convention for sample orientation relative to goniometer.

Choose one and stick to it. Here: R = Rz(γ)*Ry(β)*Rx(α)
(you can swap order later if your goniometer uses a different convention)
"""
function sample_orientation_euler(α::Float64, β::Float64, γ::Float64)
    return rigid(Rz(γ) * Ry(β) * Rx(α))
end

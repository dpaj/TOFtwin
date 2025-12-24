using StaticArrays

abstract type AbstractResolutionModel end

"No resolution broadening."
struct NoResolution <: AbstractResolutionModel end

"""
Timing-only Gaussian resolution.

`σt` may be either:
  * a constant Float64 (seconds), or
  * a callable `σt(inst, p, Ei_meV, t_center)::Float64`.

`N` is the Gauss–Hermite order (3, 5, or 7 supported).
"""
struct GaussianTimingResolution{T,N} <: AbstractResolutionModel
    σt::T
end

"Convenience constructor: GaussianTimingResolution(σt; order=3|5|7)."
function GaussianTimingResolution(σt; order::Int=3)
    if order == 3
        return GaussianTimingResolution{typeof(σt),3}(σt)
    elseif order == 5
        return GaussianTimingResolution{typeof(σt),5}(σt)
    elseif order == 7
        return GaussianTimingResolution{typeof(σt),7}(σt)
    else
        throw(ArgumentError("GaussianTimingResolution order must be 3, 5, or 7 (got $order)"))
    end
end

@inline _σt(res::GaussianTimingResolution, inst, p, Ei_meV::Float64, t::Float64) =
    res.σt isa Function ? res.σt(inst, p, Ei_meV, t) : Float64(res.σt)

"""
Return (δt, w) nodes and weights to approximate the Gaussian expectation over timing jitter:

  E[f(t + δt)],  δt ~ Normal(0,σt^2)

using Gauss–Hermite quadrature. The returned weights sum to 1.

Orders supported: 3, 5, 7.
"""
@inline function time_nodes_weights(::NoResolution, inst, p, Ei_meV::Float64, t::Float64)
    return (SVector(0.0), SVector(1.0))
end

# GH(3): δt = ±√3 σt, 0 ; weights = {1/6, 2/3, 1/6}
@inline function time_nodes_weights(res::GaussianTimingResolution{T,3}, inst, p, Ei_meV::Float64, t::Float64) where {T}
    σ = _σt(res, inst, p, Ei_meV, t)
    a = sqrt(3.0) * σ
    return (SVector(-a, 0.0, +a), SVector(1/6, 2/3, 1/6))
end

# GH(5)
@inline function time_nodes_weights(res::GaussianTimingResolution{T,5}, inst, p, Ei_meV::Float64, t::Float64) where {T}
    σ = _σt(res, inst, p, Ei_meV, t)
    s2 = sqrt(2.0) * σ
    x  = SVector(-2.0201828704560856, -0.9585724646138185, 0.0,
                 +0.9585724646138185, +2.0201828704560856)
    w  = SVector(0.019953242059045913, 0.3936193231522412, 0.9453087204829419,
                 0.3936193231522412, 0.019953242059045913) ./ sqrt(pi)
    return (s2 .* x, w)
end

# GH(7)
@inline function time_nodes_weights(res::GaussianTimingResolution{T,7}, inst, p, Ei_meV::Float64, t::Float64) where {T}
    σ = _σt(res, inst, p, Ei_meV, t)
    s2 = sqrt(2.0) * σ
    x  = SVector(-2.6519613568352335, -1.6735516287674714, -0.8162878828589647, 0.0,
                 +0.8162878828589647, +1.6735516287674714, +2.6519613568352335)
    w  = SVector(0.0009717812450995192, 0.05451558281912703, 0.4256072526101278, 0.8102646175568073,
                 0.4256072526101278, 0.05451558281912703, 0.0009717812450995192) ./ sqrt(pi)
    return (s2 .* x, w)
end

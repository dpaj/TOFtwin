using StaticArrays
using SpecialFunctions

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

# -----------------------------------------------------------------------------
# CDF-based Gaussian smearing in TOF (bin-overlap convolution)
# -----------------------------------------------------------------------------

"""
Timing-only Gaussian resolution applied as a *bin-overlap (CDF) convolution* along TOF.

This avoids "discrete lobes" that can appear for very broad σt when using a low-order
Gauss–Hermite sampling rule.

Notes:
- This is a 1D convolution in TOF *per detector pixel*.
- For speed, we assume σt is constant (Float64). If you pass a callable, we evaluate it
  once per pixel at a representative time.
"""
struct GaussianTimingCDFResolution{T} <: AbstractResolutionModel
    σt::T          # seconds (Float64) or callable
    nsigma::Float64
end

GaussianTimingCDFResolution(σt; nsigma::Float64=4.0) =
    GaussianTimingCDFResolution{typeof(σt)}(σt, nsigma)

@inline _σt(res::GaussianTimingCDFResolution, inst, p, Ei_meV::Float64, t::Float64) =
    res.σt isa Function ? res.σt(inst, p, Ei_meV, t) : Float64(res.σt)

# Standard Normal CDF using Base.Math.erf (no extra deps)
@inline Φ(z::Float64) = 0.5 * (1 + erf(z / sqrt(2.0)))

"""
Apply TOF-domain resolution to a detector×TOF matrix in-place.

Default: no-op.
"""
function apply_tof_resolution!(C::AbstractMatrix{Float64},
    inst, pixels, Ei_meV::Float64, tof_edges_s::AbstractVector{Float64},
    res::AbstractResolutionModel)
    return C
end

@inline function _is_uniform_edges(tof_edges::AbstractVector{Float64}; rtol=1e-10, atol=0.0)
    n = length(tof_edges)
    n < 3 && return true
    dt = tof_edges[2] - tof_edges[1]
    @inbounds for i in 2:n-1
        if abs((tof_edges[i+1] - tof_edges[i]) - dt) > (atol + rtol*abs(dt))
            return false
        end
    end
    return true
end


# -----------------------------------------------------------------------------
# Precomputed work for GaussianTimingCDFResolution (optional, for fast sweeps)
# -----------------------------------------------------------------------------

abstract type AbstractCDFSmearWork end

"Uniform TOF bins, shared kernel for all pixels (constant σt)."
struct CDFSmearUniformShared <: AbstractCDFSmearWork
    dt::Float64
    nsigma::Float64
    K::Int32
    w::Vector{Float64}     # length = 2K+1, normalized
end

"Uniform TOF bins, per-pixel kernels (σt callable evaluated once per pixel)."
struct CDFSmearUniformPerPixel <: AbstractCDFSmearWork
    dt::Float64
    nsigma::Float64
    K::Vector{Int32}              # length = n_used_pixels
    w::Vector{Vector{Float64}}    # per-pixel kernel, normalized
end

"Non-uniform TOF bins: cache bin centers to avoid per-call allocations."
struct CDFSmearGeneral <: AbstractCDFSmearWork
    nsigma::Float64
    tc::Vector{Float64}           # bin centers, length n_tof
end

"Build the normalized CDF overlap kernel for uniform bin width dt."
function _gaussian_cdf_uniform_kernel(dt::Float64, σ::Float64, nsigma::Float64)
    if σ <= 0
        return (0, [1.0])
    end
    K = Int(ceil(nsigma * σ / dt)) + 1
    w = Vector{Float64}(undef, 2K+1)
    @inbounds for (idx, k) in enumerate(-K:K)
        a = ((k - 0.5) * dt) / σ
        b = ((k + 0.5) * dt) / σ
        w[idx] = Φ(b) - Φ(a)
    end
    s = sum(w)
    s > 0 && (w ./= s)
    return (K, w)
end

"Fast uniform-bin smear using precomputed kernel (w,K)."
function _smear_row_gaussian_cdf_uniform_precomputed!(out::AbstractVector{Float64},
    inp::AbstractVector{Float64}, w::AbstractVector{Float64}, K::Int)

    fill!(out, 0.0)
    n = length(inp)
    @inbounds for i in 1:n
        xi = inp[i]
        xi == 0.0 && continue
        for k in -K:K
            j = i + k
            (1 <= j <= n) || continue
            out[j] += xi * w[k + K + 1]
        end
    end
    return out
end


"""
Precompute model-independent work for GaussianTimingCDFResolution smearing.

This is intended for fast sweeps over many models at fixed (instrument, pixels, Ei, tof_edges, resolution).
"""
function precompute_tof_smear_work(inst, pixels, Ei_meV::Float64,
    tof_edges_s::AbstractVector{Float64},
    res::GaussianTimingCDFResolution)

    n_tof = length(tof_edges_s) - 1
    (length(tof_edges_s) == n_tof + 1) || throw(ArgumentError("tof_edges_s length must be n_tof+1"))

    uniform = _is_uniform_edges(tof_edges_s)
    t_rep = 0.5*(tof_edges_s[1] + tof_edges_s[end])

    if uniform
        dt = tof_edges_s[2] - tof_edges_s[1]

        # If σt is constant, we can share a single kernel across all pixels.
        if !(res.σt isa Function)
            σ = Float64(res.σt)
            K, w = _gaussian_cdf_uniform_kernel(dt, σ, res.nsigma)
            return CDFSmearUniformShared(dt, res.nsigma, Int32(K), w)
        end

        # Otherwise, precompute one kernel per pixel (σt evaluated once at t_rep).
        n_used = length(pixels)
        Ks = Vector{Int32}(undef, n_used)
        ws = Vector{Vector{Float64}}(undef, n_used)
        @inbounds for (i, p) in pairs(pixels)
            σ = _σt(res, inst, p, Ei_meV, t_rep)
            K, w = _gaussian_cdf_uniform_kernel(dt, σ, res.nsigma)
            Ks[i] = Int32(K)
            ws[i] = w
        end
        return CDFSmearUniformPerPixel(dt, res.nsigma, Ks, ws)
    end

    # Non-uniform TOF bins: cache bin centers (tc).
    tc = Vector{Float64}(undef, n_tof)
    @inbounds for i in 1:n_tof
        tc[i] = 0.5*(tof_edges_s[i] + tof_edges_s[i+1])
    end
    return CDFSmearGeneral(res.nsigma, tc)
end

function _smear_row_gaussian_cdf_uniform!(out::AbstractVector{Float64},
    inp::AbstractVector{Float64}, dt::Float64, σ::Float64, nsigma::Float64)

    fill!(out, 0.0)
    n = length(inp)
    if σ <= 0
        copyto!(out, inp)
        return out
    end

    K = Int(ceil(nsigma * σ / dt)) + 1
    w = Vector{Float64}(undef, 2K+1)

    @inbounds for (idx, k) in enumerate(-K:K)
        a = ((k - 0.5) * dt) / σ
        b = ((k + 0.5) * dt) / σ
        w[idx] = Φ(b) - Φ(a)
    end
    s = sum(w)
    s > 0 && (w ./= s)  # conserve mass under truncation

    @inbounds for i in 1:n
        xi = inp[i]
        xi == 0.0 && continue
        for (idx, k) in enumerate(-K:K)
            j = i + k
            (1 <= j <= n) || continue
            out[j] += xi * w[idx]
        end
    end
    return out
end


function _smear_row_gaussian_cdf_general!(out::AbstractVector{Float64},
    inp::AbstractVector{Float64}, tof_edges::AbstractVector{Float64},
    σ::Float64, nsigma::Float64)

    n = length(inp)
    tc = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        tc[i] = 0.5*(tof_edges[i] + tof_edges[i+1])
    end
    return _smear_row_gaussian_cdf_general!(out, inp, tof_edges, tc, σ, nsigma)
end

function _smear_row_gaussian_cdf_general!(out::AbstractVector{Float64},
    inp::AbstractVector{Float64}, tof_edges::AbstractVector{Float64},
    tc::AbstractVector{Float64}, σ::Float64, nsigma::Float64)

    fill!(out, 0.0)
    n = length(inp)
    if σ <= 0
        copyto!(out, inp)
        return out
    end

    @inbounds for i in 1:n
        xi = inp[i]
        xi == 0.0 && continue
        tci = tc[i]

        tmin = tci - nsigma*σ
        tmax = tci + nsigma*σ

        jlo = 1
        while jlo <= n && tof_edges[jlo+1] < tmin
            jlo += 1
        end
        jhi = n
        while jhi >= 1 && tof_edges[jhi] > tmax
            jhi -= 1
        end
        jlo > jhi && continue

        sw = 0.0
        for j in jlo:jhi
            a = (tof_edges[j]   - tci) / σ
            b = (tof_edges[j+1] - tci) / σ
            sw += (Φ(b) - Φ(a))
        end
        sw <= 0 && continue

        scale = xi / sw
        for j in jlo:jhi
            a = (tof_edges[j]   - tci) / σ
            b = (tof_edges[j+1] - tci) / σ
            out[j] += scale * (Φ(b) - Φ(a))
        end
    end

    return out
end


"""
Apply CDF-based Gaussian timing resolution in-place.

We evaluate σt once per pixel at a representative time (midpoint of TOF window).
"""

"""
Apply CDF-based Gaussian timing resolution in-place.

If `work` is provided (from `precompute_tof_smear_work`), the smearing step avoids
recomputing the overlap kernel and allocations, which is ideal for sweeping many models.

We evaluate σt once per pixel at a representative time (midpoint of TOF window).
"""
function apply_tof_resolution!(C::AbstractMatrix{Float64},
    inst, pixels, Ei_meV::Float64, tof_edges_s::AbstractVector{Float64},
    res::GaussianTimingCDFResolution;
    work::Union{Nothing,AbstractCDFSmearWork}=nothing)

    n_tof = size(C, 2)
    (length(tof_edges_s) == n_tof + 1) || throw(ArgumentError("tof_edges_s length must be n_tof+1"))

    # Build work on demand (keeps backward compatibility and speeds up constant-σt case).
    work === nothing && (work = precompute_tof_smear_work(inst, pixels, Ei_meV, tof_edges_s, res))

    tmp = Vector{Float64}(undef, n_tof)

    if work isa CDFSmearUniformShared
        K = Int(work.K)
        w = work.w
        @inbounds for p in pixels
            row = view(C, p.id, :)
            _smear_row_gaussian_cdf_uniform_precomputed!(tmp, row, w, K)
            copyto!(row, tmp)
        end
        return C
    end

    if work isa CDFSmearUniformPerPixel
        @inbounds for (i, p) in pairs(pixels)
            row = view(C, p.id, :)
            _smear_row_gaussian_cdf_uniform_precomputed!(tmp, row, work.w[i], Int(work.K[i]))
            copyto!(row, tmp)
        end
        return C
    end

    if work isa CDFSmearGeneral
        t_rep = 0.5*(tof_edges_s[1] + tof_edges_s[end])
        @inbounds for p in pixels
            σ = _σt(res, inst, p, Ei_meV, t_rep)
            row = view(C, p.id, :)
            _smear_row_gaussian_cdf_general!(tmp, row, tof_edges_s, work.tc, σ, res.nsigma)
            copyto!(row, tmp)
        end
        return C
    end

    throw(ArgumentError("Unknown CDF smear work type: $(typeof(work))"))
end

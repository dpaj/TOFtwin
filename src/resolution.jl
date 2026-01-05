using StaticArrays
using SpecialFunctions
using SparseArrays

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
Precomputed TOF-smearing work for `GaussianTimingCDFResolution`.

This stores, for each source TOF bin `i`, a compact list of (target-bin, weight)
pairs describing how intensity in bin `i` is distributed across neighboring bins.

- `ptr[i] : ptr[i+1]-1` indexes the entries belonging to source bin `i`.
- `j[k]` gives the 1-based target-bin index.
- `w[k]` gives the normalized overlap weight for that target bin.
"""
struct TofSmearWork
    ptr::Vector{Int32}      # length n_tof+1, 1-based pointers into j/w
    j::Vector{Int32}        # target-bin indices (1-based)
    w::Vector{Float64}      # weights
end

@inline function _apply_tof_smear_work!(out::AbstractVector{Float64},
    inp::AbstractVector{Float64}, work::TofSmearWork)
    fill!(out, 0.0)
    n = length(inp)
    @inbounds for i in 1:n
        xi = inp[i]
        xi == 0.0 && continue
        k0 = work.ptr[i]
        k1 = work.ptr[i+1] - 1
        for k in k0:k1
            out[work.j[k]] += xi * work.w[k]
        end
    end
    return out
end

# Build a per-bin CDF-overlap sparse stencil (compact row lists) for uniform TOF edges.
function _precompute_tof_smear_work_uniform(tof_edges_s::AbstractVector{Float64},
    σsrc::AbstractVector{Float64}, nsigma::Float64)

    n = length(σsrc)
    dt = tof_edges_s[2] - tof_edges_s[1]

    # first pass: count entries per source bin
    counts = Vector{Int32}(undef, n)
    @inbounds for i in 1:n
        σ = σsrc[i]
        if σ <= 0
            counts[i] = 1
        else
            K = Int(ceil(nsigma * σ / dt)) + 1
            jlo = max(1, i - K)
            jhi = min(n, i + K)
            counts[i] = Int32(jhi - jlo + 1)
        end
    end

    ptr = Vector{Int32}(undef, n+1)
    ptr[1] = Int32(1)
    @inbounds for i in 1:n
        ptr[i+1] = ptr[i] + counts[i]
    end
    nnz = Int(ptr[end] - 1)

    j = Vector{Int32}(undef, nnz)
    w = Vector{Float64}(undef, nnz)

    # second pass: fill entries
    @inbounds for i in 1:n
        σ = σsrc[i]
        k = Int(ptr[i])
        if σ <= 0
            j[k] = Int32(i)
            w[k] = 1.0
            continue
        end

        K = Int(ceil(nsigma * σ / dt)) + 1
        jlo = max(1, i - K)
        jhi = min(n, i + K)

        # accumulate unnormalized overlap weights
        sw = 0.0
        for jj in jlo:jhi
            # offset in bins
            kk = jj - i
            a = ((kk - 0.5) * dt) / σ
            b = ((kk + 0.5) * dt) / σ
            wij = Φ(b) - Φ(a)
            j[k] = Int32(jj)
            w[k] = wij
            sw += wij
            k += 1
        end

        # normalize (mass conservation under truncation)
        if sw > 0
            k0 = Int(ptr[i])
            k1 = Int(ptr[i+1]-1)
            for kk in k0:k1
                w[kk] /= sw
            end
        end
    end

    return TofSmearWork(ptr, j, w)
end

# Build per-bin CDF-overlap work for non-uniform TOF edges.
function _precompute_tof_smear_work_general(tof_edges_s::AbstractVector{Float64},
    σsrc::AbstractVector{Float64}, nsigma::Float64)

    n = length(σsrc)
    # bin centers
    tc = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        tc[i] = 0.5 * (tof_edges_s[i] + tof_edges_s[i+1])
    end

    # first pass: determine j ranges and counts
    counts = Vector{Int32}(undef, n)
    jlo_v = Vector{Int32}(undef, n)
    jhi_v = Vector{Int32}(undef, n)

    @inbounds for i in 1:n
        σ = σsrc[i]
        if σ <= 0
            jlo_v[i] = Int32(i)
            jhi_v[i] = Int32(i)
            counts[i] = 1
            continue
        end
        tci = tc[i]
        tmin = tci - nsigma*σ
        tmax = tci + nsigma*σ

        # bins overlap when tof_edges[j] < tmax and tof_edges[j+1] > tmin
        jlo = max(1, searchsortedlast(tof_edges_s, tmin))
        jhi = min(n, searchsortedfirst(tof_edges_s, tmax) - 1)

        if jhi < jlo
            # pathological edge case; fall back to delta
            jlo = i
            jhi = i
        end

        jlo_v[i] = Int32(jlo)
        jhi_v[i] = Int32(jhi)
        counts[i] = Int32(jhi - jlo + 1)
    end

    ptr = Vector{Int32}(undef, n+1)
    ptr[1] = Int32(1)
    @inbounds for i in 1:n
        ptr[i+1] = ptr[i] + counts[i]
    end
    nnz = Int(ptr[end] - 1)

    j = Vector{Int32}(undef, nnz)
    w = Vector{Float64}(undef, nnz)

    # second pass: fill
    @inbounds for i in 1:n
        σ = σsrc[i]
        k = Int(ptr[i])

        if σ <= 0
            j[k] = Int32(i)
            w[k] = 1.0
            continue
        end

        tci = tc[i]
        jlo = Int(jlo_v[i])
        jhi = Int(jhi_v[i])

        sw = 0.0
        for jj in jlo:jhi
            a = (tof_edges_s[jj]   - tci) / σ
            b = (tof_edges_s[jj+1] - tci) / σ
            wij = Φ(b) - Φ(a)
            j[k] = Int32(jj)
            w[k] = wij
            sw += wij
            k += 1
        end

        if sw > 0
            k0 = Int(ptr[i])
            k1 = Int(ptr[i+1]-1)
            for kk in k0:k1
                w[kk] /= sw
            end
        end
    end

    return TofSmearWork(ptr, j, w)
end

"""
Precompute TOF-smearing work for `GaussianTimingCDFResolution`.

This is intended for cases where `res.σt` is either:
- a scalar (constant σt), or
- a vector `σt_bins` giving σt per TOF-bin center (length `length(tof_edges_s)-1`).

If `res.σt` is callable (pixel/time dependent), we return `nothing` because a
pixel-independent precompute cannot be guaranteed.
"""
function precompute_tof_smear_work(inst, pixels,
    Ei_meV::Float64, tof_edges_s::AbstractVector{Float64},
    res::GaussianTimingCDFResolution)

    n = length(tof_edges_s) - 1
    n <= 0 && throw(ArgumentError("tof_edges_s must have length >= 2"))

    if res.σt isa Function
        @warn "precompute_tof_smear_work: res.σt is callable; returning nothing (no pixel-independent precompute)."
        return nothing
    end

    σsrc = if res.σt isa AbstractVector
        Float64.(res.σt)
    else
        fill(Float64(res.σt), n)
    end

    (length(σsrc) == n) || throw(ArgumentError("σt vector length must be n_tof=$(n) (got $(length(σsrc)))"))

    if _is_uniform_edges(tof_edges_s)
        return _precompute_tof_smear_work_uniform(tof_edges_s, σsrc, res.nsigma)
    else
        return _precompute_tof_smear_work_general(tof_edges_s, σsrc, res.nsigma)
    end
end

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

    fill!(out, 0.0)
    n = length(inp)
    if σ <= 0
        copyto!(out, inp)
        return out
    end

    tc = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        tc[i] = 0.5*(tof_edges[i] + tof_edges[i+1])
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
        wtmp = Vector{Float64}(undef, jhi-jlo+1)
        for (k, j) in enumerate(jlo:jhi)
            a = (tof_edges[j]   - tci) / σ
            b = (tof_edges[j+1] - tci) / σ
            wj = Φ(b) - Φ(a)
            wtmp[k] = wj
            sw += wj
        end
        sw <= 0 && continue
        for (k, j) in enumerate(jlo:jhi)
            out[j] += xi * (wtmp[k] / sw)
        end
    end

    return out
end

"""
Apply CDF-based Gaussian timing resolution in-place.

We evaluate σt once per pixel at a representative time (midpoint of TOF window).
"""
function apply_tof_resolution!(C::AbstractMatrix{Float64},
    inst, pixels, Ei_meV::Float64, tof_edges_s::AbstractVector{Float64},
    res::GaussianTimingCDFResolution; work=nothing)

    n_tof = size(C, 2)
    (length(tof_edges_s) == n_tof + 1) || throw(ArgumentError("tof_edges_s length must be n_tof+1"))

    # Row indexing convention can differ across call sites:
    #   - some code stores rows by pixel.id (dense up to maxid),
    #   - other code stores rows in the same order as `pixels` (1..length(pixels)).
    #
    # Using the wrong convention under `@inbounds` can corrupt memory and crash Julia.
    nrow  = size(C, 1)
    npix  = length(pixels)
    maxpid = maximum(p.id for p in pixels)

    rows_by_pid = nrow >= maxpid
    if !rows_by_pid && nrow != npix
        throw(ArgumentError("apply_tof_resolution!: cannot map rows: size(C,1)=$nrow, length(pixels)=$npix, max(pixel.id)=$maxpid"))
    end

    @inline row_index(pid::Int, i::Int) = rows_by_pid ? pid : i

    # Fast path: use precomputed work (typically pixel-independent σt(TOF) curves).
    #
    # `work` may be:
    #   - a single `TofSmearWork` (shared by all pixels), or
    #   - a vector of `TofSmearWork` indexed by `pixel.id` *or* by pixel order.
    if work !== nothing
        tmp = Vector{Float64}(undef, n_tof)

        if work isa TofSmearWork
            @inbounds for (i, p) in enumerate(pixels)
                ridx = row_index(Int(p.id), i)
                row = view(C, ridx, :)
                _apply_tof_smear_work!(tmp, row, work)
                copyto!(row, tmp)
            end
            return C

        elseif work isa AbstractVector
            nwork = length(work)
            work_by_pid = nwork >= maxpid
            if !work_by_pid && nwork != npix
                throw(ArgumentError("apply_tof_resolution!: cannot map work vector: length(work)=$nwork, length(pixels)=$npix, max(pixel.id)=$maxpid"))
            end

            @inbounds for (i, p) in enumerate(pixels)
                pid = Int(p.id)
                ridx = row_index(pid, i)
                w = work_by_pid ? work[pid] : work[i]
                w isa TofSmearWork || throw(ArgumentError("work entry must be a TofSmearWork (got $(typeof(w)))"))
                row = view(C, ridx, :)
                _apply_tof_smear_work!(tmp, row, w)
                copyto!(row, tmp)
            end
            return C

        else
            throw(ArgumentError("work must be a TofSmearWork, a vector of TofSmearWork, or nothing"))
        end
    end

    uniform = _is_uniform_edges(tof_edges_s)
    dt = uniform ? (tof_edges_s[2] - tof_edges_s[1]) : NaN
    t_rep = 0.5*(tof_edges_s[1] + tof_edges_s[end])

    tmp = Vector{Float64}(undef, n_tof)

    # If σt is a per-TOF-bin curve, build a local work object once and apply.
    if res.σt isa AbstractVector
        work_local = precompute_tof_smear_work(inst, pixels, Ei_meV, tof_edges_s, res)
        work_local === nothing && throw(ArgumentError("σt is a vector but precompute_tof_smear_work returned nothing"))
        @inbounds for (i, p) in enumerate(pixels)
            ridx = row_index(Int(p.id), i)
            row = view(C, ridx, :)
            _apply_tof_smear_work!(tmp, row, work_local)
            copyto!(row, tmp)
        end
        return C
    end

    # Otherwise, fall back to the scalar σt behavior (constant or callable evaluated at a representative t).
    @inbounds for (i, p) in enumerate(pixels)
        ridx = row_index(Int(p.id), i)
        σ = _σt(res, inst, p, Ei_meV, t_rep)
        row = view(C, ridx, :)
        if uniform
            _smear_row_gaussian_cdf_uniform!(tmp, row, dt, σ, res.nsigma)
        else
            _smear_row_gaussian_cdf_general!(tmp, row, tof_edges_s, σ, res.nsigma)
        end
        copyto!(row, tmp)
    end

    return C
end


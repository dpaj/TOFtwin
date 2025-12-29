using LinearAlgebra
using Serialization
using SHA

# -----------------------------------------------------------------------------
# Simple disk-cache helpers (stdlib-only) for instrument-only precomputations.
#
# We deliberately avoid new modules/deps here.
# -----------------------------------------------------------------------------

# Default cache directory: <pkgroot>/.toftwin_cache
_toftwin_default_cache_dir() = joinpath(@__DIR__, "..", ".toftwin_cache")

function _toftwin_cache_dir(cache_dir::Union{Nothing,AbstractString})
    return cache_dir === nothing ? get(ENV, "TOFTWIN_CACHE_DIR", _toftwin_default_cache_dir()) : String(cache_dir)
end

function _sha1_bytes(x::AbstractVector{UInt8})
    return bytes2hex(sha1(x))
end

function _sha1_of_vector(v::AbstractVector)
    io = IOBuffer()
    write(io, v)
    return _sha1_bytes(take!(io))
end

function _sha1_of_string(s::AbstractString)
    return _sha1_bytes(Vector{UInt8}(codeunits(s)))
end

# Best-effort signature for the resolution model
function _resolution_sig(resolution)
    T = typeof(resolution)
    fns = fieldnames(T)
    if isempty(fns)
        return string(T)
    end
    parts = String[]
    for f in fns
        push!(parts, string(f), "=", string(getfield(resolution, f)), ";")
    end
    return string(T, "(", join(parts), ")")
end

# Best-effort signature for instrument geometry affecting TOF->(Q,ω)
function _instrument_sig(inst)
    io = IOBuffer()
    write(io, string(inst.name))
    write(io, Float64(inst.L1))
    # L2 is typically a Vector{Float64}
    write(io, inst.L2)
    # sample position can affect Q (if not at origin)
    if hasproperty(inst, :r_samp_L)
        # r_samp_L is likely a small static vector; serialize to be safe
        serialize(io, getproperty(inst, :r_samp_L))
    end
    return _sha1_bytes(take!(io))
end

function _pixel_sig(pixels::Vector{DetectorPixel})
    # stable on ids + ΔΩ (ΔΩ changes if you regroup/decimate differently)
    ids = Int32[p.id for p in pixels]
    io = IOBuffer()
    write(io, ids)
    for p in pixels
        write(io, Float64(p.ΔΩ))
    end
    return _sha1_bytes(take!(io))
end

function _cache_path(cache_dir::AbstractString, key::AbstractString; ext=".jls")
    return joinpath(cache_dir, key * ext)
end

function _cache_load(path::AbstractString)
    open(path, "r") do io
        return deserialize(io)
    end
end

function _cache_save(path::AbstractString, value)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, value)
    end
    return value
end

function _load_or_build(cache_dir::AbstractString, key::AbstractString, builder::Function; ext=".jls")
    path = _cache_path(cache_dir, key; ext=ext)
    if isfile(path)
        return _cache_load(path)
    end

function precompute_tof_smear_work_diskcached(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    resolution::GaussianTimingCDFResolution,
    cache_dir::Union{Nothing,AbstractString}=nothing,
    cache_tag::AbstractString="")

    cache_dir_s = _toftwin_cache_dir(cache_dir)

    # Conservative key: instrument geometry + pixel ids + Ei + tof grid + resolution params + user tag
    io = IOBuffer()
    serialize(io, inst.name)
    serialize(io, inst.L1)
    serialize(io, inst.L2)
    if hasproperty(inst, :r_samp_L)
        serialize(io, getproperty(inst, :r_samp_L))
    end
    serialize(io, Int32[p.id for p in pixels])
    serialize(io, Ei_meV)
    serialize(io, tof_edges_s)
    serialize(io, typeof(resolution))
    serialize(io, resolution.nsigma)
    # Note: for callable σt, we store only the fact it's callable; cache_tag should capture versioning.
    serialize(io, resolution.σt isa Function ? :callable : Float64(resolution.σt))
    serialize(io, cache_tag)

    key = "cdf_smear_work_" * bytes2hex(sha1(take!(io)))

    builder = () -> precompute_tof_smear_work(inst, pixels, Ei_meV, tof_edges_s, resolution)
    return _load_or_build(cache_dir_s, key, builder)
end




    return _cache_save(path, builder())
end

# -----------------------------------------------------------------------------
# Cached kinematics map for reduction: (pixel,tof-bin center) -> (|Q|, ω)
# -----------------------------------------------------------------------------

struct PixelTofQωMap
    "row_of_id[pid] = 1..n_used for used pixels; 0 otherwise"
    row_of_id::Vector{Int32}
    "bin-center |Q| values (Å⁻¹), size = (n_used, n_tof)"
    Qmag::Matrix{Float32}
    "bin-center ω values (meV), size = (n_used, n_tof)"
    ω::Matrix{Float32}
    "valid[row,it] indicates a physical, in-range (Ei,Ef) point"
    valid::BitMatrix
end

function precompute_pixel_tof_Qω_map(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    cache::Bool=false,
    cache_dir::Union{Nothing,AbstractString}=nothing,
    cache_tag::AbstractString="")

    cache_dir_s = _toftwin_cache_dir(cache_dir)

    key = "pix_tof_qwmap_" * _sha1_of_string(
        "v1|" *
        _instrument_sig(inst) * "|" *
        _pixel_sig(pixels) * "|" *
        string(Ei_meV) * "|" *
        _sha1_of_vector(tof_edges_s) * "|" *
        _sha1_of_string(cache_tag)
    )

    builder = function()
        n_pix = length(inst.pixels)
        n_tof = length(tof_edges_s) - 1
        row_of_id = fill(Int32(0), n_pix)

        n_used = length(pixels)
        Qmag = fill(Float32(NaN), n_used, n_tof)
        ω = fill(Float32(NaN), n_used, n_tof)
        valid = falses(n_used, n_tof)

        @inbounds for (row, p) in pairs(pixels)
            row_of_id[p.id] = Int32(row)
            L2p = L2(inst, p.id)
            for it in 1:n_tof
                t0 = tof_edges_s[it]
                t1 = tof_edges_s[it+1]
                t  = 0.5*(t0 + t1)
                t <= 0 && continue

                Ef = try
                    Ef_from_tof(inst.L1, L2p, Ei_meV, t)
                catch
                    continue
                end
                (Ef <= 0 || Ef > Ei_meV) && continue

                Q, ww = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
                Qmag[row, it] = Float32(norm(Q))
                ω[row, it] = Float32(ww)
                valid[row, it] = true
            end
        end

        return PixelTofQωMap(row_of_id, Qmag, ω, valid)
    end

    return cache ? _load_or_build(cache_dir_s, key, builder) : builder()
end

# -----------------------------------------------------------------------------
# Optional: disk-cached pixel×TOF prediction.
#
# Note: you MUST provide a cache_tag that uniquely identifies the model, otherwise
# you'll get incorrect reuse. (For Cv, use cache_tag="flat".)
# -----------------------------------------------------------------------------
function predict_pixel_tof_diskcached(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    model,
    resolution::AbstractResolutionModel = NoResolution(),
    cache_dir::Union{Nothing,AbstractString}=nothing,
    cache_tag::AbstractString)

    cache_dir_s = _toftwin_cache_dir(cache_dir)

    key = "pix_tof_C_" * _sha1_of_string(
        "v1|" *
        _instrument_sig(inst) * "|" *
        _pixel_sig(pixels) * "|" *
        string(Ei_meV) * "|" *
        _sha1_of_vector(tof_edges_s) * "|" *
        _resolution_sig(resolution) * "|" *
        _sha1_of_string(cache_tag)
    )

    builder = () -> predict_pixel_tof(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        model=model, resolution=resolution
    )

    return _load_or_build(cache_dir_s, key, builder)
end

"""
Predict a powder-style histogram in (|Q|, ω).

This supports optional *timing-only* resolution:
- NoResolution(): bin-center evaluation
- GaussianTimingResolution(σt; order=3|5|7): Gauss–Hermite sampling in TOF
- GaussianTimingCDFResolution(σt; nsigma=...): TOF-domain CDF/bin-overlap convolution (smear after compute)
"""
function predict_hist_Qω_powder(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    model,
    Q_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    resolution::AbstractResolutionModel = NoResolution(),
    cdf_work=nothing)

    # If using TOF-domain convolution, do detector×TOF once then reduce.
    if resolution isa GaussianTimingCDFResolution
        C = predict_pixel_tof(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            model=model, resolution=resolution,
            cdf_work=cdf_work
        )
        return reduce_pixel_tof_to_Qω_powder(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            C=C, Q_edges=Q_edges, ω_edges=ω_edges
        )
    end

    H = Hist2D(Q_edges, ω_edges)

    for p in pixels
        L2p = L2(inst, p.id)

        for it in 1:length(tof_edges_s)-1
            t0 = tof_edges_s[it]
            t1 = tof_edges_s[it+1]
            tc = 0.5*(t0 + t1)
            dt = (t1 - t0)

            δt, w = time_nodes_weights(resolution, inst, p, Ei_meV, tc)

            for k in eachindex(w)
                t = tc + δt[k]

                Ef = try
                    Ef_from_tof(inst.L1, L2p, Ei_meV, t)
                catch
                    continue
                end
                (Ef <= 0 || Ef > Ei_meV) && continue

                Q, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
                Qmag = norm(Q)

                jacdt = abs(dω_dt(inst.L1, L2p, Ei_meV, t)) * dt
                hist_add!(H, Qmag, ω, w[k] * model(Qmag, ω) * jacdt * p.ΔΩ)
            end
        end
    end
    return H
end

# -----------------------------------------------------------------------------
# Internal helper: detector×TOF prediction using *bin-center* evaluation only.
# This is the natural unsmeared input to a TOF-domain CDF convolution.
# -----------------------------------------------------------------------------
function _predict_pixel_tof_center(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    model)

    n_pix = length(inst.pixels)
    n_tof = length(tof_edges_s) - 1
    C = zeros(Float64, n_pix, n_tof)

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

            Q, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            Qmag = norm(Q)

            jacdt = abs(dω_dt(inst.L1, L2p, Ei_meV, t)) * dt
            C[p.id, it] += model(Qmag, ω) * jacdt * p.ΔΩ
        end
    end

    return C
end

"""
Predict detector×TOF counts (n_pix × n_tof) with optional timing resolution.

Notes:
- Resolution is applied in TOF space, which keeps the model "instrument-grounded."
- For CDF resolution, we compute unsmeared counts once and then smear along TOF.
"""
function predict_pixel_tof(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    model,
    resolution::AbstractResolutionModel = NoResolution(),
    cdf_work=nothing)

    # CDF/bin-overlap timing convolution: compute once then smear in TOF
    if resolution isa GaussianTimingCDFResolution
        C = _predict_pixel_tof_center(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s, model=model
        )
        try
            apply_tof_resolution!(C, inst, pixels, Ei_meV, tof_edges_s, resolution; work=cdf_work)
        catch err
            # Backward compatibility if apply_tof_resolution! doesn't accept `work`
            if err isa MethodError
                apply_tof_resolution!(C, inst, pixels, Ei_meV, tof_edges_s, resolution)
            else
                rethrow()
            end
        end
        return C
    end

    n_pix = length(inst.pixels)
    n_tof = length(tof_edges_s) - 1
    C = zeros(Float64, n_pix, n_tof)

    for p in pixels
        L2p = L2(inst, p.id)
        for it in 1:n_tof
            t0 = tof_edges_s[it]
            t1 = tof_edges_s[it+1]
            tc = 0.5*(t0 + t1)
            dt = t1 - t0

            δt, w = time_nodes_weights(resolution, inst, p, Ei_meV, tc)

            for k in eachindex(w)
                t = tc + δt[k]

                Ef = try
                    Ef_from_tof(inst.L1, L2p, Ei_meV, t)
                catch
                    continue
                end
                (Ef <= 0 || Ef > Ei_meV) && continue

                Q, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
                Qmag = norm(Q)

                jacdt = abs(dω_dt(inst.L1, L2p, Ei_meV, t)) * dt
                C[p.id, it] += w[k] * model(Qmag, ω) * jacdt * p.ΔΩ
            end
        end
    end

    return C
end

# Backwards-compatible positional signature (old API)
predict_pixel_tof(inst::Instrument,
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    model) = predict_pixel_tof(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s, model=model)



"Vanadium-like normalization in detector space."
function normalize_by_vanadium(C_sample::AbstractMatrix, C_van::AbstractMatrix; eps=1e-12)
    return C_sample ./ (C_van .+ eps)
end

"Reduce detector×TOF matrix into powder (|Q|,ω) histogram."
function reduce_pixel_tof_to_Qω_powder(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    C::AbstractMatrix,
    Q_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    map::Union{Nothing,PixelTofQωMap}=nothing,
    cache_map::Bool=false,
    cache_dir::Union{Nothing,AbstractString}=nothing,
    cache_tag::AbstractString="")

    # Optional: use a cached (pixel,tof)->(Q,ω) map to avoid recomputing kinematics in reduction.
    if map === nothing && cache_map
        map = precompute_pixel_tof_Qω_map(inst;
            pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            cache=true, cache_dir=cache_dir, cache_tag=cache_tag
        )
    end

    H = Hist2D(Q_edges, ω_edges)
    n_tof = length(tof_edges_s) - 1

    if map !== nothing
        @inbounds for p in pixels
            row = map.row_of_id[p.id]
            row == 0 && continue
            for it in 1:n_tof
                w = C[p.id, it]
                w == 0.0 && continue
                map.valid[row, it] || continue
                deposit_bilinear!(H, Float64(map.Qmag[row, it]), Float64(map.ω[row, it]), w)
            end
        end
        return H
    end

    # Fallback: original on-the-fly kinematics
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

            Q, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            deposit_bilinear!(H, norm(Q), ω, w)
        end
    end

    return H
end

"""
Conventional powder workflow:

model(Qmag, ω)  -> detector×TOF -> vanadium normalize -> reduce to (|Q|,ω)
and return the *mean per (Q,ω) bin* plus the per-bin weights (coverage).

Returns:
  (Hraw, Hsum, Hwt, Hmean)
"""
function predict_powder_mean_Qω(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    Q_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    model,
    eps::Float64 = 1e-12)

    # detector×TOF prediction for sample + "vanadium"
    Cs = predict_pixel_tof(inst; pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s, model=model)
    Cv = predict_pixel_tof(inst; pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s, model=(Q,ω)->1.0)

    # raw reduced sum (debug)
    Hraw = reduce_pixel_tof_to_Qω_powder(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=Cs, Q_edges=Q_edges, ω_edges=ω_edges
    )

    # vanadium-normalized in detector×TOF
    Cnorm = normalize_by_vanadium(Cs, Cv; eps=eps)

    # reduce the normalized values: this is a SUM over contributions
    Hsum = reduce_pixel_tof_to_Qω_powder(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=Cnorm, Q_edges=Q_edges, ω_edges=ω_edges
    )

    # per-bin weights/coverage (count of contributing detector×TOF samples)
    W = zeros(size(Cnorm))
    W[Cv .> 0.0] .= 1.0

    Hwt = reduce_pixel_tof_to_Qω_powder(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=W, Q_edges=Q_edges, ω_edges=ω_edges
    )

    Hmean = Hist2D(Q_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)

    return (Hraw=Hraw, Hsum=Hsum, Hwt=Hwt, Hmean=Hmean)
end
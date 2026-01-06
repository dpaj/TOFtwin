
# -----------------------------------------------------------------------------
# workflows_common.jl
#
# "Workflow layer" utilities: common helpers shared by powder/single-crystal
# pipelines. Kept Makie-free and light so it can be used in analysis code.
# -----------------------------------------------------------------------------

using Serialization
using SHA

# -----------------------------
# Stable hash helper (for disk-cache keys)
# -----------------------------
# NOTE: Base.hash is randomized per session; we need a stable digest.
# For strings: hash codeunits. For other objects: serialize then hash bytes.

function _wf_hash(s::AbstractString)
    return bytes2hex(sha1(codeunits(s)))
end

function _wf_hash(x)
    io = IOBuffer()
    serialize(io, x)
    return bytes2hex(sha1(take!(io)))
end

# Stable SHA1 digest of a file's *contents* (useful for cache invalidation when
# helper scripts change, e.g. the PyChop oracle).
function _wf_file_sha1(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(sha1(read(io)))
    end
end

# -----------------------------
# Small cache helpers (Serialization-based)
# -----------------------------

# NOTE: Serialization can be slow for very large objects on some filesystems.
# Prefer keeping large precomputes in RAM for multi-model sweeps.
# This helper exists mainly so demos/analysis can opt-in if desired.

# Sanitize filename pieces (Windows forbids <>:"/\\|?*)
_wf_safe_filename(s::AbstractString) = replace(String(s), r"[<>:\"/\\\\|?*]" => "_")

function _wf_cache_path(cache_dir::AbstractString, prefix::AbstractString, bytes::Vector{UInt8})
    p = _wf_safe_filename(prefix)
    return joinpath(cache_dir, p * "_" * bytes2hex(sha1(bytes)) * ".jls")
end

# Convenience overload: build cache path from structured parts.
# We hash the serialized tuple via _wf_hash so the key is stable across sessions.
function _wf_cache_path(cache_dir::AbstractString, prefix::AbstractString; parts=())
    p = _wf_safe_filename(prefix)
    return joinpath(cache_dir, p * "_" * _wf_hash(parts) * ".jls")
end

function _wf_load_or_compute(path::AbstractString, builder::Function; disk_cache::Bool=true)
    if disk_cache && isfile(path)
        open(path, "r") do io
            return deserialize(io)
        end
    end
    val = builder()
    if disk_cache
        mkpath(dirname(path))
        open(path, "w") do io
            serialize(io, val)
        end
    end
    return val
end

# -----------------------------
# NTOF suggestion helper (CDF striping control)
# -----------------------------
"""
    suggest_ntof(inst, pixels, Ei_meV, tmin, tmax, ω_edges, σt_s;
                 α=0.5, β=1/3, ntof_min=200, ntof_max=2000)

Suggest a reasonable number of TOF bins for CDF timing smearing.

Heuristic: choose `dt` such that:
- the ω-step per TOF bin is a fraction of the ω-bin width (avoid striping), and
- the Gaussian timing kernel itself is sampled adequately.

`dt = min( β*σt,  α*Δω_bin / max|dω/dt| )`
"""
function suggest_ntof(inst, pixels, Ei_meV, tmin, tmax, ω_edges, σt_s;
        α=0.5, β=1/3, ntof_min::Int=200, ntof_max::Int=2000)

    Δω_bin = (ω_edges[end] - ω_edges[1]) / (length(ω_edges)-1)
    ω_samples = unique(filter(ω -> (0.0 <= ω < Ei_meV),
        [0.0, 0.5, 1.0, 2.0, 5.0, 0.25Ei_meV, 0.5Ei_meV, min(10.0, Ei_meV-0.5)]))

    reps = pixels[[1, cld(length(pixels),2), length(pixels)]]
    max_dωdt = 0.0
    for p in reps
        L2p = L2(inst, p.id)
        for ω in ω_samples
            Ef = Ei_meV - ω
            Ef <= 0 && continue
            t = tof_from_EiEf(inst.L1, L2p, Ei_meV, Ef)
            max_dωdt = max(max_dωdt, abs(dω_dt(inst.L1, L2p, Ei_meV, t)))
        end
    end
    max_dωdt == 0.0 && return ntof_min, (tmax-tmin)/ntof_min, max_dωdt, Δω_bin

    dt_from_kernel = β * σt_s
    dt_from_omega  = α * Δω_bin / max_dωdt
    dt = min(dt_from_kernel, dt_from_omega)

    ntof = Int(ceil((tmax - tmin) / dt))
    ntof = clamp(ntof, ntof_min, ntof_max)
    dt = (tmax - tmin) / ntof
    return ntof, dt, max_dωdt, Δω_bin
end

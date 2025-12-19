using Random

abstract type PixelSampler end

struct AllPixels <: PixelSampler end

struct RandomSubset <: PixelSampler
    n::Int
    seed::Int
end

struct Stride <: PixelSampler
    step::Int
end

"""
Keep pixels on a coarse angular grid by subsampling the (iψ,iη) indices.
Example: AngularDecimate(3,2) keeps every 3rd ψ-bin and every 2nd η-bin.
"""
struct AngularDecimate <: PixelSampler
    stepψ::Int
    stepη::Int
end

"""
Pick up to `n_per_eta` pixels uniformly across ψ for each η-row (and each bank).
This is useful when you want to preserve vertical coverage while thinning ψ.
"""
struct StratifiedByEta <: PixelSampler
    n_per_eta::Int
    seed::Int
end

"""
Optional helper: select only one bank (:left or :right).
"""
struct ByBank <: PixelSampler
    which::Symbol
end

# --- dispatch entry points ---

sample_pixels(bank::DetectorBank, sampler::PixelSampler) = sample_pixels(bank.pixels, sampler)

sample_pixels(pix::AbstractVector{DetectorPixel}, ::AllPixels) = collect(pix)

function sample_pixels(pix::AbstractVector{DetectorPixel}, s::RandomSubset)
    rng = MersenneTwister(s.seed)
    n = min(s.n, length(pix))
    return rand(rng, collect(pix), n)
end

function sample_pixels(pix::AbstractVector{DetectorPixel}, s::Stride)
    step = max(s.step, 1)
    return collect(pix)[1:step:end]
end

function sample_pixels(pix::AbstractVector{DetectorPixel}, s::AngularDecimate)
    a = max(s.stepψ, 1)
    b = max(s.stepη, 1)
    out = DetectorPixel[]
    sizehint!(out, length(pix) ÷ (a*b))
    for p in pix
        if ((p.iψ - 1) % a == 0) && ((p.iη - 1) % b == 0)
            push!(out, p)
        end
    end
    return out
end

function sample_pixels(pix::AbstractVector{DetectorPixel}, s::ByBank)
    return [p for p in pix if p.bank == s.which]
end

function sample_pixels(pix::AbstractVector{DetectorPixel}, s::StratifiedByEta)
    rng = MersenneTwister(s.seed)

    # group by (bank, iη)
    groups = Dict{Tuple{Symbol,Int}, Vector{DetectorPixel}}()
    for p in pix
        key = (p.bank, p.iη)
        push!(get!(groups, key, DetectorPixel[]), p)
    end

    out = DetectorPixel[]
    for (key, row) in groups
        # sort by ψ so selection is stable-ish
        sort!(row, by = p -> p.ψ)
        n = min(s.n_per_eta, length(row))
        if n == length(row)
            append!(out, row)
        else
            # pick n indices spaced across the row (plus a tiny shuffle)
            idxs = round.(Int, range(1, length(row), length=n))
            # optional jitter to avoid always choosing same pixels
            idxs = [clamp(i + rand(rng, (-1):1), 1, length(row)) for i in idxs]
            append!(out, row[idxs])
        end
    end

    return out
end

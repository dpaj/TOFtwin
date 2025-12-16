using Random

"Container for a named set of pixels + metadata."
struct DetectorBank
    name::String
    pixels::Vector{DetectorPixel}
    meta::NamedTuple
end

"Convenience constructor that records the coverage keywords as metadata."
function bank_from_coverage(; name::String="coverage", kwargs...)
    pix = pixels_from_coverage(; kwargs...)
    return DetectorBank(name, pix, (; kwargs...))
end

# ------------------------- Sampling API -------------------------

abstract type PixelSampler end

struct AllPixels <: PixelSampler end

struct RandomSubset <: PixelSampler
    n::Int
    seed::Int
end
RandomSubset(n::Int; seed::Int=0) = RandomSubset(n, seed)

struct Stride <: PixelSampler
    step::Int
    offset::Int
end
Stride(step::Int; offset::Int=1) = Stride(step, offset)

"Pick up to n_per_eta pixels from each η-bin (optionally separately per bank)."
struct StratifiedByEta <: PixelSampler
    n_per_eta::Int
    seed::Int
    separate_banks::Bool
end
StratifiedByEta(n_per_eta::Int; seed::Int=0, separate_banks::Bool=true) =
    StratifiedByEta(n_per_eta, seed, separate_banks)

"Deterministic decimation in the (iψ,iη) grid."
struct AngularDecimate <: PixelSampler
    dψ::Int
    dη::Int
    startψ::Int
    startη::Int
    separate_banks::Bool
end
AngularDecimate(dψ::Int, dη::Int; startψ::Int=1, startη::Int=1, separate_banks::Bool=true) =
    AngularDecimate(dψ, dη, startψ, startη, separate_banks)

"Filter to one bank, then apply an inner sampler."
struct ByBank <: PixelSampler
    bank::Symbol
    inner::PixelSampler
end
ByBank(bank::Symbol; inner::PixelSampler=AllPixels()) = ByBank(bank, inner)

# ------------------------- Implementations -------------------------

sample_pixels(bank::DetectorBank, sampler::PixelSampler) = sample_pixels(bank.pixels, sampler)

sample_pixels(pix::Vector{DetectorPixel}, ::AllPixels) = pix

function sample_pixels(pix::Vector{DetectorPixel}, s::RandomSubset)
    n = length(pix)
    s.n >= n && return pix
    rng = MersenneTwister(s.seed)
    idx = randperm(rng, n)[1:s.n]
    sort!(idx)                 # keep stable-ish ordering by original id
    return pix[idx]
end

function sample_pixels(pix::Vector{DetectorPixel}, s::Stride)
    return pix[s.offset:s.step:end]
end

function sample_pixels(pix::Vector{DetectorPixel}, s::ByBank)
    sub = [p for p in pix if p.bank == s.bank]
    return sample_pixels(sub, s.inner)
end

function sample_pixels(pix::Vector{DetectorPixel}, s::StratifiedByEta)
    rng = MersenneTwister(s.seed)
    groups = Dict{Any, Vector{Int}}()

    for (i, p) in pairs(pix)
        key = s.separate_banks ? (p.bank, p.iη) : p.iη
        push!(get!(groups, key, Int[]), i)
    end

    chosen = Int[]
    for idxs in values(groups)
        if length(idxs) <= s.n_per_eta
            append!(chosen, idxs)
        else
            pick = randperm(rng, length(idxs))[1:s.n_per_eta]
            append!(chosen, idxs[pick])
        end
    end

    sort!(chosen)
    return pix[chosen]
end

function sample_pixels(pix::Vector{DetectorPixel}, s::AngularDecimate)
    chosen = Int[]
    for (i, p) in pairs(pix)
        okψ = (p.iψ - s.startψ) % s.dψ == 0
        okη = (p.iη - s.startη) % s.dη == 0
        if okψ && okη
            push!(chosen, i)
        end
    end
    return pix[chosen]
end

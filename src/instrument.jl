using LinearAlgebra

"""
Instrument geometry + cached per-pixel flight path.

Conventions:
- Lab frame right-handed, +z along k_i.
- pixels are indexed by pixel_id == array index (1..N).
"""
struct Instrument
    name::String
    L1::Float64              # meters
    r_samp_L::Vec3           # meters
    pixels::Vector{DetectorPixel}  # indexed by id
    L2::Vector{Float64}      # meters, indexed by id
end

function Instrument(; name::String="instrument",
    L1::Float64,
    pixels::Vector{DetectorPixel},
    r_samp_L::Vec3 = Vec3(0.0, 0.0, 0.0))

    maxid = maximum(p.id for p in pixels)
    pix_by_id = Vector{DetectorPixel}(undef, maxid)

    filled = falses(maxid)
    for p in pixels
        pix_by_id[p.id] = p
        filled[p.id] = true
    end
    all(filled) || throw(ArgumentError("Instrument: pixel ids must cover 1..$maxid with no gaps."))

    L2 = [norm(pix_by_id[i].r_L - r_samp_L) for i in 1:maxid]
    return Instrument(name, L1, r_samp_L, pix_by_id, L2)
end

pixel(inst::Instrument, id::Int) = inst.pixels[id]
L2(inst::Instrument, id::Int) = inst.L2[id]

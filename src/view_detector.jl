# src/view_detector.jl
#
# Geometry-only detector viewing utilities (no Makie dependency).
# Scripts can use these to plot with GLMakie/CairoMakie.

using Statistics

"""
    filter_pixels(pixels; bank_regex=nothing)

Filter pixels by bank name (Symbol -> String). If bank_regex is nothing, return unchanged.
"""
function filter_pixels(pixels::AbstractVector{<:DetectorPixel}; bank_regex=nothing)
    bank_regex === nothing && return pixels
    rx = bank_regex isa Regex ? bank_regex : Regex(String(bank_regex))
    return [p for p in pixels if occursin(rx, String(p.bank))]
end

"""
    decimate_pixels_angular(pixels; ψstride=1, ηstride=1)

Keep pixels on a stride grid in (iψ, iη). Assumes DetectorPixel has iψ/iη fields.
"""
function decimate_pixels_angular(pixels::AbstractVector{<:DetectorPixel}; ψstride::Int=1, ηstride::Int=1)
    (ψstride <= 1 && ηstride <= 1) && return pixels
    out = DetectorPixel[]
    for p in pixels
        ((p.iψ - 1) % ψstride == 0) || continue
        ((p.iη - 1) % ηstride == 0) || continue
        push!(out, p)
    end
    return out
end

"""
    detector_cloud(pixels; r_samp=Vec3(0,0,0), bank_regex=nothing, ψstride=1, ηstride=1)

Return plotting-friendly arrays and some metadata:
  xs, ys, zs, idxL, idxR, ringR, banks, pixels_used
"""
function detector_cloud(pixels::AbstractVector{<:DetectorPixel};
    r_samp::Vec3 = Vec3(0.0, 0.0, 0.0),
    bank_regex = nothing,
    ψstride::Int = 1,
    ηstride::Int = 1,
)
    pix = filter_pixels(pixels; bank_regex=bank_regex)
    pix = decimate_pixels_angular(pix; ψstride=ψstride, ηstride=ηstride)

    pts = getfield.(pix, :r_L)
    xs  = getindex.(pts, 1)
    ys  = getindex.(pts, 2)
    zs  = getindex.(pts, 3)

    # "right vs left" relative to sample x
    dx  = xs .- r_samp[1]
    idxR = dx .>= 0
    idxL = .!idxR

    # estimate a characteristic radius in the x–z plane around the sample
    xsr = xs .- r_samp[1]
    zsr = zs .- r_samp[3]
    ringR = isempty(pix) ? 0.0 : median(sqrt.(xsr.^2 .+ zsr.^2))

    banks = unique(getfield.(pix, :bank))

    return (
        xs=xs, ys=ys, zs=zs,
        idxL=idxL, idxR=idxR,
        ringR=ringR,
        banks=banks,
        pixels_used=pix,
        r_samp=r_samp,
    )
end

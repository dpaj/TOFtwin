# src/view_detector.jl
#
# Geometry-only detector viewing utilities (no Makie dependency).
# Scripts can use these to plot with GLMakie/CairoMakie.

using Statistics

"""
    idf_path(instr::Symbol) -> String

Resolve a built-in instrument name to an IDF path inside the TOFtwin repo.
"""
function idf_path(instr::Symbol)
    scripts_dir = normpath(joinpath(@__DIR__, "..", "scripts"))
    if instr === :CNCS
        return joinpath(scripts_dir, "CNCS_Definition_2025B.xml")
    elseif instr === :SEQUOIA
        return joinpath(scripts_dir, "SEQUOIA_Definition.xml")
    else
        throw(ArgumentError("Unknown instrument symbol: $instr. Expected :CNCS or :SEQUOIA (or pass an explicit IDF path)."))
    end
end

# Symbol overloads for convenience in scripts.
load_instrument_idf(instr::Symbol; cached::Bool=true, kwargs...) =
    load_instrument_idf(idf_path(instr); cached=cached, kwargs...)

detector_cloud_from_idf(instr::Symbol; kwargs...) =
    detector_cloud_from_idf(idf_path(instr); kwargs...)

"""
    load_instrument_idf(idf_path; cached=true, kwargs...)

Load a Mantid IDF instrument, preferring the newer disk-cache loader when available.
"""
function load_instrument_idf(idf_path::AbstractString; cached::Bool=true, kwargs...)
    # Prefer the new disk-cached path when present.
    if cached && isdefined(@__MODULE__, :MantidIDF) && isdefined(MantidIDF, :load_mantid_idf_diskcached)
        return MantidIDF.load_mantid_idf_diskcached(idf_path; kwargs...)
    end

    # Fall back to submodule loader if present.
    if isdefined(@__MODULE__, :MantidIDF) && isdefined(MantidIDF, :load_mantid_idf)
        return MantidIDF.load_mantid_idf(idf_path; kwargs...)
    end

    # Final fall back: legacy top-level loader.
    if isdefined(@__MODULE__, :load_mantid_idf)
        return load_mantid_idf(idf_path; kwargs...)
    end

    error("No Mantid IDF loader found. Expected MantidIDF.load_mantid_idf_diskcached / MantidIDF.load_mantid_idf / load_mantid_idf")
end

"""
    detector_cloud_from_idf(idf_path; r_samp=nothing, bank_regex=nothing, ψstride=1, ηstride=1,
                            cached=true,
                            grouping="", grouping_file=nothing, mask_btp="", mask_mode=:drop,
                            outdir="out", angle_step=0.5,
                            kwargs...)

Convenience helper: load an IDF instrument (optionally disk-cached), optionally apply
masking/grouping, and build a plotting-friendly detector cloud via `detector_cloud`.

Returns a NamedTuple: `(cloud, inst, meta, out)`.
"""
function detector_cloud_from_idf(idf_path::AbstractString;
    r_samp = nothing,
    bank_regex = nothing,
    ψstride::Int = 1,
    ηstride::Int = 1,
    cached::Bool = true,

    grouping::AbstractString = "",
    grouping_file::Union{Nothing,AbstractString} = nothing,
    mask_btp = "",
    mask_mode::Symbol = :drop,
    outdir::AbstractString = "out",
    angle_step::Real = 0.5,

    kwargs...,
)
    out = load_instrument_idf(idf_path; cached=cached, kwargs...)
    inst = getproperty(out, :inst)

    r_samp_L = r_samp === nothing ? inst.r_samp_L : r_samp

    pixels0 = if hasproperty(inst, :pixels)
        inst.pixels
    elseif hasproperty(out, :pixels)
        out.pixels
    else
        error("No pixels found on instrument/output. Expected `inst.pixels` or `out.pixels`.")
    end

    pixels = apply_grouping_masking(pixels0;
        instrument=inst.name,
        grouping=grouping,
        grouping_file=grouping_file,
        mask_btp=mask_btp,
        mask_mode=mask_mode,
        outdir=outdir,
        angle_step=angle_step,
        return_meta=false,
    )

    cloud = detector_cloud(pixels;
        r_samp=r_samp_L,
        bank_regex=bank_regex,
        ψstride=ψstride,
        ηstride=ηstride,
    )

    meta = hasproperty(out, :meta) ? out.meta : nothing
    return (cloud=cloud, inst=inst, meta=meta, out=out)
end

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

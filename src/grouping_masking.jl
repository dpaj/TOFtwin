# src/grouping_masking.jl
# Mantid-analog helpers: MaskBTP / GroupDetectors / GenerateGroupingPowder

using EzXML
using LinearAlgebra

# ----------------------------
# Small utilities
# ----------------------------

"""
Parse Mantid-style int specs like:
  ""               -> Int[]
  "1,2,5"          -> [1,2,5]
  "1-3,10,12-13"   -> [1,2,3,10,12,13]

(Used by MaskBTP and grouping XML parsing.)
"""
function _parse_intspec(s::AbstractString)
    s = strip(s)
    isempty(s) && return Int[]
    out = Int[]
    for tok in split(s, ',')
        tok = strip(tok)
        isempty(tok) && continue
        if occursin('-', tok)
            a, b = split(tok, '-', limit=2)
            lo = parse(Int, strip(a)); hi = parse(Int, strip(b))
            lo <= hi || throw(ArgumentError("Bad range '$tok'"))
            append!(out, lo:hi)
        else
            push!(out, parse(Int, tok))
        end
    end
    return out
end

# Extract trailing digits from bank symbols like :bank42 -> 42 (or nothing)
_banknum(bank::Symbol) = (m = match(r"(\d+)$", String(bank)); m === nothing ? nothing : parse(Int, m.captures[1]))

# Convert r_L -> (ψ, η) using your convention: ψ=atan2(x,z), η=atan2(y, sqrt(x^2+z^2))
function _psi_eta_from_r(r)
    x, y, z = r[1], r[2], r[3]
    ψ = atan(x, z)
    η = atan(y, sqrt(x*x + z*z))
    return ψ, η
end

# Scattering angle 2θ in degrees relative to +z beam
function _twotheta_deg_from_r(r)
    x, y, z = r[1], r[2], r[3]
    L = sqrt(x*x + y*y + z*z)
    L == 0 && return 0.0
    c = clamp(z / L, -1.0, 1.0)
    return acos(c) * (180 / pi)
end

# ----------------------------
# Parse Mantid XML grouping file
# ----------------------------

"""
Read a Mantid XML grouping file (the <detector-grouping> format) and return:
  groups_detids::Vector{Vector{Int}}

Supports <detids val="..."> (and also <ids val="..."> as Mantid docs show). :contentReference[oaicite:4]{index=4}
"""
function read_grouping_xml(path::AbstractString)
    doc = EzXML.readxml(path)
    root = EzXML.root(doc)
    EzXML.name(root) == "detector-grouping" || throw(ArgumentError("Not a detector-grouping XML: $path"))

    groups = Vector{Vector{Int}}()
    for g in EzXML.children(root)
        EzXML.name(g) == "group" || continue
        dets = Int[]
        for c in EzXML.children(g)
            nm = EzXML.name(c)
            (nm == "detids" || nm == "ids") || continue
            val = EzXML.getattr(c, "val", "")
            append!(dets, _parse_intspec(val))
        end
        isempty(dets) || push!(groups, dets)
    end
    return groups
end

# ----------------------------
# MaskBTP (speed-first version)
# ----------------------------

"""
MaskBTP analog.

Semantics match Mantid: if Bank/Tube/Pixel is blank -> applies to all of that type. :contentReference[oaicite:5]{index=5}

Assumptions in TOFtwin (documented so we can refine later):
- pixel.bank (Symbol) is used for Bank selection (by trailing number, e.g. :bank42)
- pixel.iψ is used as "Tube index"
- pixel.iη is used as "Pixel index"

Mode:
- :drop      -> remove masked pixels entirely (fastest downstream)
- :zeroΩ     -> keep pixels but set ΔΩ = 0.0
Returns: (pixels2, masked_detids)
"""
function MaskBTP(pixels::AbstractVector;
                 Bank::Union{Nothing,AbstractString,AbstractVector{<:Integer}}=nothing,
                 Tube::Union{Nothing,AbstractString,AbstractVector{<:Integer}}=nothing,
                 Pixel::Union{Nothing,AbstractString,AbstractVector{<:Integer}}=nothing,
                 DetectorList::Union{Nothing,AbstractString,AbstractVector{<:Integer}}=nothing,
                 Mode::Symbol = :drop)

    banks = Bank isa AbstractString ? _parse_intspec(Bank) :
            Bank isa AbstractVector{<:Integer} ? collect(Int, Bank) : Int[]
    tubes = Tube isa AbstractString ? _parse_intspec(Tube) :
            Tube isa AbstractVector{<:Integer} ? collect(Int, Tube) : Int[]
    pixs  = Pixel isa AbstractString ? _parse_intspec(Pixel) :
            Pixel isa AbstractVector{<:Integer} ? collect(Int, Pixel) : Int[]
    dets  = DetectorList isa AbstractString ? _parse_intspec(DetectorList) :
            DetectorList isa AbstractVector{<:Integer} ? collect(Int, DetectorList) : Int[]

    bankset = Set(banks); tubeset = Set(tubes); pixset = Set(pixs); detset = Set(dets)

    masked_detids = Int[]
    keep = Vector{eltype(pixels)}()
    sizehint!(keep, length(pixels))

    for p in pixels
        bnum = _banknum(p.bank)
        bank_ok  = isempty(bankset)  || (bnum !== nothing && (bnum in bankset))
        tube_ok  = isempty(tubeset)  || (p.iψ in tubeset)
        pixel_ok = isempty(pixset)   || (p.iη in pixset)
        det_ok   = isempty(detset)   || (p.mantid_id in detset)

        # Mantid-style: Bank/Tube/Pixel selection is an AND; DetectorList unions in.
        selected_btp = bank_ok && tube_ok && pixel_ok
        selected = selected_btp || det_ok

        if selected
            push!(masked_detids, p.mantid_id)
            if Mode == :drop
                continue
            elseif Mode == :zeroΩ
                # assumes DetectorPixel fields: (id,mantid_id,r_L,ψ,η,iψ,iη,bank,ΔΩ)
                p2 = typeof(p)(p.id, p.mantid_id, p.r_L, p.ψ, p.η, p.iψ, p.iη, p.bank, 0.0)
                push!(keep, p2)
            else
                throw(ArgumentError("Unknown Mode=$Mode (use :drop or :zeroΩ)"))
            end
        else
            push!(keep, p)
        end
    end

    return keep, masked_detids
end

# Julia-ish alias
mask_btp(args...; kwargs...) = MaskBTP(args...; kwargs...)

# ----------------------------
# GroupDetectors (coarsened / speed-first)
# ----------------------------

"""
GroupDetectors analog in "coarsened geometry" mode:
- Build super-pixels with r_L centroid, ψ/η from that centroid, and ΔΩ = sum(ΔΩ_i).
- mantid_id becomes negative (synthetic) so it never collides with real detector IDs.

Input grouping can be:
- GroupingFile=".../CNCS_8x2.xml"  (Mantid XML format) :contentReference[oaicite:6]{index=6}
- groups_detids::Vector{Vector{Int}} (already parsed)
Returns: (pixels_grouped, groups_members_idx)

groups_members_idx are indices into the *original* pixels vector.
"""
function GroupDetectors(pixels::AbstractVector;
                        GroupingFile::Union{Nothing,AbstractString}=nothing,
                        groups_detids::Union{Nothing,Vector{Vector{Int}}}=nothing)

    (GroupingFile === nothing) == (groups_detids === nothing) &&
        throw(ArgumentError("Provide exactly one of GroupingFile or groups_detids"))

    groups_detids = groups_detids === nothing ? read_grouping_xml(GroupingFile) : groups_detids

    # Map mantid detid -> index in pixels
    detid_to_idx = Dict{Int,Int}()
    for (i,p) in pairs(pixels)
        detid_to_idx[p.mantid_id] = i
    end

    grouped = Vector{eltype(pixels)}()
    members_idx = Vector{Vector{Int}}()
    sizehint!(grouped, length(groups_detids))
    sizehint!(members_idx, length(groups_detids))

    gid = 0
    for detids in groups_detids
        idxs = Int[]
        for d in detids
            i = get(detid_to_idx, d, 0)
            i == 0 && continue  # masked/dropped/missing
            push!(idxs, i)
        end
        isempty(idxs) && continue

        gid += 1
        # Use first member as template for "discrete" indices
        p0 = pixels[idxs[1]]

        # Centroid + summed solid angle
        r̄ = p0.r_L * 0
        Ω = 0.0
        for i in idxs
            pi = pixels[i]
            r̄ += pi.r_L
            Ω += pi.ΔΩ
        end
        r̄ /= length(idxs)
        ψ, η = _psi_eta_from_r(r̄)

        # synthetic mantid_id: negative
        mantid_id = -gid
        id = gid

        pnew = typeof(p0)(id, mantid_id, r̄, ψ, η, p0.iψ, p0.iη, p0.bank, Ω)

        push!(grouped, pnew)
        push!(members_idx, idxs)
    end

    return grouped, members_idx
end

# Julia-ish alias
group_detectors(args...; kwargs...) = GroupDetectors(args...; kwargs...)

# ----------------------------
# GenerateGroupingPowder (simple 2θ binning -> XML groups)
# ----------------------------

"""
GenerateGroupingPowder analog: group detectors by 2θ bins of width AngleStep degrees. :contentReference[oaicite:7]{index=7}

Returns groups_detids::Vector{Vector{Int}} (detector IDs).
Optionally writes GroupingFilename in Mantid XML format so it can be reused with GroupDetectors.
"""
function GenerateGroupingPowder(pixels::AbstractVector;
                                AngleStep::Real,
                                GroupingFilename::Union{Nothing,AbstractString}=nothing)

    step = float(AngleStep)
    step > 0 || throw(ArgumentError("AngleStep must be > 0"))

    bins = Dict{Int, Vector{Int}}()  # bin_id -> detids

    for p in pixels
        tt = _twotheta_deg_from_r(p.r_L)
        bid = Int(floor(tt / step))
        push!(get!(bins, bid, Int[]), p.mantid_id)
    end

    # stable order by increasing 2θ bin
    bids = sort(collect(keys(bins)))
    groups = [bins[bid] for bid in bids]

    if GroupingFilename !== nothing
        open(GroupingFilename, "w") do io
            println(io, """<?xml version="1.0" encoding="UTF-8" ?>""")
            println(io, """<detector-grouping>""")
            for (i, detids) in enumerate(groups)
                val = join(detids, ", ")
                println(io, """  <group name="$(i-1)"><detids val="$val"/></group>""")
            end
            println(io, """</detector-grouping>""")
        end
    end

    return groups
end

generate_grouping_powder(args...; kwargs...) = GenerateGroupingPowder(args...; kwargs...)

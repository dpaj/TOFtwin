# src/grouping_masking.jl
#
# Mantid-analog helpers (speed-first):
#   - MaskBTP: mask detectors by Bank/Tube/Pixel (or DetectorList)
#   - GroupDetectors: coarsen detector pixels using Mantid XML grouping files
#   - GenerateGroupingPowder: simple 2θ binning -> Mantid XML grouping
#
# The key compatibility trick for TOFtwin:
#   GroupDetectors defaults to IdMode=:representative, meaning the grouped pixel
#   keeps the id of one real detector pixel from the group (the "representative").
#   This preserves compatibility with existing code that calls L2(inst, p.id)
#   and indexes matrices by pid = p.id.

using EzXML
using LinearAlgebra
using SHA

# --- EzXML compatibility shims (older EzXML on Windows) -----------------------

# Name of a node/element
function _xmlname(node)
    # Prefer the property API (works on lots of EzXML versions)
    try
        return node.name
    catch
    end
    # Fallbacks
    if isdefined(EzXML, :name)
        return EzXML.name(node)
    elseif isdefined(EzXML, :nodename)
        return EzXML.nodename(node)
    else
        return ""
    end
end

# Iterate child *elements* (skip whitespace/text nodes) across EzXML versions
function _xmlelements(node)
    if isdefined(EzXML, :eachelement)
        return EzXML.eachelement(node)              # iterator
    elseif isdefined(EzXML, :elements)
        return EzXML.elements(node)                 # Vector
    else
        # Fallback to DOM traversal via firstelement/nextelement
        out = EzXML.Node[]
        el = try node.firstelement catch; nothing end
        while el !== nothing
            push!(out, el)
            el = try el.nextelement catch; nothing end
        end
        return out
    end
end

# Attribute getter across versions
function _xmlattr(node, key::AbstractString, default::AbstractString = "")
    if isdefined(EzXML, :getattr)
        try
            return EzXML.getattr(node, key, default)
        catch
        end
    end
    # Many versions support dict-like access for attributes on elements
    try
        return haskey(node, key) ? String(node[key]) : default
    catch
        return default
    end
end

# ----------------------------
# Small utilities
# ----------------------------

"""
Parse Mantid-style int specs like:
  ""               -> Int[]
  "1,2,5"          -> [1,2,5]
  "1-3,10,12-13"   -> [1,2,3,10,12,13]
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
function _banknum(bank::Symbol)
    s = String(bank)
    # Prefer patterns like :bank36, :bank_36, :CNCS_bank36_panel1, etc.
    if (m = match(r"(?i)\bbank\D*(\d+)\b", s)) !== nothing
        return parse(Int, m.captures[1])
    end
    # Fallback: first run of digits anywhere
    if (m = match(r"(\d+)", s)) !== nothing
        return parse(Int, m.captures[1])
    end
    return nothing
end

# Convert r_L -> (ψ, η) using TOFtwin convention: ψ=atan2(x,z), η=atan2(y, sqrt(x^2+z^2))
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
# Mantid XML grouping file parsing
# ----------------------------

"""
Read a Mantid XML grouping file (the <detector-grouping> format) and return:
  groups_detids::Vector{Vector{Int}}

Supports <detids val="..."> and <ids val="...">.
"""
function read_grouping_xml(path::AbstractString)
    doc = EzXML.readxml(path)

    # root access differs across EzXML versions
    root = try
        EzXML.root(doc)
    catch
        doc.root
    end

    _xmlname(root) == "detector-grouping" ||
        throw(ArgumentError("Not a detector-grouping XML: $path (root='$(_xmlname(root))')"))

    groups = Vector{Vector{Int}}()

    for g in _xmlelements(root)
        _xmlname(g) == "group" || continue

        dets = Int[]
        for c in _xmlelements(g)
            nm = _xmlname(c)
            (nm == "detids" || nm == "ids") || continue
            val = _xmlattr(c, "val", "")
            append!(dets, _parse_intspec(val))
        end

        isempty(dets) || push!(groups, dets)
    end

    return groups
end


# ----------------------------
# MaskBTP (speed-first)
# ----------------------------

"""
MaskBTP analog.

Semantics:
- If Bank/Tube/Pixel is blank (or not provided), it matches all of that axis.
- Bank/Tube/Pixel selection is ANDed.
- DetectorList is ORed in.

Assumptions in TOFtwin (can refine later):
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
                 Mode::Symbol = :drop,
                 BTP::AbstractString = "",
                 mode::Symbol = :drop)

    # Allow Mantid-style spec string in `BTP`, e.g. "Bank=36-50;Tube=...;Mode=drop".
    # Values provided explicitly via Bank/Tube/Pixel/DetectorList/Mode take precedence.
    if !isempty(strip(BTP))
        nt = parse_mask_btp_spec(BTP)
        if Bank === nothing && hasproperty(nt, :Bank); Bank = nt.Bank; end
        if Tube === nothing && hasproperty(nt, :Tube); Tube = nt.Tube; end
        if Pixel === nothing && hasproperty(nt, :Pixel); Pixel = nt.Pixel; end
        if DetectorList === nothing && hasproperty(nt, :DetectorList); DetectorList = nt.DetectorList; end
        if hasproperty(nt, :Mode)
            Mode = nt.Mode
        elseif Mode == :drop && mode != :drop
            Mode = mode
        end
    end

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

    banknum_missing = 0
    banknum_total = 0

    for p in pixels
        bnum = _banknum(p.bank)
        banknum_total += 1
        bnum === nothing && (banknum_missing += 1)
        bank_ok  = isempty(bankset)  || (bnum !== nothing && (bnum in bankset))
        tube_ok  = isempty(tubeset)  || (p.iψ in tubeset)
        pixel_ok = isempty(pixset)   || (p.iη in pixset)
        det_in_list = (!isempty(detset)) && (p.mantid_id in detset)

        selected_btp = bank_ok && tube_ok && pixel_ok
        selected = selected_btp || det_in_list

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


    if !isempty(bankset) && banknum_missing > 0
        @warn "MaskBTP: some pixels have unparseable bank symbols; Bank=... masking may be incomplete" banknum_missing=banknum_missing banknum_total=banknum_total
    end

    return keep, masked_detids
end

mask_btp(args...; kwargs...) = MaskBTP(args...; kwargs...)

# ----------------------------
# GroupDetectors (coarsened / speed-first)
# ----------------------------

"""
GroupDetectors analog in "coarsened geometry" mode:
- Build super-pixels with r_L centroid, ψ/η from that centroid, and ΔΩ = sum(ΔΩ_i).
- mantid_id becomes negative (synthetic) so it never collides with real detector IDs.

IdMode:
- :representative (default): grouped pixel id = representative real pixel id (p0.id)
  This preserves compatibility with existing TOFtwin code that uses L2(inst, p.id)
  and indexes Cs/Cv by pid = p.id.
- :contiguous: grouped pixel id = 1..nGroups (requires deeper changes elsewhere)

Input grouping can be:
- GroupingFile=".../CNCS_8x2.xml"  (Mantid XML format)
- groups_detids::Vector{Vector{Int}} (already parsed)

Returns: (pixels_grouped, groups_members_idx)

groups_members_idx are indices into the *original* pixels vector.
"""
function GroupDetectors(pixels::AbstractVector;
                        GroupingFile::Union{Nothing,AbstractString}=nothing,
                        groups_detids::Union{Nothing,Vector{Vector{Int}}}=nothing,
                        IdMode::Symbol = :representative)

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
        p0 = pixels[idxs[1]]  # representative

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

        id = if IdMode == :representative
            p0.id
        elseif IdMode == :contiguous
            gid
        else
            throw(ArgumentError("Unknown IdMode=$IdMode (use :representative or :contiguous)"))
        end

        # Keep discrete indices (iψ,iη,bank) from representative
        pnew = typeof(p0)(id, mantid_id, r̄, ψ, η, p0.iψ, p0.iη, p0.bank, Ω)

        push!(grouped, pnew)
        push!(members_idx, idxs)
    end

    return grouped, members_idx
end

group_detectors(args...; kwargs...) = GroupDetectors(args...; kwargs...)

# ----------------------------
# GenerateGroupingPowder (simple 2θ binning -> XML groups)
# ----------------------------

"""
GenerateGroupingPowder analog: group detectors by 2θ bins of width AngleStep degrees.

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

    bids = sort(collect(keys(bins)))
    groups = [bins[bid] for bid in bids]

    if GroupingFilename !== nothing
        mkpath(dirname(GroupingFilename))
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

# ----------------------------
# Convenience: find built-in grouping files + apply workflow
# ----------------------------

"""
Return candidate search directories for grouping XML files inside the repo.
We check a few common locations so you can drop the XMLs in either scripts/ or assets/.
"""
function _grouping_search_dirs()
    # grouping_masking.jl is in src/, so repo root is one level up
    repo_root = normpath(joinpath(@__DIR__, ".."))
    return (
        joinpath(repo_root, "assets", "grouping"),
        joinpath(repo_root, "scripts", "grouping"),
        joinpath(repo_root, "scripts"),
    )
end

"""
Resolve a grouping tag like "8x2" to a file path, searching repo locations.

For CNCS, expects filenames like:
  CNCS_2x1.xml, CNCS_4x1.xml, CNCS_8x1.xml, CNCS_4x2.xml, CNCS_8x2.xml
"""
function grouping_xml_path(instrument::Union{Symbol,AbstractString}, grouping::AbstractString)
    instr = instrument isa Symbol ? String(instrument) : String(instrument)
    instr = uppercase(instr)

    # Normalize common instrument aliases
    names = String[instr]
    if instr == "SEQUOIA"
        push!(names, "SEQ")
    elseif instr == "SEQ"
        push!(names, "SEQUOIA")
    end
    names = unique(names)

    # Candidate filenames (we try a few common conventions)
    fnames = String[]
    for nm in names
        push!(fnames, "$(nm)_$(grouping)_grouping.xml")
        push!(fnames, "$(nm)_$(grouping).xml")
        push!(fnames, "$(nm)_$(grouping)_grouping.XML")  # harmless on Windows; helps on case-sensitive FS if someone used caps
        push!(fnames, "$(nm)_$(grouping).XML")
    end
    fnames = unique(fnames)

    for d in _grouping_search_dirs()
        for fname in fnames
            path = joinpath(d, fname)
            isfile(path) && return path
        end
    end

    throw(ArgumentError(
        "Grouping XML not found for instrument=$instr grouping='$(grouping)'. " *
        "Tried files: $(fnames). " *
        "Searched dirs: $(_grouping_search_dirs())"
    ))
end

"""
Parse a simple MaskBTP spec string like:
  "Bank=40-50;Tube=;Pixel=;DetectorList=123,124;Mode=drop"

Returns a NamedTuple suitable for passing as kwargs to MaskBTP.
"""
function parse_mask_btp_spec(spec::AbstractString)
    spec = strip(spec)
    isempty(spec) && return NamedTuple()
    parts = split(spec, ';')
    kv = Dict{Symbol,Any}()
    for p in parts
        p = strip(p)
        isempty(p) && continue
        if !occursin('=', p)
            throw(ArgumentError("Bad MaskBTP spec chunk '$p' (expected key=value)"))
        end
        k, v = split(p, '=', limit=2)
        k0 = lowercase(strip(k))
        # Normalize keys (case-insensitive) so users can write bank/Bank/BANK etc.
        ksym = k0 == "mode" ? :Mode :
               k0 == "bank" ? :Bank :
               k0 == "tube" ? :Tube :
               k0 == "pixel" ? :Pixel :
               (k0 == "detectorlist" || k0 == "detectors" || k0 == "detids") ? :DetectorList :
               Symbol(strip(k))
        vstr = strip(v)
        if ksym == :Mode
            kv[:Mode] = Symbol(lowercase(vstr))
        else
            kv[ksym] = vstr
        end
    end
    return (; kv...)
end



"""
Apply masking + grouping in one call.

Arguments:
- pixels: Vector{DetectorPixel}
- instrument: :CNCS / :SEQUOIA (used to resolve built-in grouping xml if needed)
- grouping:
    "" (no grouping)
    "2x1","4x1","8x1","4x2","8x2"  (load grouping XML)
    "powder"                      (generate grouping by 2θ bins, write xml to outdir)
- grouping_file: optional explicit path overriding built-in resolution
- mask_btp:
    "" (no mask) OR a spec string OR a Dict/NamedTuple of MaskBTP kwargs
- mask_mode: :drop or :zeroΩ (used if mask_btp is spec string without Mode)
- angle_step: used only for grouping="powder"
- return_meta: if true, returns (pixels=..., meta=NamedTuple)
"""
function apply_grouping_masking(pixels::AbstractVector;
                                instrument::Union{Symbol,AbstractString} = :CNCS,
                                grouping::AbstractString = "",
                                grouping_file::Union{Nothing,AbstractString} = nothing,
                                mask_btp = "",
                                mask_mode::Symbol = :drop,
                                outdir::AbstractString = "out",
                                angle_step::Real = 0.5,
                                return_meta::Bool = false)

    meta = (mask = nothing, grouping = nothing)

    # ---- masking
    pix = pixels
    if mask_btp isa AbstractString
        if !isempty(strip(mask_btp))
            nt = parse_mask_btp_spec(mask_btp)
            if !haskey(nt, :Mode)
                nt = merge(nt, (; Mode=mask_mode))
            end
            pix2, masked = MaskBTP(pix; nt...)
            pix = pix2
            meta = merge(meta, (mask=(masked_detids=masked, spec=mask_btp, mode=nt.Mode),))
        end
    elseif mask_btp isa Dict || mask_btp isa NamedTuple
        pix2, masked = MaskBTP(pix; mask_btp...)
        pix = pix2
        meta = merge(meta, (mask=(masked_detids=masked, spec=mask_btp, mode=get(mask_btp, :Mode, mask_mode)),))
    elseif mask_btp === nothing
        # do nothing
    else
        throw(ArgumentError("mask_btp must be String, Dict, NamedTuple, or nothing"))
    end

    # ---- grouping
    g = strip(grouping)
    if !isempty(g)
        if lowercase(g) == "powder"
            gxml = joinpath(outdir, "powdergroupfile.xml")
            groups = GenerateGroupingPowder(pix; AngleStep=angle_step, GroupingFilename=gxml)
            pixg, members = GroupDetectors(pix; groups_detids=groups, IdMode=:representative)
            pix = pixg
            meta = merge(meta, (grouping=(kind=:powder, file=gxml, angle_step=angle_step, members_idx=members),))
        else
            gfile = grouping_file === nothing ? grouping_xml_path(instrument, g) : grouping_file
            pixg, members = GroupDetectors(pix; GroupingFile=gfile, IdMode=:representative)
            pix = pixg
            meta = merge(meta, (grouping=(kind=:xml, file=gfile, tag=g, members_idx=members),))
        end
    end

    return return_meta ? (pixels=pix, meta=meta) : pix
end
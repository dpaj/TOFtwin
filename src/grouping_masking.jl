# src/grouping_masking.jl
# Mantid-analog helpers: MaskBTP / GroupDetectors / GenerateGroupingPowder
#
# Goals:
# - "speed-first" detector coarsening / grouping (reduce pixels)
# - Mantid-ish API: MaskBTP, GroupDetectors, GenerateGroupingPowder
# - Work with CNCS AND SEQUOIA grouping XML assets

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

# Extract an integer "bank number" from a Symbol tag like :bank40, :bank_40, :bank40_left, :left40, etc.
_banknum(bank::Symbol) = begin
    s = String(bank)
    m = match(r"(\\d+)", s)
    m === nothing ? nothing : parse(Int, m.captures[1])
end

# ----------------------------
# Grouping-file lookup
# ----------------------------

# Where we look for shipped grouping XMLs.
function _grouping_search_dirs()
    # assets/grouping is the preferred location
    dirs = String[]
    # @__DIR__ is src/
    push!(dirs, normpath(joinpath(@__DIR__, "..", "assets", "grouping")))
    # Scripts sometimes drop grouping xmls alongside scripts
    push!(dirs, normpath(joinpath(@__DIR__, "..", "scripts", "grouping")))
    push!(dirs, normpath(joinpath(@__DIR__, "..", "scripts")))
    return dirs
end

# Normalize instrument identifiers for grouping-file naming conventions.
# CNCS files are usually "CNCS_8x2.xml"
# SEQUOIA files (as you have them) are usually "SEQ_4x2_grouping.xml"
function _grouping_instkey(instrument::Union{Symbol,AbstractString})
    s = instrument isa Symbol ? String(instrument) : String(instrument)
    u = uppercase(strip(s))
    if occursin("SEQUOIA", u) || u == "SEQ"
        return "SEQ"
    elseif occursin("CNCS", u)
        return "CNCS"
    else
        return u
    end
end

"""
Resolve a built-in grouping XML filename given (instrument, grouping) and search paths.

This is used when you pass `grouping="8x2"` (CNCS) or `grouping="4x2"` (SEQUOIA) without
explicit `grouping_file=...`.

You can always bypass this resolver by providing `grouping_file` directly.
"""
function grouping_xml_path(instrument::Union{Symbol,AbstractString}, grouping::AbstractString)
    g = strip(String(grouping))
    isempty(g) && throw(ArgumentError("grouping_xml_path: grouping is empty"))
    instkey = _grouping_instkey(instrument)

    candidates = String[]
    if instkey == "CNCS"
        # Allow a couple naming styles
        append!(candidates, [
            "CNCS_$(g).xml",
            "CNCS_$(g)_grouping.xml",
            "CNCS_$(g)_groupingfile.xml",
        ])
    elseif instkey == "SEQ"
        append!(candidates, [
            "SEQ_$(g)_grouping.xml",
            "SEQ_$(g).xml",
            "SEQUOIA_$(g)_grouping.xml",
            "SEQUOIA_$(g).xml",
        ])
    else
        append!(candidates, [
            "$(instkey)_$(g).xml",
            "$(instkey)_$(g)_grouping.xml",
        ])
    end

    dirs = _grouping_search_dirs()
    for d in dirs
        for f in candidates
            p = joinpath(d, f)
            isfile(p) && return p
        end
    end

    throw(ArgumentError("Grouping XML not found for instrument=$(instkey) grouping='$(g)'. Tried filenames=$(candidates) in dirs=$(dirs)"))
end

# ----------------------------
# Read Mantid grouping XML (fast regex parser)
# ----------------------------

"""
Read a Mantid grouping XML and return `Vector{Vector{Int}}` where each element is a group’s detector IDs.

Supported patterns inside each <group>:
  <detids val="..."/>
  <detector ids="..."/>
  <detector id="..."/>

This uses a lightweight regex parser to avoid EzXML API/version differences.
"""
function read_grouping_xml(path::AbstractString)
    txt = read(path, String)

    groups = Vector{Vector{Int}}()

    # Grab each <group ...> ... </group> in document order
    for m in eachmatch(r"<group\b[^>]*>(.*?)</group>"s, txt)
        inner = m.captures[1]
        dets = Int[]

        # Mantid grouping XML commonly uses <detids val="..."/>
        for d in eachmatch(r"<detids\b[^>]*\bval=\"([^\"]+)\""s, inner)
            append!(dets, _parse_intspec(replace(d.captures[1], r"\s+" => "")))
        end

        # Some docs/examples use <ids val="..."/>
        for d in eachmatch(r"<ids\b[^>]*\bval=\"([^\"]+)\""s, inner)
            append!(dets, _parse_intspec(replace(d.captures[1], r"\s+" => "")))
        end

        # Legacy-ish possibilities
        for d in eachmatch(r"<detector\b[^>]*\bids=\"([^\"]+)\""s, inner)
            append!(dets, _parse_intspec(replace(d.captures[1], r"\s+" => "")))
        end
        for d in eachmatch(r"<detector\b[^>]*\bid=\"([^\"]+)\""s, inner)
            append!(dets, _parse_intspec(replace(d.captures[1], r"\s+" => "")))
        end

        isempty(dets) || push!(groups, dets)
    end

    isempty(groups) && throw(ArgumentError("No <group ...> entries found in grouping XML: $path"))
    return groups
end


# ----------------------------
# MaskBTP (subset) analog
# ----------------------------

"""
MaskBTP analog for DetectorPixel vectors.

Currently supports:
  - Bank=... using digits inside `pixel.bank` Symbol (e.g. :bank40 -> 40)

Args:
- pixels: Vector{DetectorPixel}
- BTP: Mantid-like string, e.g. "Bank=40-50"
- mode:
    :drop (default) => drop masked pixels
    :keep           => keep only masked pixels
Returns:
  pixels2, meta::Dict
"""
function MaskBTP(pixels::Vector; BTP::AbstractString="", mode::Symbol=:drop)
    btp = strip(String(BTP))
    isempty(btp) && return pixels, Dict(:masked => 0, :kept => length(pixels))

    # Parse key=value pairs
    kv = Dict{String,String}()
    for tok in split(btp, ',')
        tok = strip(tok)
        isempty(tok) && continue
        if occursin('=', tok)
            k, v = split(tok, '=', limit=2)
            kv[strip(k)] = strip(v)
        end
    end

    banks_spec = get(kv, "Bank", "")
    banks = Set(_parse_intspec(banks_spec))

    if isempty(banks)
        # Nothing recognized => no-op
        return pixels, Dict(:masked => 0, :kept => length(pixels), :note => "MaskBTP: no recognized Bank=... spec")
    end

    keep = BitVector(undef, length(pixels))
    masked = 0
    for (i,p) in pairs(pixels)
        b = _banknum(p.bank)
        hit = (b !== nothing) && (b in banks)
        if hit
            masked += 1
        end
        keep[i] = (mode == :drop) ? !hit : (mode == :keep ? hit : !hit)
    end

    return pixels[findall(keep)], Dict(:masked => masked, :kept => count(keep), :mode => mode, :Bank => banks_spec)
end

# ----------------------------
# GroupDetectors analog
# ----------------------------

"""
GroupDetectors analog.

Provide exactly one of:
- GroupingFile="...xml"
- groups_detids=[ [detid1,detid2,...], [detidA,...], ... ]

IdMode controls how detector IDs are interpreted:
- :mantid => match against pixel.mantid_id
- :id     => match against pixel.id

Returns:
  grouped_pixels::Vector{DetectorPixel}, meta::Dict
"""
function GroupDetectors(pixels::Vector;
    GroupingFile::AbstractString="",
    groups_detids=nothing,
    IdMode::Symbol=:mantid,
)
    if (!isempty(strip(String(GroupingFile))) + (groups_detids !== nothing)) != 1
        throw(ArgumentError("Provide exactly one of GroupingFile or groups_detids"))
    end

    groups_detids === nothing && (groups_detids = read_grouping_xml(String(GroupingFile)))

    # Fast lookup: detid -> pixel index
    idmap = Dict{Int,Int}()
    if IdMode == :mantid
        for (i,p) in pairs(pixels)
            p.mantid_id > 0 && (idmap[p.mantid_id] = i)
        end
    elseif IdMode == :id
        for (i,p) in pairs(pixels)
            idmap[p.id] = i
        end
    else
        throw(ArgumentError("IdMode must be :mantid or :id"))
    end

    grouped = Vector{DetectorPixel}()
    sizehint!(grouped, length(groups_detids))

    group_members = Vector{Vector{Int}}(undef, length(groups_detids))

    # Helper to compute ψ,η from centroid r
    function _psi_eta(r)
        x,y,z = r[1], r[2], r[3]
        ρ = sqrt(x*x + z*z)
        ψ = atan(x, z)         # atan(y,x) in Julia is atan(y,x) => atan2
        η = atan(y, ρ)
        return ψ, η
    end

    for (gi, dets) in enumerate(groups_detids)
        members = Int[]
        for detid in dets
            idx = get(idmap, detid, 0)
            idx == 0 && continue
            push!(members, idx)
        end
        group_members[gi] = members

        isempty(members) && continue

        # centroid r_L
        r = zeros(eltype(pixels[1].r_L), 3)
        ΔΩ = 0.0
        for idx in members
            p = pixels[idx]
            r .+= p.r_L
            ΔΩ += p.ΔΩ
        end
        r ./= length(members)

        ψ, η = _psi_eta(r)
        # A stable tag for debugging
        bank = Symbol("g$(gi)")
        # Keep mantid_id=-1 (grouped pixel is synthetic)
        push!(grouped, DetectorPixel(length(grouped)+1, -1, r, ψ, η, 0, 0, bank, ΔΩ))
    end

    meta = Dict(
        :ngroups_requested => length(groups_detids),
        :ngroups_built => length(grouped),
        :IdMode => IdMode,
        :GroupingFile => isempty(strip(String(GroupingFile))) ? "" : String(GroupingFile),
        :group_members => group_members,
    )
    return grouped, meta
end

# ----------------------------
# GenerateGroupingPowder analog (event-less)
# ----------------------------

"""
GenerateGroupingPowder analog for pixel vectors.

Groups detectors by scattering angle 2θ in bins of width `AngleStep` degrees.
Writes a Mantid-style grouping XML to `GroupingFilename` and returns (GroupingFilename, groups_detids).

IdMode:
- :mantid => write pixel.mantid_id
- :id     => write pixel.id
"""
function GenerateGroupingPowder(; InputPixels, AngleStep::Real=0.5, GroupingFilename::AbstractString, IdMode::Symbol=:mantid)
    step = float(AngleStep)
    step > 0 || throw(ArgumentError("AngleStep must be > 0"))

    # Bin by 2θ (degrees). Use ψ/η if available.
    groups = Dict{Int,Vector{Int}}()
    for p in InputPixels
        # 2θ = angle between +z and rhat: acos(cosψ*cosη)
        c = cos(p.ψ) * cos(p.η)
        c = clamp(c, -1.0, 1.0)
        twotheta = acos(c) * (180/pi)
        bid = Int(floor(twotheta / step))
        detid = (IdMode == :mantid) ? p.mantid_id : p.id
        detid <= 0 && continue
        push!(get!(groups, bid, Int[]), detid)
    end

    isempty(groups) && throw(ArgumentError("GenerateGroupingPowder produced 0 groups (no valid detector IDs)."))

    # Write a simple grouping XML
    # (Mantid accepts group name strings; we use integer bin ids.)
    open(String(GroupingFilename), "w") do io
        println(io, "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>")
        println(io, "<detector-grouping>")
        for bid in sort!(collect(keys(groups)))
            dets = groups[bid]
            println(io, "  <group name=\"$(bid)\">")
            println(io, "    <detids val=\"", join(dets, ", "), "\"/>")
            println(io, "  </group>")
        end
        println(io, "</detector-grouping>")
    end

    return String(GroupingFilename), [groups[k] for k in sort!(collect(keys(groups)))]
end

# ----------------------------
# apply_grouping_masking (one-shot convenience)
# ----------------------------

"""
Apply (optional) masking and grouping to a pixel list.

Keywords mirror Mantid-ish concepts:
- instrument: used only to resolve built-in grouping XML assets when `grouping_file == nothing`
- grouping:
    "" / "none"   => no grouping
    "powder"      => GenerateGroupingPowder + GroupDetectors (writes file in outdir)
    otherwise     => resolve grouping XML + GroupDetectors
- grouping_file: explicit XML path (bypasses resolver)
- mask_btp: MaskBTP string (e.g. "Bank=40-50"), empty => no masking
- mask_mode: :drop or :keep
- outdir: directory for generated powder grouping xml
- angle_step: AngleStep for powder grouping (deg)
- return_meta: return extra metadata

Returns:
  pixels2 (Vector{DetectorPixel}) or (pixels2, meta) if return_meta=true
"""
function apply_grouping_masking(pixels::Vector;
    instrument::Union{Symbol,AbstractString}=:CNCS,
    grouping::AbstractString="",
    grouping_file=nothing,
    mask_btp::AbstractString="",
    mask_mode::Symbol=:drop,
    outdir::AbstractString="out",
    angle_step::Real=0.5,
    return_meta::Bool=false,
)
    meta = Dict{Symbol,Any}()

    # 1) Mask
    pixels1, meta_mask = MaskBTP(pixels; BTP=mask_btp, mode=mask_mode)
    meta[:mask] = meta_mask
    meta[:after_mask] = length(pixels1)

    # 2) Group
    g = lowercase(strip(String(grouping)))
    if isempty(g) || g == "none"
        meta[:grouping] = Dict(:mode => :none)
        pixels2 = pixels1
    elseif g == "powder"
        mkpath(String(outdir))
        gf = joinpath(String(outdir), "powdergroupfile.xml")
        gf, groups_detids = GenerateGroupingPowder(; InputPixels=pixels1, AngleStep=angle_step, GroupingFilename=gf, IdMode=:mantid)
        pixels2, meta_g = GroupDetectors(pixels1; GroupingFile=gf, IdMode=:mantid)
        meta[:grouping] = merge(meta_g, Dict(:mode => :powder, :AngleStep => angle_step))
    else
        gf = grouping_file === nothing ? grouping_xml_path(instrument, g) : String(grouping_file)
        pixels2, meta_g = GroupDetectors(pixels1; GroupingFile=gf, IdMode=:mantid)
        meta[:grouping] = merge(meta_g, Dict(:mode => :file, :grouping => g))
    end
    meta[:after_groupmask] = length(pixels2)

    return return_meta ? (pixels2, meta) : pixels2
end

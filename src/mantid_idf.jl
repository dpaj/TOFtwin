module MantidIDF

using EzXML
using LinearAlgebra
using StaticArrays: SVector, SMatrix
using Serialization
using SHA

# Bring parent module into scope (this is the RIGHT way for a submodule)
import ..TOFtwin

# -------------------------
# Session cache (keeps the IDF-loaded instrument in memory across repeated demos)
# -------------------------

const _IDF_MEMCACHE = Dict{Any,Any}()

# Normalize r_samp_override for use in cache keys (avoid huge SVector keys / type instability)
_norm_r_override(x) = x === nothing ? nothing : (Float64(x[1]), Float64(x[2]), Float64(x[3]))

# -------------------------
# Disk cache (persists across Julia sessions)
# -------------------------

const _IDF_DISKCACHE_VERSION = 1

"""A small fingerprint so we can invalidate disk caches when the IDF changes."""
function _idf_disk_fingerprint(idf_src::AbstractString)
    s = strip(idf_src)
    if startswith(s, "<") || startswith(s, "<?xml")
        # NOTE: Base.hash is salted per Julia session; use a stable digest
        # so disk caches can be reused across runs.
        return (kind=:xml, sha1=bytes2hex(sha1(codeunits(s))))
    end

    p = abspath(String(idf_src))
    st = stat(p)
    # mtime is a Float64 in seconds; include file size too.
    return (kind=:file, path=p, mtime=st.mtime, size=st.size)
end

function _default_disk_cache_dir(idf_src::AbstractString)
    s = strip(idf_src)
    if startswith(s, "<") || startswith(s, "<?xml")
        return joinpath(pwd(), ".toftwin_cache")
    else
        return joinpath(dirname(abspath(String(idf_src))), ".toftwin_cache")
    end
end

function _disk_cache_path(idf_src::AbstractString, key; cache_dir::Union{Nothing,AbstractString}=nothing)
    dir = cache_dir === nothing ? _default_disk_cache_dir(idf_src) : String(cache_dir)
    mkpath(dir)

    base = strip(idf_src)
    stem = (startswith(base, "<") || startswith(base, "<?xml")) ? "idf" : basename(String(idf_src))
    # NOTE: Base.hash is salted per Julia session, so do NOT use it for a
    # persistent filename. Use a stable digest instead.
    id = bytes2hex(sha1(codeunits(repr(key))))
    return joinpath(dir, "$(stem)_$(id).tofgeom.jls")
end

"""Remove all TOFtwin IDF disk-cache files in `cache_dir` (defaults to the IDF's directory)."""
function clear_mantid_idf_disk_cache!(idf_src::AbstractString; cache_dir::Union{Nothing,AbstractString}=nothing)
    dir = cache_dir === nothing ? _default_disk_cache_dir(idf_src) : String(cache_dir)
    isdir(dir) || return nothing
    for f in readdir(dir; join=true)
        endswith(f, ".tofgeom.jls") && rm(f; force=true)
    end
    return nothing
end

"""
    load_mantid_idf_diskcached(idf_src; kwargs...)

Load a processed, TOFtwin-native instrument geometry from a disk cache if available.
If not present (or invalid), it will parse the Mantid IDF, build the geometry, then
write a cache file for subsequent runs.

This is the recommended entrypoint for demo scripts that are launched as a fresh
Julia process each time.
"""
function load_mantid_idf_diskcached(idf_src::AbstractString;
    instrument_name::AbstractString = "CNCS",
    component_type::AbstractString  = "detectors",
    bank_prefix::AbstractString     = "bank",
    ψbins::Int = 720,
    ηbins::Int = 256,
    r_samp_override = nothing,
    center_on_sample::Bool = true,
    cache_dir::Union{Nothing,AbstractString} = nothing,
    rebuild::Bool = false)

    # Key must include anything that changes the returned `geo` object.
    fp  = _idf_disk_fingerprint(idf_src)
    key = (_IDF_DISKCACHE_VERSION,
           fp,
           String(instrument_name), String(component_type), String(bank_prefix),
           ψbins, ηbins, _norm_r_override(r_samp_override), center_on_sample,
           # safety: Julia major/minor can affect Serialization compatibility
           string(VERSION))

    path = _disk_cache_path(idf_src, key; cache_dir=cache_dir)

    if !rebuild && isfile(path)
        # Validate cache key
        ok = false
        cached = try
            deserialize(path)
        catch
            nothing
        end

        if cached !== nothing && cached isa NamedTuple && hasproperty(cached, :key) && hasproperty(cached, :payload)
            ok = (cached.key == key)
        end

        if ok
            @info "Loaded TOFtwin IDF disk-cache" cache = path
            return cached.payload
        else
            @warn "Ignoring stale/invalid TOFtwin IDF disk-cache; rebuilding" cache = path
        end
    end

    geo = load_mantid_idf(idf_src;
        instrument_name=instrument_name,
        component_type=component_type,
        bank_prefix=bank_prefix,
        ψbins=ψbins,
        ηbins=ηbins,
        r_samp_override=r_samp_override,
        center_on_sample=center_on_sample)

    # Write cache as (key, payload)
    tmp = path * ".tmp"
    serialize(tmp, (key=key, payload=geo))
    mv(tmp, path; force=true)
    @info "Wrote TOFtwin IDF disk-cache" cache = path

    return geo
end

"""Clear the in-memory Mantid IDF cache (useful while iterating with Revise)."""
function clear_mantid_idf_cache!()
    empty!(_IDF_MEMCACHE)
    return nothing
end

function _idf_cache_id(idf_src::AbstractString)
    s = strip(idf_src)
    if startswith(s, "<") || startswith(s, "<?xml")
        # Cache by XML-content hash (best-effort; avoids huge keys)
        return ("xml", hash(s))
    else
        # Cache by absolute path
        return ("file", abspath(String(idf_src)))
    end
end

"""Memoized wrapper around `load_mantid_idf` for the current Julia session."""
function load_mantid_idf_cached(idf_src::AbstractString;
    instrument_name::AbstractString = "CNCS",
    component_type::AbstractString  = "detectors",
    bank_prefix::AbstractString     = "bank",
    ψbins::Int = 720,
    ηbins::Int = 256,
    r_samp_override = nothing,
    center_on_sample::Bool = true,
    copy::Bool = false)

    key = (_idf_cache_id(idf_src),
           String(instrument_name), String(component_type), String(bank_prefix),
           ψbins, ηbins, _norm_r_override(r_samp_override), center_on_sample)

    geo = get!(_IDF_MEMCACHE, key) do
        load_mantid_idf(idf_src;
            instrument_name=instrument_name,
            component_type=component_type,
            bank_prefix=bank_prefix,
            ψbins=ψbins,
            ηbins=ηbins,
            r_samp_override=r_samp_override,
            center_on_sample=center_on_sample)
    end

    return copy ? deepcopy(geo) : geo
end

const I3 = SMatrix{3,3,Float64,9}(1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0)

# -------------------------
# XML helpers (EzXML is a little version-sensitive)
# -------------------------

# Read from file path OR parse from xml text
function _read_doc(src::AbstractString)
    s = strip(src)
    if startswith(s, "<") || startswith(s, "<?xml")
        return EzXML.parsexml(String(src))
    else
        return EzXML.readxml(String(src))
    end
end

# Attribute lookup with fallback
function _getattr(node::EzXML.Node, key::AbstractString, default::AbstractString="")
    # Most EzXML versions: attributes(node) behaves like a Dict
    try
        attrs = EzXML.attributes(node)
        if attrs !== nothing && haskey(attrs, key)
            return String(attrs[key])
        end
    catch
    end

    # Fallback: node["attr"] may work
    v = try
        node[key]
    catch
        nothing
    end

    return v === nothing ? default : String(v)
end

_parsef(s::AbstractString, default::Float64=0.0) = isempty(s) ? default : parse(Float64, s)
_parsei(s::AbstractString, default::Int=0)       = isempty(s) ? default : parse(Int, s)

# Axis-angle helper for Mantid-style attributes
function _axis_angle_from(node::EzXML.Node; deg_key::AbstractString)
    deg = _parsef(_getattr(node, deg_key, "0"), 0.0)
    ax  = SVector{3,Float64}(
        _parsef(_getattr(node, "axis-x", "0"), 0.0),
        _parsef(_getattr(node, "axis-y", "0"), 0.0),
        _parsef(_getattr(node, "axis-z", "1"), 1.0),
    )
    return deg, ax
end

# Mantid can nest <rot> nodes to build composite rotations. The docs state:
# outermost applied first, then the next, etc. For column vectors (v' = R*v),
# that means left-multiplying as we walk forward in that application order.
function _R_from_rot_node(rot::EzXML.Node)
    deg, ax = _axis_angle_from(rot; deg_key="val")
    R = _rot_axis_angle(ax, deg)

    kids = EzXML.findall("./*[local-name()='rot']", rot)
    for k in kids
        Rk = _R_from_rot_node(k)
        R = Rk * R
    end
    return R
end

# Mantid IDF uses axis-angle rotations (val in degrees)
function _rot_axis_angle(axis::SVector{3,Float64}, deg::Float64)
    θ = deg * (pi/180)
    ax = axis / max(norm(axis), 1e-15)
    x,y,z = ax
    c = cos(θ); s = sin(θ); C = 1 - c

    # NOTE: StaticArrays' SMatrix constructor consumes entries in *column-major* order.
    # So we must pass (a11,a21,a31,a12,a22,a32,a13,a23,a33).
    a11 = c + x*x*C
    a12 = x*y*C - z*s
    a13 = x*z*C + y*s

    a21 = y*x*C + z*s
    a22 = c + y*y*C
    a23 = y*z*C - x*s

    a31 = z*x*C - y*s
    a32 = z*y*C + x*s
    a33 = c + z*z*C

    return SMatrix{3,3,Float64,9}(
        a11, a21, a31,
        a12, a22, a32,
        a13, a23, a33
    )
end

_xyz_from_location(loc::EzXML.Node) = begin
    # Prefer Cartesian if any of x/y/z are present
    xs = _getattr(loc, "x", "")
    ys = _getattr(loc, "y", "")
    zs = _getattr(loc, "z", "")
    if !(isempty(xs) && isempty(ys) && isempty(zs))
        x = _parsef(xs, 0.0)
        y = _parsef(ys, 0.0)
        z = _parsef(zs, 0.0)
        return SVector{3,Float64}(x,y,z)
    end

    # Mantid also supports spherical coordinates: r, t (theta, from +z), p (phi, in x-y)
    r = _parsef(_getattr(loc, "r", "0"), 0.0)
    t = _parsef(_getattr(loc, "t", "0"), 0.0) * (pi/180)
    p = _parsef(_getattr(loc, "p", "0"), 0.0) * (pi/180)

    x = r * sin(t) * cos(p)
    y = r * sin(t) * sin(p)
    z = r * cos(t)
    return SVector{3,Float64}(x,y,z)
end

function _R_from_location(loc::EzXML.Node)
    R = I3

    # a) rotation given directly on <location rot="..." axis-x="..." ...>
    deg0, ax0 = _axis_angle_from(loc; deg_key="rot")
    if deg0 != 0.0
        R0 = _rot_axis_angle(ax0, deg0)
        R = R0 * R
    end

    # b) rotations as child <rot> elements (possibly nested)
    rots = EzXML.findall("./*[local-name()='rot']", loc)
    for r in rots
        Rr = _R_from_rot_node(r)
        R = Rr * R
    end

    return R
end

# Your convention:
# ψ: horizontal angle atan(x,z)
# η: elevation atan(y, sqrt(x^2+z^2))
function _psi_eta(r::SVector{3,Float64})
    x,y,z = r
    ψ = atan(x, z)
    ρ = sqrt(x*x + z*z)
    η = atan(y, ρ)
    return ψ, η
end

# -------------------------
# DetectorPixel construction that adapts to your struct fields
# -------------------------

function _default_value(T::Type, fname::Symbol)
    if T <: Integer
        return zero(T)
    elseif T <: AbstractFloat
        return zero(T)
    elseif T == Symbol
        return :unknown
    elseif T == String
        return ""
    elseif T == Bool
        return false
    elseif T == TOFtwin.Vec3
        return TOFtwin.Vec3(0.0, 0.0, 0.0)
    else
        # If you add a new custom field type later, update this.
        error("MantidIDF: don't know default for DetectorPixel field $(fname)::$(T).")
    end
end

function _make_pixel(; id::Int, detid::Int,
    r_L::TOFtwin.Vec3, ψ::Float64, η::Float64,
    iψ::Int, iη::Int, bank::Symbol, ΔΩ::Float64)

    fns = fieldnames(TOFtwin.DetectorPixel)
    vals = Vector{Any}(undef, length(fns))

    for (j, f) in enumerate(fns)
        T = fieldtype(TOFtwin.DetectorPixel, j)

        if f == :id
            vals[j] = id
        elseif f in (:mantid_id, :detid, :detector_id, :mantidid)
            vals[j] = detid
        elseif f == :r_L
            vals[j] = r_L
        elseif f == :ψ
            vals[j] = ψ
        elseif f == :η
            vals[j] = η
        elseif f == :iψ
            vals[j] = iψ
        elseif f == :iη
            vals[j] = iη
        elseif f == :bank
            vals[j] = bank
        elseif f in (:ΔΩ, :dΩ, :solid_angle)
            vals[j] = ΔΩ
        else
            vals[j] = _default_value(T, f)
        end

        # try to convert if the field is typed (keeps you honest)
        if !(T === Any) && !(vals[j] isa T)
            vals[j] = convert(T, vals[j])
        end
    end

    return TOFtwin.DetectorPixel(vals...)
end

# Fast path: avoid per-pixel reflection if DetectorPixel matches the expected fields.
const _FAST_PIXEL_CTOR_OK = fieldnames(TOFtwin.DetectorPixel) ==
    (:id, :mantid_id, :r_L, :ψ, :η, :iψ, :iη, :bank, :ΔΩ)

# -------------------------
# Main loader
# -------------------------

"""
    load_mantid_idf(idf_path_or_xml;
        instrument_name="CNCS",
        component_type="detectors",
        bank_prefix="bank",
        ψbins=720,
        ηbins=256)

Returns a NamedTuple:
  (inst, bank, pixels, meta)

- inst   : TOFtwin.Instrument (L1 inferred from moderator↔sample-position)
- bank   : TOFtwin.DetectorBank(name, pixels) (if constructor exists; otherwise `nothing`)
- pixels : Vector{TOFtwin.DetectorPixel} with ids 1..N and Mantid ids from IDF idlist
- meta   : (L1, detid_start, detid_end, pixel_radius, pixel_height, npixels)
"""
function load_mantid_idf(idf_src::AbstractString;
    instrument_name::AbstractString = "CNCS",
    component_type::AbstractString  = "detectors",
    bank_prefix::AbstractString     = "bank",
    ψbins::Int = 720,
    ηbins::Int = 256,
    r_samp_override = nothing,
    center_on_sample::Bool = true)

    @info "Loading IDF" idf_path = idf_src

    doc  = _read_doc(idf_src)
    root = EzXML.root(doc)

    # ---- L1 from moderator and sample-position ----
    mod_locs = EzXML.findall("/*[local-name()='instrument']/*[local-name()='component' and @type='moderator']/*[local-name()='location']", root)
    samp_locs = EzXML.findall("/*[local-name()='instrument']/*[local-name()='component' and @type='sample-position']/*[local-name()='location']", root)

    isempty(mod_locs)  && error("Could not find moderator location; cannot infer L1.")
    isempty(samp_locs) && error("Could not find sample-position; cannot infer L1.")

    z_mod  = _parsef(_getattr(mod_locs[1],  "z", "0"))
    z_samp = _parsef(_getattr(samp_locs[1], "z", "0"))
    L1 = abs(z_samp - z_mod)

    r_samp = SVector{3,Float64}(
        _parsef(_getattr(samp_locs[1], "x", "0")),
        _parsef(_getattr(samp_locs[1], "y", "0")),
        _parsef(_getattr(samp_locs[1], "z", "0"))
    )

    # Allow caller override (useful for geometry sanity checks against experiments)
    if r_samp_override !== nothing
        r_samp = SVector{3,Float64}(
            Float64(r_samp_override[1]),
            Float64(r_samp_override[2]),
            Float64(r_samp_override[3])
        )
    end

    # ---- detector idlists (Mantid detector IDs) ----
    # Some IDFs (e.g. CNCS) define a single <component type="detectors" idlist="detectors"> and a
    # matching <idlist idname="detectors">. Others (e.g. SEQUOIA) define multiple instrument-level
    # components with their own idlists ("A row", "B row", ...). We support both.
    detids = Int[]

    function _push_idlist!(out::Vector{Int}, idlist_name::String)
        isempty(idlist_name) && return
        idlists = EzXML.findall("/*[local-name()='instrument']/*[local-name()='idlist' and @idname='$idlist_name']", root)
        isempty(idlists) && return

        for idnode in EzXML.findall("./*[local-name()='id']", idlists[1])
            # Either <id val="..."/> or <id start="..." end="..." [step="..."] />
            val = _getattr(idnode, "val", "")
            if !isempty(val)
                push!(out, _parsei(val, 0))
                continue
            end

            s = _getattr(idnode, "start", "")
            e = _getattr(idnode, "end", "")
            isempty(s) && continue
            isempty(e) && continue

            st = _parsei(s, 0)
            en = _parsei(e, st)
            step = _parsei(_getattr(idnode, "step", "1"), 1)
            step == 0 && (step = 1)

            if st <= en
                for v in st:step:en
                    push!(out, v)
                end
            else
                # Rare, but handle descending ranges
                for v in st:-abs(step):en
                    push!(out, v)
                end
            end
        end
    end

    # CNCS-style: component type="detectors" has an idlist
    det_comp = EzXML.findall("/*[local-name()='instrument']/*[local-name()='component' and @type='detectors' and @idlist]", root)
    if !isempty(det_comp)
        _push_idlist!(detids, _getattr(det_comp[1], "idlist", ""))
    else
        # SEQUOIA-style: instrument-level detector group components each have an idlist
        root_comps = EzXML.findall("/*[local-name()='instrument']/*[local-name()='component' and @idlist]", root)
        for rc in root_comps
            idname = _getattr(rc, "idlist", "")
            idname == "monitors" && continue
            _push_idlist!(detids, idname)
        end
    end

    detid_start = isempty(detids) ? 0  : minimum(detids)
    detid_end   = isempty(detids) ? -1 : maximum(detids)
    # ---- collect <type name="..."> nodes into dict ----
    types = Dict{String,EzXML.Node}()
    type_nodes = EzXML.findall("/*[local-name()='instrument']/*[local-name()='type']", root)
    for t in type_nodes
        nm = _getattr(t, "name", "")
        isempty(nm) && continue
        types[nm] = t
    end
    has_det_type = haskey(types, component_type)

    # ---- pixel geometry for ΔΩ estimate ----
    pix_radius = 0.0
    pix_height = 0.0
    if haskey(types, "pixel")
        tn = types["pixel"]
        rnode = EzXML.findall(".//*[local-name()='radius']", tn)
        hnode = EzXML.findall(".//*[local-name()='height']", tn)
        if !isempty(rnode); pix_radius = _parsef(_getattr(rnode[1], "val", "0")); end
        if !isempty(hnode); pix_height = _parsef(_getattr(hnode[1], "val", "0")); end
    end

    pixel_area = (2*pix_radius) * pix_height

    # ---- expand hierarchy down to pixels ----
    pix_pos  = SVector{3,Float64}[]
    pix_bank = Symbol[]
    pix_ψ    = Float64[]
    pix_η    = Float64[]
    pix_dΩ   = Float64[]

    # If we have a detector idlist, we can preallocate roughly the right number of pixels.
    if !isempty(detids)
        np_hint = length(detids)
        sizehint!(pix_pos,  np_hint)
        sizehint!(pix_bank, np_hint)
        sizehint!(pix_ψ,    np_hint)
        sizehint!(pix_η,    np_hint)
        sizehint!(pix_dΩ,   np_hint)
    elseif detid_end >= detid_start
        np_hint = detid_end - detid_start + 1
        if np_hint > 0
            sizehint!(pix_pos,  np_hint)
            sizehint!(pix_bank, np_hint)
            sizehint!(pix_ψ,    np_hint)
            sizehint!(pix_η,    np_hint)
            sizehint!(pix_dΩ,   np_hint)
        end
    end

    function expand_type(type_name::String,
                         t_parent::SVector{3,Float64},
                         R_parent::SMatrix{3,3,Float64,9},
                         bank_sym::Symbol)

        tnode = get(types, type_name, nothing)
        tnode === nothing && return

        comps = EzXML.findall("./*[local-name()='component']", tnode)
        for comp in comps
            child_type = _getattr(comp, "type", "")
            isempty(child_type) && continue

            locs = EzXML.findall("./*[local-name()='location']", comp)

            if isempty(locs)
                # no explicit placement
                if child_type == "pixel"
                    r_abs = t_parent
                    r_rel = r_abs - r_samp
                    ψ, η = _psi_eta(r_rel)
                    L2 = norm(r_rel)
                    dΩ = (pixel_area > 0 && L2 > 0) ? pixel_area / (L2^2) : 1.0

                    push!(pix_pos, center_on_sample ? r_rel : r_abs)
                    push!(pix_bank, bank_sym)
                    push!(pix_ψ, ψ)
                    push!(pix_η, η)
                    push!(pix_dΩ, dΩ)
                else
                    expand_type(child_type, t_parent, R_parent, bank_sym)
                end
                continue
            end

            for loc in locs
                t_loc = _xyz_from_location(loc)
                R_loc = _R_from_location(loc)

                t_new = t_parent + R_parent * t_loc
                R_new = R_parent * R_loc

                if child_type == "pixel"
                    r_abs = t_new
                    r_rel = r_abs - r_samp
                    ψ, η = _psi_eta(r_rel)

                    L2 = norm(r_rel)
                    dΩ = (pixel_area > 0 && L2 > 0) ? pixel_area / (L2^2) : 1.0

                    push!(pix_pos, center_on_sample ? r_rel : r_abs)
                    push!(pix_bank, bank_sym)
                    push!(pix_ψ, ψ)
                    push!(pix_η, η)
                    push!(pix_dΩ, dΩ)
                else
                    expand_type(child_type, t_new, R_new, bank_sym)
                end
            end
        end
    end

    if has_det_type
        det_type = types[component_type]
        bank_comps = EzXML.findall("./*[local-name()='component']", det_type)

        for bc in bank_comps
            btype = _getattr(bc, "type", "")
            isempty(btype) && continue
            startswith(btype, bank_prefix) || continue

            locs = EzXML.findall("./*[local-name()='location']", bc)
            isempty(locs) && continue

            # detectors -> bank placement is specified here
            for loc in locs
                t0  = _xyz_from_location(loc)
                R0  = _R_from_location(loc)
                expand_type(btype, t0, R0, Symbol(btype))
            end
        end
    else
        # Fallback for IDFs like SEQUOIA: instrument-level components carry idlists (A row/B row/...).
        root_comps = EzXML.findall("/*[local-name()='instrument']/*[local-name()='component' and @idlist]", root)
        for rc in root_comps
            idname = _getattr(rc, "idlist", "")
            idname == "monitors" && continue

            rtype = _getattr(rc, "type", "")
            isempty(rtype) && continue
            haskey(types, rtype) || (@warn "No <type name=\"$rtype\"> for idlist component \"$idname\"; skipping."; continue)

            bank_sym = Symbol(replace(rtype, " " => "_"))

            locs = EzXML.findall("./*[local-name()='location']", rc)
            if isempty(locs)
                expand_type(rtype, SVector{3,Float64}(0.0, 0.0, 0.0), I3, bank_sym)
            else
                for loc in locs
                    t0 = _xyz_from_location(loc)
                    R0 = _R_from_location(loc)
                    expand_type(rtype, t0, R0, bank_sym)
                end
            end
        end
    end
    np = length(pix_pos)

    # sanity vs idlist length if present
    if !isempty(detids) && length(detids) != np
        @warn "Pixel count ($np) does not match IDF idlist count ($(length(detids))). Mantid IDs may be off."
    end
    pixels = Vector{TOFtwin.DetectorPixel}(undef, np)

    @inbounds for i in 1:np
        rL = TOFtwin.Vec3(pix_pos[i]...)
        ψ  = pix_ψ[i]
        η  = pix_η[i]

        iψ = clamp(floor(Int, (ψ + pi)/(2pi) * ψbins) + 1, 1, ψbins)
        iη = clamp(floor(Int, (η + (pi/2))/pi * ηbins) + 1, 1, ηbins)

        bank = pix_bank[i]
        dΩ   = pix_dΩ[i]

        detid = !isempty(detids) ? detids[i] : (detid_start + (i - 1))

        if _FAST_PIXEL_CTOR_OK
            pixels[i] = TOFtwin.DetectorPixel(i, detid, rL, ψ, η, iψ, iη, bank, dΩ)
        else
            pixels[i] = _make_pixel(id=i, detid=detid, r_L=rL, ψ=ψ, η=η,
                                    iψ=iψ, iη=iη, bank=bank, ΔΩ=dΩ)
        end
    end

    inst = TOFtwin.Instrument(name=instrument_name, L1=L1, pixels=pixels)

    bank = TOFtwin.DetectorBank(String(instrument_name), pixels)

    meta = (L1=L1,
            detid_start=detid_start,
            detid_end=detid_end,
            r_samp=r_samp,
            center_on_sample=center_on_sample,
            pixel_radius=pix_radius,
            pixel_height=pix_height,
            npixels=np)

    return (inst=inst, bank=bank, pixels=pixels, meta=meta)
end

export load_mantid_idf,
       load_mantid_idf_cached,
       load_mantid_idf_diskcached,
       clear_mantid_idf_cache!,
       clear_mantid_idf_disk_cache!

end # module MantidIDF

module MantidIDF

using EzXML
using LinearAlgebra
using StaticArrays: SVector, SMatrix

# Bring parent module into scope (this is the RIGHT way for a submodule)
import ..TOFtwin

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

# Mantid IDF uses axis-angle rotations (val in degrees)
function _rot_axis_angle(axis::SVector{3,Float64}, deg::Float64)
    θ = deg * (pi/180)
    ax = axis / max(norm(axis), 1e-15)
    x,y,z = ax
    c = cos(θ); s = sin(θ); C = 1 - c
    return SMatrix{3,3,Float64,9}(
        c + x*x*C,   x*y*C - z*s, x*z*C + y*s,
        y*x*C + z*s, c + y*y*C,   y*z*C - x*s,
        z*x*C - y*s, z*y*C + x*s, c + z*z*C
    )
end

_xyz_from_location(loc::EzXML.Node) = begin
    x = _parsef(_getattr(loc, "x", "0"))
    y = _parsef(_getattr(loc, "y", "0"))
    z = _parsef(_getattr(loc, "z", "0"))
    SVector{3,Float64}(x,y,z)
end

function _R_from_location(loc::EzXML.Node)
    R = I3
    rots = EzXML.findall("./*[local-name()='rot']", loc)
    for r in rots
        val = _parsef(_getattr(r, "val", "0"))
        ax  = SVector{3,Float64}(
            _parsef(_getattr(r, "axis-x", "0")),
            _parsef(_getattr(r, "axis-y", "0")),
            _parsef(_getattr(r, "axis-z", "1"))
        )
        R = R * _rot_axis_angle(ax, val)
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
    ηbins::Int = 256)

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

    # ---- idlist for detectors (Mantid detector ids) ----
    detid_start = 0
    detid_end   = -1

    # Find <component type="detectors" idlist="...">
    det_comp = EzXML.findall("/*[local-name()='instrument']/*[local-name()='component' and @type='detectors']", root)
    if !isempty(det_comp)
        idlist_name = _getattr(det_comp[1], "idlist", "")
        if !isempty(idlist_name)
            idlists = EzXML.findall("/*[local-name()='instrument']/*[local-name()='idlist' and @idname='$idlist_name']", root)
            if !isempty(idlists)
                rng = EzXML.findall("./*[local-name()='id']", idlists[1])
                if !isempty(rng)
                    detid_start = _parsei(_getattr(rng[1], "start", "0"), 0)
                    detid_end   = _parsei(_getattr(rng[1], "end",   "-1"), -1)
                end
            end
        end
    end

    # ---- collect <type name="..."> nodes into dict ----
    types = Dict{String,EzXML.Node}()
    type_nodes = EzXML.findall("/*[local-name()='instrument']/*[local-name()='type']", root)
    for t in type_nodes
        nm = _getattr(t, "name", "")
        isempty(nm) && continue
        types[nm] = t
    end
    haskey(types, component_type) || error("IDF has no <type name=\"$component_type\">")

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

    # ---- expand hierarchy down to pixels ----
    pix_pos  = SVector{3,Float64}[]
    pix_bank = Symbol[]
    pix_ψ    = Float64[]
    pix_η    = Float64[]
    pix_dΩ   = Float64[]

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
                    rL = t_parent
                    r  = rL - r_samp
                    ψ, η = _psi_eta(r)
                    push!(pix_pos, rL)
                    push!(pix_bank, bank_sym)
                    push!(pix_ψ, ψ)
                    push!(pix_η, η)
                    push!(pix_dΩ, 1.0)
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
                    rL = t_new
                    r  = rL - r_samp
                    ψ, η = _psi_eta(r)

                    L2 = norm(r)
                    area = (2*pix_radius) * pix_height
                    dΩ = (area > 0 && L2 > 0) ? area / (L2^2) : 1.0

                    push!(pix_pos, rL)
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

    det_type = types[component_type]
    bank_comps = EzXML.findall("./*[local-name()='component']", det_type)

    for bc in bank_comps
        btype = _getattr(bc, "type", "")
        isempty(btype) && continue
        startswith(btype, bank_prefix) || continue

        locs = EzXML.findall("./*[local-name()='location']", bc)
        isempty(locs) && continue

        # detectors -> bank placement is specified here
        loc = locs[1]
        t0  = _xyz_from_location(loc)
        R0  = _R_from_location(loc)

        expand_type(btype, t0, R0, Symbol(btype))
    end

    np = length(pix_pos)

    # sanity vs idlist range if present
    if detid_end >= detid_start
        expected = detid_end - detid_start + 1
        if expected != np
            @warn "Pixel count ($np) does not match IDF detector idlist range ($expected). Mantid IDs may be off."
        end
    end

    pixels = Vector{TOFtwin.DetectorPixel}(undef, np)

    for i in 1:np
        rL = TOFtwin.Vec3(pix_pos[i]...)
        ψ  = pix_ψ[i]
        η  = pix_η[i]

        iψ = clamp(floor(Int, (ψ + pi)/(2pi) * ψbins) + 1, 1, ψbins)
        iη = clamp(floor(Int, (η + (pi/2))/pi * ηbins) + 1, 1, ηbins)

        bank = pix_bank[i]
        dΩ   = pix_dΩ[i]

        detid = detid_start + (i - 1)

        pixels[i] = _make_pixel(id=i, detid=detid, r_L=rL, ψ=ψ, η=η,
                                iψ=iψ, iη=iη, bank=bank, ΔΩ=dΩ)
    end

    inst = TOFtwin.Instrument(name=instrument_name, L1=L1, pixels=pixels)

    bank = TOFtwin.DetectorBank(String(instrument_name), pixels)

    meta = (L1=L1,
            detid_start=detid_start,
            detid_end=detid_end,
            pixel_radius=pix_radius,
            pixel_height=pix_height,
            npixels=np)

    return (inst=inst, bank=bank, pixels=pixels, meta=meta)
end

export load_mantid_idf

end # module MantidIDF

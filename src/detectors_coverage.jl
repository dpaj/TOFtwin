using StaticArrays
using LinearAlgebra

const Vec3 = SVector{3,Float64}

"""
A single detector pixel.

Fields:
- id:        1-based contiguous index (used for array indexing)
- mantid_id: original Mantid detector ID (or -1 if synthetic)
- r_L:       position in Lab frame (meters)
- ψ:         horizontal angle (rad), ψ=0 forward (+z), ψ>0 toward +x
- η:         elevation angle (rad), η>0 up (+y)
- iψ:        horizontal-bin index (for samplers like AngularDecimate)
- iη:        elevation-bin index
- bank:      symbol tag (e.g. :left/:right or :bank17, etc.)
- ΔΩ:        approximate solid angle per pixel (sr)
"""
struct DetectorPixel
    id::Int
    mantid_id::Int
    r_L::Vec3
    ψ::Float64
    η::Float64
    iψ::Int
    iη::Int
    bank::Symbol
    ΔΩ::Float64
end

"""
A bank is just a named collection of pixels (optionally from IDF).
"""
struct DetectorBank
    name::String
    pixels::Vector{DetectorPixel}
end

# midpoint grid helper: n bin-centers between [a,b]
grid(a, b, n) = [a + (i+0.5)*(b-a)/n for i in 0:n-1]

"Pixel position on a cylinder of equatorial radius L2 about the z axis."
pos_cylinder(L2, ψ, η) = Vec3(L2*sin(ψ), L2*tan(η), L2*cos(ψ))

"Pixel position on a sphere of radius L2 (true distance fixed)."
pos_sphere(L2, ψ, η) = L2 * Vec3(cos(η)*sin(ψ), sin(η), cos(ψ)*cos(η))

"Compute (ψ,η) from a lab position r_L."
@inline function angles_from_rL(r::Vec3)
    ψ = atan(r[1], r[3])                 # atan(x, z)
    η = atan(r[2], hypot(r[1], r[3]))    # elevation
    return ψ, η
end

"""
Approximate solid angle for a small angular cell around (ψ,η):
ΔΩ ≈ cos(η) * Δψ * Δη.
"""
@inline solid_angle_cell(η, Δψ, Δη) = abs(cos(η) * Δψ * Δη)

"""
Generate detector pixels from angular coverage.

Conventions:
  +z = beam (k_i), +y = up, +x = right (looking downstream).
ψ: horizontal angle in x–z plane; ψ>0 toward +x (right), ψ<0 toward -x (left).
η: elevation; η>0 up.

left_deg  = (10, 140) means LEFT side; mapped to ψ ∈ [-140, -10] deg
right_deg = (10, 50)  means RIGHT side; mapped to ψ ∈ [10, 50] deg
"""
function pixels_from_coverage(; L2=3.5,
    left_deg=(10.0, 140.0),
    right_deg=(10.0, 50.0),
    oop_deg=(-16.0, 16.0),
    nψ_left=180, nψ_right=80, nη=33,
    surface::Symbol = :cylinder)

    pos = surface === :cylinder ? pos_cylinder :
          surface === :sphere   ? pos_sphere   :
          error("surface must be :cylinder or :sphere")

    ψLmin, ψLmax = deg2rad.((-left_deg[2], -left_deg[1]))   # [-140, -10]
    ψRmin, ψRmax = deg2rad.(right_deg)                      # [10, 50]
    ηmin,  ηmax  = deg2rad.(oop_deg)

    ηs  = grid(ηmin, ηmax, nη)
    ψLs = grid(ψLmin, ψLmax, nψ_left)
    ψRs = grid(ψRmin, ψRmax, nψ_right)

    # angular cell sizes (midpoint grid => uniform)
    ΔψL = (ψLmax - ψLmin)/nψ_left
    ΔψR = (ψRmax - ψRmin)/nψ_right
    Δη  = (ηmax  - ηmin)/nη

    pix = DetectorPixel[]
    id = 1

    for (iη, η) in enumerate(ηs), (iψ, ψ) in enumerate(ψLs)
        r = pos(L2, ψ, η)
        ΔΩ = solid_angle_cell(η, ΔψL, Δη)
        push!(pix, DetectorPixel(id, -1, r, ψ, η, iψ, iη, :left, ΔΩ))
        id += 1
    end
    for (iη, η) in enumerate(ηs), (iψ, ψ) in enumerate(ψRs)
        r = pos(L2, ψ, η)
        ΔΩ = solid_angle_cell(η, ΔψR, Δη)
        push!(pix, DetectorPixel(id, -1, r, ψ, η, iψ, iη, :right, ΔΩ))
        id += 1
    end

    return pix
end

"Convenience wrapper used in your demos."
function bank_from_coverage(; name="example", kwargs...)
    DetectorBank(name, pixels_from_coverage(; kwargs...))
end

"Quick numeric sanity-check."
function summarize_pixels(pix::AbstractVector{DetectorPixel})
    xs = [p.r_L[1] for p in pix]
    ys = [p.r_L[2] for p in pix]
    zs = [p.r_L[3] for p in pix]
    ψs = [p.ψ for p in pix]
    ηs = [p.η for p in pix]
    dΩ = [p.ΔΩ for p in pix]

    return (
        N = length(pix),
        x = (minimum(xs), maximum(xs)),
        y = (minimum(ys), maximum(ys)),
        z = (minimum(zs), maximum(zs)),
        ψ_deg = (rad2deg(minimum(ψs)), rad2deg(maximum(ψs))),
        η_deg = (rad2deg(minimum(ηs)), rad2deg(maximum(ηs))),
        ΔΩ_sr = (minimum(dΩ), maximum(dΩ), mean(dΩ)),
    )
end

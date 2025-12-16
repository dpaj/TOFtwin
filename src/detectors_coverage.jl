using StaticArrays
const Vec3 = SVector{3,Float64}

struct DetectorPixel
    id::Int
    r_L::Vec3      # position in Lab frame (meters)
end

# midpoint grid helper
grid(a, b, n) = [a + (i+0.5)*(b-a)/n for i in 0:n-1]

"Pixel position on a cylinder of radius L2 about the z axis."
pos_cylinder(L2, ψ, η) = Vec3(L2*sin(ψ), L2*tan(η), L2*cos(ψ))

"Pixel position on a sphere of radius L2 (true distance fixed)."
pos_sphere(L2, ψ, η) = L2 * Vec3(cos(η)*sin(ψ), sin(η), cos(η)*cos(ψ))

"""
Generate detector pixels from angular coverage.

Conventions (right-handed):
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

    pix = DetectorPixel[]
    id = 1

    for η in ηs, ψ in ψLs
        push!(pix, DetectorPixel(id, pos(L2, ψ, η))); id += 1
    end
    for η in ηs, ψ in ψRs
        push!(pix, DetectorPixel(id, pos(L2, ψ, η))); id += 1
    end

    return pix
end

"Summarize pixel cloud and infer (ψ, η) ranges from positions."
function summarize_pixels(pix::Vector{DetectorPixel})
    pts = getfield.(pix, :r_L)
    xs  = getindex.(pts, 1)
    ys  = getindex.(pts, 2)
    zs  = getindex.(pts, 3)

    # inferred angles from position
    ψ = atan.(xs, zs)                           # horizontal angle about +y, ψ=0 at +z
    ρ = sqrt.(xs.^2 .+ zs.^2)
    η = atan.(ys, ρ)                            # elevation

    return (
        N = length(pix),
        x = (minimum(xs), maximum(xs)),
        y = (minimum(ys), maximum(ys)),
        z = (minimum(zs), maximum(zs)),
        ψ_deg = (rad2deg(minimum(ψ)), rad2deg(maximum(ψ))),
        η_deg = (rad2deg(minimum(η)), rad2deg(maximum(η))),
    )
end

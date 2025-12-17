using StaticArrays

const Vec3 = SVector{3,Float64}

"""
A single detector pixel.

Fields:
- id: unique integer id (global across all banks generated together)
- r_L: position in Lab frame (meters)
- ψ: horizontal angle (radians), ψ=0 forward (+z), ψ>0 toward +x (right)
- η: elevation angle (radians), η>0 up (+y)
- iψ: horizontal-bin index (1..nψ for that bank)
- iη: elevation-bin index (1..nη)
- bank: :left or :right
- ΔΩ: approximate solid-angle weight (sr) for this angular patch
"""
struct DetectorPixel
    id::Int
    r_L::Vec3
    ψ::Float64
    η::Float64
    iψ::Int
    iη::Int
    bank::Symbol
    ΔΩ::Float64
end

# midpoint grid helper: n bin-centers between [a,b]
grid(a, b, n::Int) = [a + (i+0.5)*(b-a)/n for i in 0:n-1]

# midpoint grid helper that also returns the step size
function grid_with_step(a, b, n::Int)
    Δ = (b - a) / n
    xs = [a + (i+0.5)*Δ for i in 0:n-1]
    return xs, Δ
end

"Pixel position on a cylinder of equatorial radius L2 about the z axis."
pos_cylinder(L2, ψ, η) = Vec3(L2*sin(ψ), L2*tan(η), L2*cos(ψ))

"Pixel position on a sphere of radius L2 (true distance fixed)."
pos_sphere(L2, ψ, η) = L2 * Vec3(cos(η)*sin(ψ), sin(η), cos(ψ)*cos(η))

"""
Generate detector pixels from angular coverage.

Conventions (right-handed):
  +z = beam (k_i), +y = up, +x = right (looking downstream).
ψ: horizontal angle in x–z plane; ψ>0 toward +x (right), ψ<0 toward -x (left).
η: elevation; η>0 up.

left_deg  = (10, 140) means LEFT side; mapped to ψ ∈ [-140, -10] deg
right_deg = (10, 50)  means RIGHT side; mapped to ψ ∈ [10, 50] deg

Sampling:
- Uniform midpoint grid in (ψ, η) with nψ_* × nη points per bank.
- For :cylinder, L2 is the equatorial-plane radius; y is set via y = L2*tan(η).

ΔΩ:
- Approximate solid-angle patch per pixel as ΔΩ ≈ cos(η) * Δψ * Δη
  where Δψ and Δη are the angular bin widths used to generate the midpoint grid.
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

    ηs,  Δη  = grid_with_step(ηmin,  ηmax,  nη)
    ψLs, ΔψL = grid_with_step(ψLmin, ψLmax, nψ_left)
    ψRs, ΔψR = grid_with_step(ψRmin, ψRmax, nψ_right)

    pix = DetectorPixel[]
    id = 1

    for (iη, η) in enumerate(ηs), (iψ, ψ) in enumerate(ψLs)
        ΔΩ = abs(cos(η)) * abs(ΔψL) * abs(Δη)
        push!(pix, DetectorPixel(id, pos(L2, ψ, η), ψ, η, iψ, iη, :left, ΔΩ))
        id += 1
    end

    for (iη, η) in enumerate(ηs), (iψ, ψ) in enumerate(ψRs)
        ΔΩ = abs(cos(η)) * abs(ΔψR) * abs(Δη)
        push!(pix, DetectorPixel(id, pos(L2, ψ, η), ψ, η, iψ, iη, :right, ΔΩ))
        id += 1
    end

    return pix
end

"Quick numeric sanity-check."
function summarize_pixels(pix::AbstractVector{DetectorPixel})
    xs = [p.r_L[1] for p in pix]
    ys = [p.r_L[2] for p in pix]
    zs = [p.r_L[3] for p in pix]
    ψs = [p.ψ for p in pix]
    ηs = [p.η for p in pix]
    Ωs = [p.ΔΩ for p in pix]

    Ω_left  = sum(p.ΔΩ for p in pix if p.bank == :left)
    Ω_right = sum(p.ΔΩ for p in pix if p.bank == :right)

    return (
        N = length(pix),
        N_left  = count(p -> p.bank == :left,  pix),
        N_right = count(p -> p.bank == :right, pix),
        x = (minimum(xs), maximum(xs)),
        y = (minimum(ys), maximum(ys)),
        z = (minimum(zs), maximum(zs)),
        ψ_deg = (rad2deg(minimum(ψs)), rad2deg(maximum(ψs))),
        η_deg = (rad2deg(minimum(ηs)), rad2deg(maximum(ηs))),
        nη = maximum(p.iη for p in pix),
        ΔΩ = (minimum(Ωs), maximum(Ωs)),
        Ω_sum = sum(Ωs),
        Ω_left = Ω_left,
        Ω_right = Ω_right,
    )
end

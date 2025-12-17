using TOFtwin

# -----------------------------------------------------------------------------
# Backend selection:
#   TOFTWIN_BACKEND=gl    (interactive)
#   TOFTWIN_BACKEND=cairo (PNG, good when GL colors are weird)
# -----------------------------------------------------------------------------
backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))

if backend == "cairo"
    @info "Using CairoMakie backend"
    using CairoMakie
else
    @info "Using GLMakie backend"
    using GLMakie
end

# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
pix = TOFtwin.pixels_from_coverage(L2=3.5, surface=:cylinder)
#pix = pix_used

pts = getfield.(pix, :r_L)
xs  = getindex.(pts, 1)
ys  = getindex.(pts, 2)
zs  = getindex.(pts, 3)

# Separate banks without relying on color (marker shapes work even if GL colors are off)
idxR = xs .>= 0
idxL = .!idxR

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig = Figure(resolution = (1100, 850))
ax  = Axis3(fig[1, 1],
    aspect = :data,
    xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)",
    title  = "TOFtwin: detector coverage (L2=3.5 m, cylinder)"
)

# Sample at origin
scatter!(ax, [0.0], [0.0], [0.0], markersize = 18)

# Banks: different markers so it’s readable even if color is funky
scatter!(ax, xs[idxL], ys[idxL], zs[idxL]; markersize = 4, marker = :circle, label = "left (x < 0)")
scatter!(ax, xs[idxR], ys[idxR], zs[idxR]; markersize = 4, marker = :rect,   label = "right (x ≥ 0)")

# Beam direction (+z)
lines!(ax, [0.0, 0.0], [0.0, 0.0], [0.0, 4.0])

# A ring at y=0 to show the L2 cylinder in the equatorial plane
L2 = 3.5
θ = range(-pi, pi; length=361)
ringx = L2 .* sin.(θ)
ringy = 0.0 .* θ
ringz = L2 .* cos.(θ)
lines!(ax, ringx, ringy, ringz)

axislegend(ax)

# Save a snapshot either way (nice for logging / CI)
out = get(ENV, "TOFTWIN_OUT", "detector_view.png")
save(out, fig)
@info "Wrote $out"

display(fig)

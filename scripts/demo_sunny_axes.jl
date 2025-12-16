using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin

# optional plotting
backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
if backend == "cairo"
    using CairoMakie
else
    using GLMakie
end

bank = bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
inst = Instrument(name="demo", L1=36.262, pixels=bank.pixels)

pix_used = sample_pixels(bank, AngularDecimate(3,2))
Ei = 12.0

L2min, L2max = minimum(inst.L2), maximum(inst.L2)
Ef_min, Ef_max = 1.0, Ei
tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ef_max)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, Ef_min)
tof_edges = collect(range(tmin, tmax; length=500+1))

sug = suggest_sunny_powder_axes(inst;
    pixels=pix_used,
    Ei_meV=Ei,
    tof_edges_s=tof_edges,
    nQ=200,
    nω=220,
    w_min_clip=0.0,
    w_max_clip=Ei
)

@info "Suggested Q range = $(sug.Q_range) Å^-1"
@info "Suggested ω range = $(sug.ω_range) meV"
@info "Sunny radii length=$(length(sug.radii)) energies length=$(length(sug.energies))"

pts = coverage_points_Qω(inst; pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges)

fig = Figure(size=(900,700))
ax = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)",
          title="Instrument coverage points (pixel×TOF centers)")
scatter!(ax, pts.Qmag, pts.ω; markersize=1)
save("coverage_points_Qw.png", fig)
display(fig)

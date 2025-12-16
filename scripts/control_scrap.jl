using Pkg
cd("/maiqmag/vdp/TOFtwin")
Pkg.activate("/maiqmag/vdp/TOFtwin")
Pkg.add(["Random", "StaticArrays", "GLMakie", "CairoMakie"])
Pkg.add("Revise")
Pkg.add("Interpolations")
Pkg.add("LinearAlgebra")
using Revise

Pkg.precompile()



using TOFtwin
using LinearAlgebra

bank = TOFtwin.bank_from_coverage(
    name="example",
    L2=3.5,
    surface=:cylinder
)

TOFtwin.summarize_pixels(bank.pixels)

# sample 500 random pixels (no replacement)
pix500 = TOFtwin.sample_pixels(bank, TOFtwin.RandomSubset(500, seed=1))

# pick 5 pixels from each η-bin, separately per bank
pix_strat = TOFtwin.sample_pixels(bank, TOFtwin.StratifiedByEta(5, seed=2))

# deterministic angular decimation (every 3rd ψ and every 2nd η)
pix_dec = TOFtwin.sample_pixels(bank, TOFtwin.AngularDecimate(3, 2))

bank = TOFtwin.bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
p = bank.pixels[1]

Ei = 12.0
Ef = 8.0

Q, ω = TOFtwin.Qω_from_pixel(p.r_L, Ei, Ef)
L1 = 20.0                       # example; set your instrument value later
L2 = norm(p.r_L)                # true sample→pixel distance for this pixel
t  = TOFtwin.tof_from_EiEf(L1, L2, Ei, Ef)

using TOFtwin

bank = bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
p = bank.pixels[100]

Ei = 12.0
Ef = 8.0

pose = Pose(Rigid(), sample_orientation_euler(0.0, 0.0, 0.0))
Q_L = Q_L_from_pixel(p.r_L, Ei, Ef)
Q_S = Q_S_from_pixel(p.r_L, Ei, Ef, pose)

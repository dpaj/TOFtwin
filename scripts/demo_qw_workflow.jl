using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin

# geometry
bank = bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
inst = Instrument(name="demo", L1=36.262, pixels=bank.pixels)
pose = Pose()

# toy “experimental” events: sample some pixels and fake a TOF distribution
Ei = 12.0
pix = sample_pixels(bank, RandomSubset(2000, seed=1))

events = Event[]
# choose Ef range to generate events (purely for demo)
Ef_vals = range(1.0, Ei; length=300)

for p in pix, Ef in Ef_vals
    t = tof_from_EiEf(inst.L1, L2(inst, p.id), Ei, Ef)
    push!(events, Event(p.id, t, Ei))
end

# histogram in (|Q|, ω)
Q_edges = collect(range(0.0, 8.0; length=160))
ω_edges = collect(range(-2.0, 12.0; length=180))
H = hist_Qω_powder(events, inst; pose=pose, Q_edges=Q_edges, ω_edges=ω_edges)

@show summarize_pixels(bank.pixels)
@show size(H.counts)

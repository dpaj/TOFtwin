using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin
using Optim
using Random

# ---------------- instrument / bins ----------------
bank = bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
inst = Instrument(name="demo", L1=36.262, pixels=bank.pixels)

pix_used = sample_pixels(bank, AngularDecimate(3, 2))
Ei = 12.0

L2min = minimum(inst.L2)
L2max = maximum(inst.L2)
Ef_min, Ef_max = 1.0, Ei
tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ef_max)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, Ef_min)
ntof = 500
tof_edges = collect(range(tmin, tmax; length=ntof+1))

Q_edges = collect(range(0.0, 8.0; length=220))
ω_edges = collect(range(-2.0, Ei; length=240))

# ---------------- synthetic "data" ----------------
ptrue = (Δ=2.0, v=0.8, σ=0.25)
model_true = ToyModePowder(Δ=ptrue.Δ, v=ptrue.v, σ=ptrue.σ, amp=1.0)

pred_true = predict_powder_mean_Qω(inst;
    pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
    Q_edges=Q_edges, ω_edges=ω_edges, model=model_true
)

I_true = copy(pred_true.Hmean.counts)
W      = pred_true.Hwt.counts

# only trust bins with decent coverage
mask = W .> 20.0

# add noise
Random.seed!(1)
σ_rel = 0.05
σ0 = σ_rel * maximum(I_true[mask])
σ = similar(I_true)
σ .= σ0
I_obs = I_true .+ σ0 .* randn(size(I_true))

# --------- fit helper: best scale+offset analytically ---------
function best_scale_offset(p::AbstractVector, y::AbstractVector, w::AbstractVector)
    # minimize Σ w*(a p + b - y)^2
    Spp = sum(w .* p .* p)
    Sp1 = sum(w .* p)
    S11 = sum(w)
    Spy = sum(w .* p .* y)
    S1y = sum(w .* y)
    det = Spp*S11 - Sp1^2
    a = (Spy*S11 - S1y*Sp1) / det
    b = (Spp*S1y - Sp1*Spy) / det
    return a, b
end

# ---------------- objective ----------------
function obj(x)
    Δ = x[1]
    v = x[2]
    σm = exp(x[3])          # enforce σ > 0

    # keep it sane (soft guardrails)
    if !(0.0 < Δ < Ei && 0.0 < v < 5.0 && 0.01 < σm < 3.0)
        return 1e30
    end

    model = ToyModePowder(Δ=Δ, v=v, σ=σm, amp=1.0)
    pred  = predict_powder_mean_Qω(inst;
        pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
        Q_edges=Q_edges, ω_edges=ω_edges, model=model
    )

    P = pred.Hmean.counts

    # masked vectors
    idx = findall(mask)
    p = P[idx]
    y = I_obs[idx]
    w = (1.0 ./ (σ[idx].^2))

    a, b = best_scale_offset(p, y, w)
    r = a .* p .+ b .- y
    return sum(w .* r .* r)
end

x0 = [1.5, 0.6, log(0.4)]  # initial guess: (Δ, v, logσ)

opts = Optim.Options(iterations=500, show_trace=true)
res  = optimize(obj, x0, NelderMead(), opts)

x̂ = Optim.minimizer(res)

Δ̂ = x̂[1]
v̂ = x̂[2]
σ̂ = exp(x̂[3])

@info "true  Δ=$(ptrue.Δ) v=$(ptrue.v) σ=$(ptrue.σ)"
@info "fit   Δ=$Δ̂ v=$v̂ σ=$σ̂"
@info "status $(Optim.converged(res))  f=$(Optim.minimum(res))"

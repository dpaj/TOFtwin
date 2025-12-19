using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin
using Statistics
using Base.Threads

backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
if backend == "gl"
    using GLMakie
else
    using CairoMakie
end

# ---------------- IDF -> Instrument ----------------
idf_path = joinpath(@__DIR__, "CNCS_Definition_2025B.xml")
out = TOFtwin.load_mantid_idf(idf_path)
@info "IDF loaded" out.meta

inst = out.inst

# bank may or may not be returned by your loader; handle both cases
bank = if hasproperty(out, :bank) && out.bank !== nothing
    out.bank
else
    TOFtwin.DetectorBank(inst.name, out.pixels)   # <-- positional ctor (no keywords)
end

# Decimate heavily at first (CNCS is ~51k pixels)
ψstride, ηstride = 10, 6
pix_used = TOFtwin.sample_pixels(bank, TOFtwin.AngularDecimate(ψstride, ηstride))
@info "pixels used = $(length(pix_used)) (of $(length(bank.pixels)))  stride=(ψ=$ψstride, η=$ηstride)"
@info "Threads = $(nthreads())"

# If DetectorPixel has ΔΩ, print a sanity check
if hasfield(TOFtwin.DetectorPixel, :ΔΩ)
    dΩ = [getfield(p, :ΔΩ) for p in pix_used]
    @info "ΔΩ (sr) range = ($(minimum(dΩ)), $(maximum(dΩ)))  mean=$(mean(dΩ))"
else
    @warn "DetectorPixel has no ΔΩ field; weights will omit solid-angle factor."
end

# ---------------- Axes / TOF ----------------
Ei = 12.0
L2min, L2max = minimum(inst.L2), maximum(inst.L2)
tmin = TOFtwin.tof_from_EiEf(inst.L1, L2min, Ei, Ei)
tmax = TOFtwin.tof_from_EiEf(inst.L1, L2max, Ei, 1.0)
tof_edges = collect(range(tmin, tmax; length=450+1))

Q_edges = collect(range(0.0, 6.0; length=260))
ω_edges = collect(range(-2.0, Ei; length=260))
Q_cent = 0.5 .* (Q_edges[1:end-1] .+ Q_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

# ---------------- Model S(|Q|, ω) ----------------
model = (Q, ω) -> exp(-0.5*((ω - 3.0)/0.35)^2)  # toy: peaked at 3 meV

# ---------------- Powder prediction (single-crystal-style weighted mean) ----------------
function predict_powder_mean_Qω_fast(inst::TOFtwin.Instrument;
    pixels::AbstractVector{TOFtwin.DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s,
    Q_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    model,
    eps::Float64 = 1e-12,
    threaded::Bool = true)

    kin = TOFtwin.precompute_pixel_tof_kinematics(inst, pixels, Ei_meV, tof_edges_s)

    QLx, QLy, QLz = kin.QLx, kin.QLy, kin.QLz
    ωL, jacdt     = kin.ωL, kin.jacdt
    itmin, itmax  = kin.itmin, kin.itmax

    np = length(pixels)

    # ΔΩ per pixel (or 1.0 if not present)
    dΩ = if hasfield(TOFtwin.DetectorPixel, :ΔΩ)
        [getfield(p, :ΔΩ) for p in pixels]
    else
        ones(Float64, np)
    end

    Hsum_thr = [TOFtwin.Hist2D(Q_edges, ω_edges) for _ in 1:nthreads()]
    Hwt_thr  = [TOFtwin.Hist2D(Q_edges, ω_edges) for _ in 1:nthreads()]

    function do_one_pixel!(ip::Int)
        tid  = threadid()
        Hsum = Hsum_thr[tid]
        Hwt  = Hwt_thr[tid]

        hi = itmax[ip]
        hi == 0 && return
        lo = itmin[ip]
        dΩp = dΩ[ip]

        @inbounds for it in lo:hi
            w = jacdt[ip,it] * dΩp            # <- Jacobian * dt * solid angle
            w == 0.0 && continue

            qx = QLx[ip,it]; qy = QLy[ip,it]; qz = QLz[ip,it]
            Qmag = sqrt(qx*qx + qy*qy + qz*qz)
            ω = ωL[ip,it]

            TOFtwin.deposit_bilinear!(Hsum, Qmag, ω, model(Qmag, ω) * w)
            TOFtwin.deposit_bilinear!(Hwt,  Qmag, ω, w)
        end
    end

    if threaded && nthreads() > 1
        @threads for ip in 1:np
            do_one_pixel!(ip)
        end
    else
        for ip in 1:np
            do_one_pixel!(ip)
        end
    end

    Hsum = TOFtwin.Hist2D(Q_edges, ω_edges)
    Hwt  = TOFtwin.Hist2D(Q_edges, ω_edges)
    for k in 1:nthreads()
        Hsum.counts .+= Hsum_thr[k].counts
        Hwt.counts  .+= Hwt_thr[k].counts
    end

    Hmean = TOFtwin.Hist2D(Q_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)

    return (Hsum=Hsum, Hwt=Hwt, Hmean=Hmean)
end

@info "Predicting powder (weighted mean, no detector×TOF matrix)..."
pred = predict_powder_mean_Qω_fast(inst;
    pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
    Q_edges=Q_edges, ω_edges=ω_edges, model=model,
    threaded=true
)

# ---------------- Plot ----------------
fig = Figure(size=(1100, 420))

ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)",
           title="CNCS powder (weighted mean per bin)")
heatmap!(ax1, Q_cent, ω_cent, pred.Hmean.counts)

ax2 = Axis(fig[1,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)",
           title="Weights (Jacobian×ΔΩ)  log10(w+1)")
heatmap!(ax2, Q_cent, ω_cent, log10.(pred.Hwt.counts .+ 1.0))

mkpath("out")
save("out/demo_cncs_powder_from_idf.png", fig)
display(fig)

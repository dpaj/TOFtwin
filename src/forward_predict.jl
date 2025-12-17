using LinearAlgebra

"""
Predict a powder-style histogram in (|Q|, ω) by sweeping pixels and TOF bins.

- model(Qmag, ω) returns relative intensity (arbitrary units)
- includes Jacobian dω/dt and pixel solid angle ΔΩ
- still ignores resolution, efficiency, multiple scattering, etc.

You can pass either:
- pixels = bank.pixels (all), OR
- pixels = sample_pixels(bank, ...) (subsample)
"""
function predict_hist_Qω_powder(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    Q_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    model)

    H = Hist2D(Q_edges, ω_edges)

    for p in pixels
        L2p = L2(inst, p.id)

        for it in 1:length(tof_edges_s)-1
            t0 = tof_edges_s[it]
            t1 = tof_edges_s[it+1]
            t  = 0.5*(t0 + t1)
            dt = (t1 - t0)

            Ef = try
                Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Q, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            Qmag = norm(Q)

            # expected counts per TOF bin ~ S(Q,ω) * dω * ΔΩ = S * |dω/dt| dt * ΔΩ
            jacdt = abs(dω_dt(inst.L1, L2p, Ei_meV, t)) * dt
            w = model(Qmag, ω) * jacdt * p.ΔΩ

            deposit_bilinear!(H, Qmag, ω, w)
        end
    end

    return H
end

"Predict expected counts in detector space: counts[pixel_id, itof]."
function predict_pixel_tof(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    model)

    n_pix = length(inst.pixels)
    n_tof = length(tof_edges_s) - 1
    C = zeros(Float64, n_pix, n_tof)

    for p in pixels
        L2p = L2(inst, p.id)
        for it in 1:n_tof
            t0 = tof_edges_s[it]
            t1 = tof_edges_s[it+1]
            t  = 0.5*(t0 + t1)
            dt = t1 - t0

            Ef = try
                Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Q, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            Qmag = norm(Q)

            # expected counts per TOF bin ~ S(Q,ω) * dω * ΔΩ = S * |dω/dt| dt * ΔΩ
            jacdt = abs(dω_dt(inst.L1, L2p, Ei_meV, t)) * dt
            C[p.id, it] += model(Qmag, ω) * jacdt * p.ΔΩ
        end
    end
    return C
end

"Vanadium-like normalization in detector space."
function normalize_by_vanadium(C_sample::AbstractMatrix, C_van::AbstractMatrix; eps=1e-12)
    return C_sample ./ (C_van .+ eps)
end

"Reduce detector×TOF matrix into powder (|Q|,ω) histogram."
function reduce_pixel_tof_to_Qω_powder(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    C::AbstractMatrix,
    Q_edges::Vector{Float64},
    ω_edges::Vector{Float64})

    H = Hist2D(Q_edges, ω_edges)
    n_tof = length(tof_edges_s) - 1

    for p in pixels
        L2p = L2(inst, p.id)
        for it in 1:n_tof
            w = C[p.id, it]
            w == 0.0 && continue

            t0 = tof_edges_s[it]
            t1 = tof_edges_s[it+1]
            t  = 0.5*(t0 + t1)

            Ef = try
                Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Q, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            deposit_bilinear!(H, norm(Q), ω, w)
        end
    end

    return H
end

"""
Conventional powder workflow:

model(Qmag, ω)  -> detector×TOF -> vanadium normalize -> reduce to (|Q|,ω)
and return the *mean per (Q,ω) bin* plus the per-bin weights (coverage).

Returns:
  (Hraw, Hsum, Hwt, Hmean)
"""
function predict_powder_mean_Qω(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    Q_edges::Vector{Float64},
    ω_edges::Vector{Float64},
    model,
    eps::Float64 = 1e-12)

    # detector×TOF prediction for sample + "vanadium"
    Cs = predict_pixel_tof(inst; pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s, model=model)
    Cv = predict_pixel_tof(inst; pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s, model=(Q,ω)->1.0)

    # raw reduced sum (debug)
    Hraw = reduce_pixel_tof_to_Qω_powder(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=Cs, Q_edges=Q_edges, ω_edges=ω_edges
    )

    # vanadium-normalized in detector×TOF
    Cnorm = normalize_by_vanadium(Cs, Cv; eps=eps)

    # reduce the normalized values: this is a SUM over contributions
    Hsum = reduce_pixel_tof_to_Qω_powder(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=Cnorm, Q_edges=Q_edges, ω_edges=ω_edges
    )

    # per-bin weights/coverage (count of contributing detector×TOF samples)
    W = zeros(size(Cnorm))
    W[Cv .> 0.0] .= 1.0

    Hwt = reduce_pixel_tof_to_Qω_powder(inst;
        pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
        C=W, Q_edges=Q_edges, ω_edges=ω_edges
    )

    Hmean = Hist2D(Q_edges, ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)

    return (Hraw=Hraw, Hsum=Hsum, Hwt=Hwt, Hmean=Hmean)
end

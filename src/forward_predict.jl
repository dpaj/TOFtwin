using LinearAlgebra

"""
Predict a powder-style histogram in (|Q|, ω) by sweeping pixels and TOF bins.

- model(Qmag, ω) returns relative intensity (arbitrary units)
- ignores resolution, efficiency, solid-angle corrections, Jacobians (for now)

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
    tof_centers = bin_center_edges(tof_edges_s)

    for p in pixels
        L2p = L2(inst, p.id)
        for t in tof_centers
            Ef = try
                Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Q, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            Qmag = norm(Q)

            ix = bin_index(Q_edges, Qmag)
            iy = bin_index(ω_edges, ω)
            (ix == 0 || iy == 0) && continue

            H.counts[ix, iy] += model(Qmag, ω)
        end
    end

    return H
end

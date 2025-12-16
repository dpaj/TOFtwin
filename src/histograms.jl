using LinearAlgebra

# ---------- small helpers ----------
bin_center_edges(edges::Vector{Float64}) = 0.5 .* (edges[1:end-1] .+ edges[2:end])

function bin_index(edges::Vector{Float64}, x::Float64)
    # returns i s.t. edges[i] <= x < edges[i+1], or 0 if out of range
    n = length(edges) - 1
    (x < edges[1] || x >= edges[end]) && return 0
    # binary search
    i = searchsortedlast(edges, x)
    return (1 <= i <= n) ? i : 0
end

struct Hist2D
    xedges::Vector{Float64}
    yedges::Vector{Float64}
    counts::Matrix{Float64}
end

Hist2D(xedges::Vector{Float64}, yedges::Vector{Float64}) =
    Hist2D(xedges, yedges, zeros(length(xedges)-1, length(yedges)-1))

"""
Histogram events into powder-style (|Q|, ω).

Returns Hist2D with:
- x = |Q| (Å^-1)
- y = ω (meV)
"""
function hist_Qω_powder(events::AbstractVector{Event}, inst::Instrument;
                        pose::Pose=Pose(), frame::Symbol=:sample,
                        Q_edges::Vector{Float64}, ω_edges::Vector{Float64})
    H = Hist2D(Q_edges, ω_edges)

    for ev in events
        m = Qω_from_event(ev, inst; pose=pose, frame=frame)
        m === nothing && continue
        Qmag = norm(m.Q)
        ix = bin_index(Q_edges, Qmag)
        iy = bin_index(ω_edges, m.ω)
        (ix == 0 || iy == 0) && continue
        H.counts[ix, iy] += ev.weight
    end

    return H
end

"""
Histogram events in detector space: (pixel_id, TOF).

Returns a matrix counts[pixel_id, itof].
"""
function hist_pixel_tof(events::AbstractVector{Event}, inst::Instrument;
                        tof_edges::Vector{Float64})
    n_pix = length(inst.pixels)
    n_tof = length(tof_edges) - 1
    C = zeros(Float64, n_pix, n_tof)

    for ev in events
        (1 <= ev.pixel_id <= n_pix) || continue
        it = bin_index(tof_edges, ev.tof_s)
        it == 0 && continue
        C[ev.pixel_id, it] += ev.weight
    end

    return C
end

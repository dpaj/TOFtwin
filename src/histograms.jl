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

function deposit_bilinear!(H::Hist2D, x::Float64, y::Float64, w::Float64)
    nx = length(H.xedges) - 1
    ny = length(H.yedges) - 1

    ix = searchsortedlast(H.xedges, x)
    iy = searchsortedlast(H.yedges, y)
    (1 <= ix <= nx && 1 <= iy <= ny) || return

    x0, x1 = H.xedges[ix], H.xedges[ix+1]
    y0, y1 = H.yedges[iy], H.yedges[iy+1]
    fx = (x - x0) / (x1 - x0)
    fy = (y - y0) / (y1 - y0)

    # always deposit into the current bin
    H.counts[ix, iy] += w * (1-fx) * (1-fy)

    # deposit to neighbors if they exist
    if ix < nx
        H.counts[ix+1, iy] += w * fx * (1-fy)
    end
    if iy < ny
        H.counts[ix, iy+1] += w * (1-fx) * fy
    end
    if ix < nx && iy < ny
        H.counts[ix+1, iy+1] += w * fx * fy
    end
end

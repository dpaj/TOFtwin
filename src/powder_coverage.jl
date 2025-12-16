using LinearAlgebra

# ----------------------------
# Small utilities (no deps)
# ----------------------------

"Return quantile using sorted indexing (no Statistics.jl). p in [0,1]."
function _quantile(v::AbstractVector{<:Real}, p::Real)
    n = length(v)
    n == 0 && throw(ArgumentError("quantile of empty vector"))
    ps = clamp(float(p), 0.0, 1.0)
    w = sort!(collect(float.(v)))
    # nearest-rank-ish (good enough for range suggestion)
    i = Int(clamp(round(ps*(n-1) + 1), 1, n))
    return w[i]
end

"Pad [lo,hi] by frac of span. If span==0, pad by abs_pad."
function _pad_range(lo::Float64, hi::Float64; frac::Float64=0.05, abs_pad::Float64=1e-6)
    lo2, hi2 = min(lo,hi), max(lo,hi)
    span = hi2 - lo2
    pad = span > 0 ? frac*span : abs_pad
    return (lo2 - pad, hi2 + pad)
end

# ----------------------------
# Coverage point cloud
# ----------------------------

"""
Compute reachable points in (|Q|, ω) for an instrument configuration.

Returns NamedTuple with vectors:
  Qmag::Vector{Float64}   (Å^-1)
  ω::Vector{Float64}      (meV)

Notes:
- Uses TOF bin centers.
- Skips kinematically invalid points (Ef<=0 or Ef>Ei, etc.).
"""
function coverage_points_Qω(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64})

    n_tof = length(tof_edges_s) - 1
    Qm = Float64[]
    ww = Float64[]

    for p in pixels
        L2p = L2(inst, p.id)
        for it in 1:n_tof
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
            push!(Qm, norm(Q))
            push!(ww, ω)
        end
    end

    return (Qmag=Qm, ω=ww)
end

"""
Suggest Sunny powder axes (radii, energies) based on instrument coverage.

Uses quantiles to avoid outliers:
  q_quant=(0.01, 0.99), w_quant=(0.01, 0.99)

Pads by a fraction of the span (q_pad_frac, w_pad_frac).

Optionally clamps ω to [w_min_clip, w_max_clip] (useful since Sunny powder
often uses ω>=0 and ω<=Ei for first pass).

Returns NamedTuple with:
  radii, energies, Q_range, ω_range
"""
function suggest_sunny_powder_axes(inst::Instrument;
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    nQ::Union{Int,Nothing}=nothing,
    nω::Union{Int,Nothing}=nothing,
    dq_target::Float64=0.01,
    dω_target::Float64=0.05,
    q_quant::Tuple{Float64,Float64}=(0.01, 0.99),
    w_quant::Tuple{Float64,Float64}=(0.01, 0.99),
    q_pad_frac::Float64=0.05,
    w_pad_frac::Float64=0.05,
    w_min_clip::Union{Nothing,Float64}=0.0,
    w_max_clip::Union{Nothing,Float64}=nothing)

    pts = coverage_points_Qω(inst; pixels=pixels, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s)
    Qm, ww = pts.Qmag, pts.ω
    isempty(Qm) && throw(ArgumentError("No valid coverage points."))

    qlo = _quantile(Qm, q_quant[1]); qhi = _quantile(Qm, q_quant[2])
    wlo = _quantile(ww, w_quant[1]); whi = _quantile(ww, w_quant[2])

    (qlo, qhi) = _pad_range(qlo, qhi; frac=q_pad_frac, abs_pad=1e-3)
    (wlo, whi) = _pad_range(wlo, whi; frac=w_pad_frac, abs_pad=1e-3)

    if w_min_clip !== nothing; wlo = max(wlo, w_min_clip); end
    if w_max_clip !== nothing; whi = min(whi, w_max_clip); end

    if nQ === nothing
        nQ = Int(clamp(ceil((qhi - qlo)/dq_target) + 1, 50, 4000))
    end
    if nω === nothing
        nω = Int(clamp(ceil((whi - wlo)/dω_target) + 1, 50, 4000))
    end

    radii    = collect(range(qlo, qhi; length=nQ))
    energies = collect(range(wlo, whi; length=nω))

    return (radii=radii, energies=energies, Q_range=(qlo,qhi), ω_range=(wlo,whi))
end


# ----------------------------
# Numerical powder kernel + interpolation
# ----------------------------

"""
Numerical powder kernel S(|Q|, ω) on a rectangular grid.

- Q: vector of |Q| radii (Å^-1), ascending
- ω: vector of energies (meV), ascending
- S: matrix size (length(Q), length(ω))

Call k(Qmag, ω) to evaluate with bilinear interpolation.
Outside the grid returns outside (default 0.0).
"""
struct GridKernelPowder
    Q::Vector{Float64}
    ω::Vector{Float64}
    S::Matrix{Float64}
    outside::Float64
end

function GridKernelPowder(Q::Vector{Float64}, ω::Vector{Float64}, S::AbstractMatrix{<:Real};
    outside::Float64=0.0)

    length(Q) == size(S,1) || throw(ArgumentError("S size mismatch: size(S,1) must equal length(Q)"))
    length(ω) == size(S,2) || throw(ArgumentError("S size mismatch: size(S,2) must equal length(ω)"))

    issorted(Q) || throw(ArgumentError("Q must be sorted ascending"))
    issorted(ω) || throw(ArgumentError("ω must be sorted ascending"))

    return GridKernelPowder(Q, ω, Matrix{Float64}(S), outside)
end

"Find i such that grid[i] <= x <= grid[i+1]. Return 0 if out of bounds."
function _cell_index(grid::Vector{Float64}, x::Float64)
    n = length(grid)
    (x < grid[1] || x > grid[end]) && return 0
    i = searchsortedlast(grid, x)
    i == n && (i = n-1)  # if exactly at upper edge, clamp into last cell
    return (1 <= i < n) ? i : 0
end

function (k::GridKernelPowder)(Qmag::Float64, ω::Float64)
    iQ = _cell_index(k.Q, Qmag)
    iW = _cell_index(k.ω, ω)
    (iQ == 0 || iW == 0) && return k.outside

    Q0, Q1 = k.Q[iQ],   k.Q[iQ+1]
    W0, W1 = k.ω[iW],   k.ω[iW+1]
    fQ = (Qmag - Q0) / (Q1 - Q0)
    fW = (ω    - W0) / (W1 - W0)

    S00 = k.S[iQ,   iW]
    S10 = k.S[iQ+1, iW]
    S01 = k.S[iQ,   iW+1]
    S11 = k.S[iQ+1, iW+1]

    return (1-fQ)*(1-fW)*S00 + fQ*(1-fW)*S10 + (1-fQ)*fW*S01 + fQ*fW*S11
end

"Convenience: build a GridKernelPowder by sampling a function f(Q,ω) on (Q,ω) grids."
function kernel_from_function(Q::Vector{Float64}, ω::Vector{Float64}, f; outside::Float64=0.0)
    S = [f(q,w) for q in Q, w in ω]  # size (nQ, nω)
    return GridKernelPowder(Q, ω, S; outside=outside)
end

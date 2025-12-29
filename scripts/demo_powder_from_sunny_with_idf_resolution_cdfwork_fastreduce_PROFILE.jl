
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin
using JLD2
using LinearAlgebra
using Serialization
using SHA

# -----------------------------------------------------------------------------
# Demo (PROFILE): Same as demo_powder_from_sunny_with_idf_resolution_cdfwork_fastreduce,
# but instruments each major step (time + allocations) to identify bottlenecks.
#
# Extra env vars:
#   TOFTWIN_PROFILE=1                 (default 1) print per-step timings + summary
#   TOFTWIN_PROFILE_SORT=time|bytes   (default time)
# -----------------------------------------------------------------------------

# Backend selection
backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
if backend == "gl"
    using GLMakie
else
    using CairoMakie
end

# -----------------------------
# Profiling helpers (stdlib-only)
# -----------------------------
const DO_PROFILE = lowercase(get(ENV, "TOFTWIN_PROFILE", "1")) in ("1","true","yes")
const PROFILE_SORT = lowercase(get(ENV, "TOFTWIN_PROFILE_SORT", "time"))  # time | bytes

pretty_bytes(b::Real) = b < 1024 ? "$(round(b,digits=1)) B" :
                     b < 1024^2 ? "$(round(b/1024,digits=2)) KiB" :
                     b < 1024^3 ? "$(round(b/1024^2,digits=2)) MiB" :
                                  "$(round(b/1024^3,digits=2)) GiB"

const _PROFILE_ROWS = NamedTuple[]

function step(name::AbstractString, f::Function)
    if !DO_PROFILE
        return f()
    end
    t = @timed f()
    push!(_PROFILE_ROWS, (
        name=String(name),
        time_s=Float64(t.time),
        gctime_s=Float64(t.gctime),
        bytes=Int64(t.bytes),
        allocs=Int64(0),  # placeholder; Julia versions differ on allocation-count reporting
    ))
    @info "STEP" name=name time_s=round(t.time, digits=3) gctime_s=round(t.gctime, digits=3) bytes=pretty_bytes(t.bytes)
    return t.value
end

function print_profile_summary()
    DO_PROFILE || return
    rows = copy(_PROFILE_ROWS)
    if PROFILE_SORT == "bytes"
        sort!(rows, by = r -> -r.bytes)
    else
        sort!(rows, by = r -> -r.time_s)
    end
    println()
    println("==== TOFtwin demo timing summary (sorted by $(PROFILE_SORT)) ====")
    println(rpad("step", 44), rpad("time (s)", 12), rpad("GC (s)", 12), rpad("alloc", 12))
    println("-"^82)
    for r in rows
        println(rpad(r.name[1:min(end,44)], 44),
                rpad(string(round(r.time_s,digits=3)), 12),
                rpad(string(round(r.gctime_s,digits=3)), 12),
                rpad(pretty_bytes(r.bytes), 12))
    end
    tot_t = sum(r.time_s for r in rows)
    tot_gc = sum(r.gctime_s for r in rows)
    tot_b = sum(r.bytes for r in rows)
    println("-"^82)
    println(rpad("TOTAL", 44),
            rpad(string(round(tot_t,digits=3)), 12),
            rpad(string(round(tot_gc,digits=3)), 12),
            rpad(pretty_bytes(tot_b), 12))
    println()
end

atexit(print_profile_summary)

# -----------------------------
# Output knobs
# -----------------------------
do_hist      = lowercase(get(ENV, "TOFTWIN_DO_HIST", "1")) in ("1","true","yes")
plot_kernel  = lowercase(get(ENV, "TOFTWIN_PLOT_KERNEL", "1")) in ("1","true","yes")
do_save      = lowercase(get(ENV, "TOFTWIN_SAVE", "1")) in ("1","true","yes")
outdir       = get(ENV, "TOFTWIN_OUTDIR", joinpath(@__DIR__, "out"))
do_save && mkpath(outdir)

# -----------------------------
# Disk caching
# -----------------------------
disk_cache = lowercase(get(ENV, "TOFTWIN_DISK_CACHE", "1")) in ("1","true","yes")
cache_dir  = get(ENV, "TOFTWIN_CACHE_DIR", joinpath(@__DIR__, ".toftwin_cache"))
cache_ver  = get(ENV, "TOFTWIN_CACHE_VERSION", "v1")
cache_Cv   = lowercase(get(ENV, "TOFTWIN_CACHE_CV", disk_cache ? "1" : "0")) in ("1","true","yes")
disk_cache && mkpath(cache_dir)

# -----------------------------
# Helper: load Sunny powder table as a TOFtwin kernel
# -----------------------------
function load_sunny_powder_jld2(path::AbstractString; outside=0.0)
    radii = energies = S_Qω = nothing
    @load path radii energies S_Qω

    Q = collect(radii)
    W = collect(energies)
    S = Matrix(S_Qω)

    @info "Loaded Sunny table: size(S)=$(size(S)), len(Q)=$(length(Q)), len(ω)=$(length(W))"

    # Sunny often stores as (ω, Q). Fix to (Q, ω).
    if size(S) == (length(W), length(Q))
        @info "Transposing Sunny table (ω,Q) -> (Q,ω)"
        S = permutedims(S)
    end
    @assert size(S) == (length(Q), length(W))

    return TOFtwin.GridKernelPowder(Q, W, S; outside=outside)
end

# -----------------------------
# Helper: suggest a good NTOF for CDF mode
# -----------------------------
function suggest_ntof(inst, pixels, Ei_meV, tmin, tmax, ω_edges, σt_s;
        α=0.5, β=1/3, ntof_min=200, ntof_max=2000)

    Δω_bin = (ω_edges[end] - ω_edges[1]) / (length(ω_edges)-1)
    ω_samples = unique(filter(ω -> (0.0 <= ω < Ei_meV),
        [0.0, 0.5, 1.0, 2.0, 5.0, 0.25Ei_meV, 0.5Ei_meV, min(10.0, Ei_meV-0.5)]))

    reps = pixels[[1, cld(length(pixels),2), length(pixels)]]
    max_dωdt = 0.0
    for p in reps
        L2p = L2(inst, p.id)
        for ω in ω_samples
            Ef = Ei_meV - ω
            Ef <= 0 && continue
            t = tof_from_EiEf(inst.L1, L2p, Ei_meV, Ef)
            max_dωdt = max(max_dωdt, abs(dω_dt(inst.L1, L2p, Ei_meV, t)))
        end
    end
    max_dωdt == 0.0 && return ntof_min, (tmax-tmin)/ntof_min, max_dωdt, Δω_bin

    dt_from_kernel = β * σt_s
    dt_from_omega  = α * Δω_bin / max_dωdt
    dt = min(dt_from_kernel, dt_from_omega)

    ntof = Int(ceil((tmax - tmin) / dt))
    ntof = clamp(ntof, ntof_min, ntof_max)
    dt = (tmax - tmin) / ntof
    return ntof, dt, max_dωdt, Δω_bin
end

# -----------------------------
# Instrument selection
# -----------------------------
function parse_instrument()
    s = uppercase(get(ENV, "TOFTWIN_INSTRUMENT", "CNCS"))
    if s in ("CNCS",)
        return :CNCS
    elseif s in ("SEQUOIA", "SEQ")
        return :SEQUOIA
    else
        throw(ArgumentError("TOFTWIN_INSTRUMENT must be CNCS or SEQUOIA (got '$s')"))
    end
end

instr = parse_instrument()
idf_path = instr === :CNCS ? joinpath(@__DIR__, "CNCS_Definition_2025B.xml") :
                            joinpath(@__DIR__, "SEQUOIA_Definition.xml")

# -----------------------------
# Sunny input
# -----------------------------
sunny_path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "sunny_powder_corh2o4.jld2")
@info "Sunny input file: $sunny_path"

kern = step("load Sunny kernel", () -> load_sunny_powder_jld2(sunny_path; outside=0.0))
model = (q, w) -> kern(q, w)

# -----------------------------
# IDF -> Instrument
# -----------------------------
rebuild_geom = lowercase(get(ENV, "TOFTWIN_REBUILD_GEOM", "0")) in ("1","true","yes")
out = step("load instrument from IDF", () -> begin
    if isdefined(TOFtwin.MantidIDF, :load_mantid_idf_diskcached)
        TOFtwin.MantidIDF.load_mantid_idf_diskcached(idf_path; rebuild=rebuild_geom)
    else
        @warn "load_mantid_idf_diskcached not found; falling back to load_mantid_idf (will be slower)."
        TOFtwin.MantidIDF.load_mantid_idf(idf_path)
    end
end)

inst = out.inst
bank = (hasproperty(out, :bank) && out.bank !== nothing) ? out.bank : TOFtwin.DetectorBank(inst.name, out.pixels)

default_stride = instr === :SEQUOIA ? "1" : "1"
ψstride = parse(Int, get(ENV, "TOFTWIN_PSI_STRIDE", default_stride))
ηstride = parse(Int, get(ENV, "TOFTWIN_ETA_STRIDE", default_stride))

pix_used = step("select pixels (AngularDecimate)", () -> sample_pixels(bank, AngularDecimate(ψstride, ηstride)))

@info "Instrument = $instr"
@info "IDF: $idf_path"
@info "pixels used = $(length(pix_used)) (of $(length(bank.pixels)))  stride=(ψ=$ψstride, η=$ηstride)"

# -----------------------------
# Axes / TOF
# -----------------------------
default_Ei = instr === :SEQUOIA ? "30.0" : "12.0"
Ei = parse(Float64, get(ENV, "TOFTWIN_EI", default_Ei))  # meV

nQbins = parse(Int, get(ENV, "TOFTWIN_NQBINS", "220"))
nωbins = parse(Int, get(ENV, "TOFTWIN_NWBINS", "240"))

Q_edges, ω_edges = step("build Q,ω axes", () -> begin
    Qe = collect(range(kern.Q[1], kern.Q[end]; length=nQbins+1))
    ωlo = min(-2.0, kern.ω[1])
    ωhi = max(Ei,  kern.ω[end])
    ωe = collect(range(ωlo, ωhi; length=nωbins+1))
    return (Qe, ωe)
end)

# TOF window (rough)
L2min = minimum(inst.L2)
L2max = maximum(inst.L2)
Ef_min, Ef_max = 1.0, Ei
tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ef_max)
tmax = tof_from_EiEf(inst.L1, L2max, Ei, Ef_min)

res_mode = lowercase(get(ENV, "TOFTWIN_RES_MODE", "cdf"))  # none | gh | cdf
σt_us    = parse(Float64, get(ENV, "TOFTWIN_SIGMA_T_US", "100.0"))
σt       = σt_us * 1e-6

# NTOF selection
ntof_env = lowercase(get(ENV, "TOFTWIN_NTOF", "auto"))
α = parse(Float64, get(ENV, "TOFTWIN_NTOF_ALPHA", "2.0"))
β = parse(Float64, get(ENV, "TOFTWIN_NTOF_BETA",  "1.0"))
ntof_min = parse(Int, get(ENV, "TOFTWIN_NTOF_MIN", "200"))
ntof_max = parse(Int, get(ENV, "TOFTWIN_NTOF_MAX", "1000"))
ntof = (ntof_env in ("auto","suggest","0")) ? 0 : parse(Int, ntof_env)

gh_order = parse(Int, get(ENV, "TOFTWIN_GH_ORDER", "3"))
nsigma   = parse(Float64, get(ENV, "TOFTWIN_NSIGMA", "6.0"))

ntof = step("choose NTOF", () -> begin
    if ntof <= 0 && res_mode == "cdf" && σt_us > 0
        ntof_s, dt_s, maxd, dωbin = suggest_ntof(inst, pix_used, Ei, tmin, tmax, ω_edges, σt;
            α=α, β=β, ntof_min=ntof_min, ntof_max=ntof_max)
        @info "Auto-selected TOFTWIN_NTOF=$ntof_s  (dt=$(round(1e6*dt_s,digits=2)) µs; max|dω/dt|=$(round(maxd,digits=4)) meV/s; Δωbin=$(round(dωbin,digits=4)) meV)"
        return ntof_s
    elseif ntof <= 0
        @info "TOFTWIN_NTOF=auto but res_mode=$res_mode or σt<=0; using default NTOF=500"
        return 500
    else
        return ntof
    end
end)

tof_edges = step("build TOF edges", () -> collect(range(tmin, tmax; length=ntof+1)))

@info "TOF window (ms) = $(1e3*tmin) .. $(1e3*tmax)"
@info "Sunny kernel domain: Q∈($(kern.Q[1]), $(kern.Q[end]))  ω∈($(kern.ω[1]), $(kern.ω[end]))"

resolution = step("construct resolution model", () -> begin
    (res_mode == "none" || σt_us <= 0) ? NoResolution() :
    (res_mode == "gh")  ? GaussianTimingResolution(σt; order=gh_order) :
    (res_mode == "cdf") ? GaussianTimingCDFResolution(σt; nsigma=nsigma) :
    throw(ArgumentError("TOFTWIN_RES_MODE must be none|gh|cdf (got '$res_mode')"))
end)

# -----------------------------
# Disk cache helper
# -----------------------------
function _cache_path(prefix::AbstractString, bytes::Vector{UInt8})
    return joinpath(cache_dir, prefix * "_" * bytes2hex(sha1(bytes)) * ".jls")
end

function _load_or_compute(path::AbstractString, builder::Function)
    if disk_cache && isfile(path)
        open(path, "r") do io
            return deserialize(io)
        end
    end
    val = builder()
    if disk_cache
        open(path, "w") do io
            serialize(io, val)
        end
    end
    return val
end

# -----------------------------
# Precompute (optional) CDF smear work
# -----------------------------
cdf_work = step("precompute CDF smear work (optional)", () -> begin
    if !(resolution isa GaussianTimingCDFResolution)
        return nothing
    end
    if isdefined(TOFtwin, :precompute_tof_smear_work_diskcached) && disk_cache
        w = TOFtwin.precompute_tof_smear_work_diskcached(inst;
            pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges, resolution=resolution,
            cache_dir=cache_dir,
            cache_tag=cache_ver * "|instr=$(String(instr))|ψ=$(ψstride)|η=$(ηstride)"
        )
        @info "Using disk-cached CDF smear work"
        return w
    elseif isdefined(TOFtwin, :precompute_tof_smear_work)
        w = TOFtwin.precompute_tof_smear_work(inst, pix_used, Ei, tof_edges, resolution)
        @info "Using in-session precomputed CDF smear work"
        return w
    else
        @warn "TOFtwin.precompute_tof_smear_work not found; continuing without precomputed work"
        return nothing
    end
end)

# -----------------------------
# Cache Cv to disk (model-independent)
# -----------------------------
function compute_Cv_flat()
    return predict_pixel_tof(inst;
        pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
        model=(Q,ω)->1.0, resolution=resolution,
        cdf_work=cdf_work
    )
end

Cv = step("Cv (flat) load/compute", () -> begin
    if cache_Cv && disk_cache
        io = IOBuffer()
        serialize(io, cache_ver)
        serialize(io, String(instr))
        serialize(io, idf_path)
        serialize(io, rebuild_geom)
        st = try stat(idf_path) catch; nothing end
        serialize(io, st === nothing ? nothing : (st.mtime, st.size))
        serialize(io, Int32[p.id for p in pix_used])
        serialize(io, Ei)
        serialize(io, tof_edges)
        serialize(io, typeof(resolution))
        if resolution isa GaussianTimingCDFResolution
            serialize(io, resolution.nsigma)
            serialize(io, resolution.σt isa Function ? :callable : Float64(resolution.σt))
        elseif resolution isa GaussianTimingResolution
            serialize(io, resolution.order)
            serialize(io, Float64(resolution.σt))
        end
        path = _cache_path("Cv_flat", take!(io))
        @info "Cv cache: $path"
        return _load_or_compute(path, compute_Cv_flat)
    else
        return compute_Cv_flat()
    end
end)

Cs = step("Cs predict_pixel_tof (model)", () -> predict_pixel_tof(inst;
    pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
    model=(q,w)->kern(q,w), resolution=resolution,
    cdf_work=cdf_work
))

if !(lowercase(get(ENV, "TOFTWIN_DO_HIST", "1")) in ("1","true","yes"))
    @info "TOFTWIN_DO_HIST=0: skipping histogramming/plotting."
    exit()
end

# -----------------------------
# Reduce map + one-pass reduction
# -----------------------------
struct ReduceMapPowder
    iQ::Matrix{Int32}
    iW::Matrix{Int32}
    fQ::Matrix{Float32}
    fW::Matrix{Float32}
    valid::BitMatrix
end

function build_reduce_map_powder(inst::Instrument, pixels::Vector{DetectorPixel},
        Ei_meV::Float64, tof_edges_s::Vector{Float64}, Q_edges::Vector{Float64}, ω_edges::Vector{Float64})

    nP   = length(pixels)
    nT   = length(tof_edges_s) - 1
    nQ   = length(Q_edges) - 1
    nW   = length(ω_edges) - 1

    iQ = fill(Int32(0), nP, nT)
    iW = fill(Int32(0), nP, nT)
    fQ = fill(Float32(0), nP, nT)
    fW = fill(Float32(0), nP, nT)
    valid = falses(nP, nT)

    @inbounds for ip in 1:nP
        p = pixels[ip]
        L2p = L2(inst, p.id)
        for it in 1:nT
            t = 0.5 * (tof_edges_s[it] + tof_edges_s[it+1])

            Ef = try
                Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Qvec, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            Qmag = norm(Qvec)

            iq = searchsortedlast(Q_edges, Qmag)
            iw = searchsortedlast(ω_edges, ω)
            if iq < 1 || iq >= nQ || iw < 1 || iw >= nW
                continue
            end

            dq = Q_edges[iq+1] - Q_edges[iq]
            dw = ω_edges[iw+1] - ω_edges[iw]
            (dq <= 0 || dw <= 0) && continue

            fq = Float32((Qmag - Q_edges[iq]) / dq)
            fw = Float32((ω    - ω_edges[iw]) / dw)
            fq = min(max(fq, 0.0f0), 1.0f0)
            fw = min(max(fw, 0.0f0), 1.0f0)

            iQ[ip,it] = Int32(iq)
            iW[ip,it] = Int32(iw)
            fQ[ip,it] = fq
            fW[ip,it] = fw
            valid[ip,it] = true
        end
    end
    return ReduceMapPowder(iQ, iW, fQ, fW, valid)
end

function load_or_build_reduce_map()
    io = IOBuffer()
    serialize(io, cache_ver)
    serialize(io, String(instr))
    serialize(io, idf_path)
    st = try stat(idf_path) catch; nothing end
    serialize(io, st === nothing ? nothing : (st.mtime, st.size))
    serialize(io, Int32[p.id for p in pix_used])
    serialize(io, Ei)
    serialize(io, tof_edges)
    serialize(io, Q_edges)
    serialize(io, ω_edges)
    path = _cache_path("reduce_map_powder", take!(io))
    @info "Reduce-map cache: $path"
    return _load_or_compute(path, () -> build_reduce_map_powder(inst, pix_used, Ei, tof_edges, Q_edges, ω_edges))
end

rmap = step("reduce-map load/build", () -> (disk_cache ? load_or_build_reduce_map() :
                                           build_reduce_map_powder(inst, pix_used, Ei, tof_edges, Q_edges, ω_edges)))

function reduce_three!(Hraw::Hist2D, Hsum::Hist2D, Hwt::Hist2D,
        pixels::Vector{DetectorPixel}, rmap::ReduceMapPowder,
        Cs::AbstractMatrix, Cv::AbstractMatrix; eps=1e-12)

    nP = length(pixels)
    nT = size(rmap.valid, 2)

    @inbounds for ip in 1:nP
        pid = pixels[ip].id
        for it in 1:nT
            rmap.valid[ip,it] || continue
            cv = Cv[pid, it]
            cv <= 0.0 && continue

            cs = Cs[pid, it]
            iq = Int(rmap.iQ[ip,it])
            iw = Int(rmap.iW[ip,it])
            fq = Float64(rmap.fQ[ip,it])
            fw = Float64(rmap.fW[ip,it])

            w00 = (1 - fq) * (1 - fw)
            w10 = fq       * (1 - fw)
            w01 = (1 - fq) * fw
            w11 = fq       * fw

            Hraw.counts[iq,   iw]   += w00 * cs
            Hraw.counts[iq+1, iw]   += w10 * cs
            Hraw.counts[iq,   iw+1] += w01 * cs
            Hraw.counts[iq+1, iw+1] += w11 * cs

            v = cs / (cv + eps)
            Hsum.counts[iq,   iw]   += w00 * v
            Hsum.counts[iq+1, iw]   += w10 * v
            Hsum.counts[iq,   iw+1] += w01 * v
            Hsum.counts[iq+1, iw+1] += w11 * v

            Hwt.counts[iq,   iw]   += w00
            Hwt.counts[iq+1, iw]   += w10
            Hwt.counts[iq,   iw+1] += w01
            Hwt.counts[iq+1, iw+1] += w11
        end
    end
    return nothing
end

Hraw = Hist2D(Q_edges, ω_edges)
Hsum = Hist2D(Q_edges, ω_edges)
Hwt  = Hist2D(Q_edges, ω_edges)

step("reduce (Hraw/Hsum/Hwt one pass)", () -> reduce_three!(Hraw, Hsum, Hwt, pix_used, rmap, Cs, Cv; eps=1e-12))

Hmean = Hist2D(Q_edges, ω_edges)
step("compute Hmean", () -> (Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ 1e-12)))

# Plot
Q_cent = 0.5 .* (Q_edges[1:end-1] .+ Q_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

kernel_grid = step("kernel grid (optional)", () -> (plot_kernel ? [kern(q, w) for q in Q_cent, w in ω_cent] : nothing))
wt_log = step("log10 weights", () -> log10.(Hwt.counts .+ 1.0))

fig = step("build figure", () -> begin
    fig = Figure(size=(1400, 900))
    title_tag = (σt_us <= 0 || res_mode == "none") ? "no resolution" :
                (res_mode == "gh") ? "GH(order=$(gh_order)), σt=$(σt_us) µs" :
                "CDF(nsigma=$(nsigma)), σt=$(σt_us) µs"

    if plot_kernel
        ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="Sunny kernel S(|Q|,ω) on TOFtwin grid")
        heatmap!(ax1, Q_cent, ω_cent, kernel_grid)
    else
        ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="(kernel plot disabled) set TOFTWIN_PLOT_KERNEL=1")
    end

    ax2 = Axis(fig[1,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin predicted (raw SUM) — $(String(instr)) IDF ($title_tag)")
    heatmap!(ax2, Q_cent, ω_cent, Hraw.counts)

    ax3 = Axis(fig[2,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin weights log10(N+1) — $(String(instr)) IDF")
    heatmap!(ax3, Q_cent, ω_cent, wt_log)

    ax4 = Axis(fig[2,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin vanadium-normalized MEAN ($title_tag)")
    heatmap!(ax4, Q_cent, ω_cent, Hmean.counts)

    return fig
end)

step("save PNG (optional)", () -> begin
    if do_save
        outpng = joinpath(outdir, "demo_powder_fastreduce_profile_$(lowercase(String(instr)))_$(res_mode).png")
        save(outpng, fig)
        @info "Wrote $outpng"
    end
    return nothing
end)

display(fig)

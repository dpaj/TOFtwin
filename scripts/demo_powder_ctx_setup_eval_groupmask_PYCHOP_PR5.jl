
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using TOFtwin
using JLD2
using LinearAlgebra

# -----------------------------------------------------------------------------
# Demo: setup_ctx + eval_model (grouping/masking aware) using TOFtwin workflow API
#
# Key idea:
#   - setup_powder_ctx(...) is the expensive, model-independent part
#   - eval_powder(ctx, model) is the per-model part (Cs + reduce)
#
# Typical:
#   TOFTWIN_INSTRUMENT=CNCS TOFTWIN_GROUPING=8x2 TOFTWIN_MASK_BTP="Bank=36-50" \
#   julia --project=. scripts/demo_powder_ctx_setup_eval_groupmask_refac.jl \
#       ../sunny_powder_corh2o4.jld2 ../sunny_powder_other.jld2
# -----------------------------------------------------------------------------

# -----------------------------
# Backend selection
# -----------------------------
backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
if backend == "gl"
    using GLMakie
else
    using CairoMakie
end

# -----------------------------
# Optional: lightweight timing helpers
# -----------------------------
const DO_PROFILE = lowercase(get(ENV, "TOFTWIN_PROFILE", "0")) in ("1","true","yes")

pretty_bytes(b::Real) = b < 1024 ? "$(round(b,digits=1)) B" :
                     b < 1024^2 ? "$(round(b/1024,digits=2)) KiB" :
                     b < 1024^3 ? "$(round(b/1024^2,digits=2)) MiB" :
                                  "$(round(b/1024^3,digits=2)) GiB"

function step(name::AbstractString, f::Function)
    if !DO_PROFILE
        return f()
    end
    t = @timed f()
    @info "STEP" name=name time_s=round(t.time, digits=3) gctime_s=round(t.gctime, digits=3) bytes=pretty_bytes(t.bytes)
    return t.value
end

# -----------------------------
# Output knobs
# -----------------------------
do_hist     = lowercase(get(ENV, "TOFTWIN_DO_HIST", "1")) in ("1","true","yes")
plot_kernel = lowercase(get(ENV, "TOFTWIN_PLOT_KERNEL", "1")) in ("1","true","yes")
do_save     = lowercase(get(ENV, "TOFTWIN_SAVE", "1")) in ("1","true","yes")
outdir      = get(ENV, "TOFTWIN_OUTDIR", joinpath(@__DIR__, "out"))
do_save && mkpath(outdir)

# -----------------------------
# Disk caching (Cv + reduce-map only)
# -----------------------------
disk_cache  = lowercase(get(ENV, "TOFTWIN_DISK_CACHE", "1")) in ("1","true","yes")
cache_dir   = get(ENV, "TOFTWIN_CACHE_DIR", joinpath(@__DIR__, ".toftwin_cache"))
cache_ver   = get(ENV, "TOFTWIN_CACHE_VERSION", "ctx_v2")

# IMPORTANT default: off for these large objects (prefer in-RAM reuse for sweeps)
cache_Cv    = lowercase(get(ENV, "TOFTWIN_CACHE_CV", "0")) in ("1","true","yes")
cache_rmap  = lowercase(get(ENV, "TOFTWIN_CACHE_RMAP", "0")) in ("1","true","yes")

disk_cache && mkpath(cache_dir)

# -----------------------------
# Grouping/masking (Mantid-style analogs)
# -----------------------------
#   TOFTWIN_GROUPING=""|"2x1"|"4x1"|"8x1"|"4x2"|"8x2"|"powder"
#   TOFTWIN_GROUPING_FILE=""   optional explicit xml path override
#   TOFTWIN_MASK_BTP=""        e.g. "Bank=40-50;Mode=drop" or "DetectorList=123,124"
#   TOFTWIN_MASK_MODE=drop|zeroΩ (used if spec omits Mode)
#   TOFTWIN_POWDER_ANGLESTEP=0.5
grouping      = strip(get(ENV, "TOFTWIN_GROUPING", "4x2"))
grouping_file = strip(get(ENV, "TOFTWIN_GROUPING_FILE", ""))
mask_btp      = get(ENV, "TOFTWIN_MASK_BTP", "Bank=36-50")
mask_mode     = Symbol(lowercase(get(ENV, "TOFTWIN_MASK_MODE", "drop")))
angle_step    = parse(Float64, get(ENV, "TOFTWIN_POWDER_ANGLESTEP", "0.5"))

# -----------------------------
# Sunny loader
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

# -----------------------------
# Plot
# -----------------------------
function plot_result(ctx::TOFtwin.PowderCtx, kern::TOFtwin.GridKernelPowder, pred; tag::AbstractString="")
    Q_cent = 0.5 .* (ctx.Q_edges[1:end-1] .+ ctx.Q_edges[2:end])
    ω_cent = 0.5 .* (ctx.ω_edges[1:end-1] .+ ctx.ω_edges[2:end])

    kernel_grid = plot_kernel ? step("kernel grid", () -> [kern(q,w) for q in Q_cent, w in ω_cent]) : nothing
    wt_log = step("log10 weights", () -> log10.(pred.Hwt.counts .+ 1.0))

    fig = step("build figure", () -> begin
        fig = Figure(size=(1400, 900))

        if plot_kernel
            ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="Sunny kernel S(|Q|,ω)")
            heatmap!(ax1, Q_cent, ω_cent, kernel_grid)
        else
            ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="(kernel plot disabled)")
        end

        ax2 = Axis(fig[1,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin predicted (raw SUM)")
        heatmap!(ax2, Q_cent, ω_cent, pred.Hraw.counts)

        ax3 = Axis(fig[2,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="weights log10(N+1)")
        heatmap!(ax3, Q_cent, ω_cent, wt_log)

        ax4 = Axis(fig[2,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="vanadium-normalized MEAN")
        heatmap!(ax4, Q_cent, ω_cent, pred.Hmean.counts)

        return fig
    end)

    if do_save
        fname = isempty(tag) ? "demo_powder_ctx_$(lowercase(String(ctx.instr))).png" :
                               "demo_powder_ctx_$(lowercase(String(ctx.instr)))_$(tag).png"
        outpng = joinpath(outdir, fname)
        step("save PNG", () -> save(outpng, fig))
        @info "Wrote $outpng"
    end
    display(fig)
end

# -----------------------------------------------------------------------------
# Main: build ctx once, then evaluate 1..N Sunny kernels
# -----------------------------------------------------------------------------
instr = parse_instrument()
idf_path = instr === :CNCS ? joinpath(@__DIR__, "CNCS_Definition_2025B.xml") :
                            joinpath(@__DIR__, "SEQUOIA_Definition.xml")

sunny_paths = length(ARGS) > 0 ? ARGS : [joinpath(@__DIR__, "..", "sunny_powder_corh2o4.jld2")]

kern0 = step("load Sunny kernel (axes anchor)", () -> load_sunny_powder_jld2(sunny_paths[1]; outside=0.0))

default_Ei = instr === :SEQUOIA ? "30.0" : "12.0"
Ei = parse(Float64, get(ENV, "TOFTWIN_EI", default_Ei))  # meV

nQbins = parse(Int, get(ENV, "TOFTWIN_NQBINS", "220"))
nωbins = parse(Int, get(ENV, "TOFTWIN_NWBINS", "240"))

ψstride = parse(Int, get(ENV, "TOFTWIN_PSI_STRIDE", "1"))
ηstride = parse(Int, get(ENV, "TOFTWIN_ETA_STRIDE", "1"))

res_mode = lowercase(get(ENV, "TOFTWIN_RES_MODE", "cdf"))
σt_us    = parse(Float64, get(ENV, "TOFTWIN_SIGMA_T_US", "100.0"))
gh_order = parse(Int, get(ENV, "TOFTWIN_GH_ORDER", "3"))
nsigma   = parse(Float64, get(ENV, "TOFTWIN_NSIGMA", "6.0"))

# Timing width source:
#   TOFTWIN_SIGMA_T_SOURCE=manual (default) -> use TOFTWIN_SIGMA_T_US
#   TOFTWIN_SIGMA_T_SOURCE=pychop          -> derive an *effective* σt(t) curve from PyChop ΔE(ω)
σt_source = lowercase(get(ENV, "TOFTWIN_SIGMA_T_SOURCE", "pychop"))

# Optional: sanity-check that the TOFtwin TOF-smearing implies a ΔE(ω) consistent
# with the raw PyChop oracle. (Turn off with TOFTWIN_PYCHOP_CHECK=0.)
pychop_check = lowercase(get(ENV, "TOFTWIN_PYCHOP_CHECK", "1")) in ("1", "true", "yes")

# If non-empty, these Etrans values (meV) are used for the check.
# (Otherwise, TOFtwin picks a small default list and clips to valid range.)
pychop_check_etrans_meV = Float64[]

# PyChop oracle script (used when TOFTWIN_SIGMA_T_SOURCE=pychop)
pychop_python = get(ENV, "TOFTWIN_PYCHOP_PYTHON", get(ENV, "PYTHON", Sys.iswindows() ? raw"C:\\Users\\vdp\\AppData\\Local\\Microsoft\\WindowsApps\\python3.11.exe" : "python3"))
pychop_script = get(ENV, "TOFTWIN_PYCHOP_SCRIPT", joinpath(@__DIR__, "pychop_oracle_dE.py"))
pychop_variant = get(ENV, "TOFTWIN_PYCHOP_VARIANT", "")
pychop_freq_str = get(ENV, "TOFTWIN_PYCHOP_FREQ_HZ", "")
pychop_freq_hz = isempty(strip(pychop_freq_str)) ? Float64[] : parse.(Float64, split(pychop_freq_str, r"[^0-9eE+\-.]+"; keepempty=false))
pychop_tc_index = parse(Int, get(ENV, "TOFTWIN_PYCHOP_TC_INDEX", "0"))
pychop_use_tc_rss = lowercase(get(ENV, "TOFTWIN_PYCHOP_TC_RSS", "0")) in ("1","true","yes")
pychop_delta_td_us = parse(Float64, get(ENV, "TOFTWIN_PYCHOP_DELTA_TD_US", "0.0"))
pychop_npts = parse(Int, get(ENV, "TOFTWIN_PYCHOP_NPTS", "401"))
pychop_sigma_q = parse(Float64, get(ENV, "TOFTWIN_PYCHOP_SIGMA_Q", "0.25"))

# Optional: generate a more detailed resolution comparison plot/table.
#   TOFTWIN_PYCHOP_CHECK_PLOT=1
pychop_check_plot = lowercase(get(ENV, "TOFTWIN_PYCHOP_CHECK_PLOT", "1")) in ("1","true","yes")
pychop_check_plot_npts = parse(Int, get(ENV, "TOFTWIN_PYCHOP_CHECK_PLOT_NPTS", "121"))
pychop_check_plot_npix = parse(Int, get(ENV, "TOFTWIN_PYCHOP_CHECK_PLOT_NPIX", "3"))

ntof_env   = get(ENV, "TOFTWIN_NTOF", "auto")
ntof_alpha = parse(Float64, get(ENV, "TOFTWIN_NTOF_ALPHA", "0.5"))
ntof_beta  = parse(Float64, get(ENV, "TOFTWIN_NTOF_BETA",  "0.333333333333"))
ntof_min   = parse(Int, get(ENV, "TOFTWIN_NTOF_MIN", "200"))
ntof_max   = parse(Int, get(ENV, "TOFTWIN_NTOF_MAX", "2000"))

rebuild_geom = lowercase(get(ENV, "TOFTWIN_REBUILD_GEOM", "0")) in ("1","true","yes")

# -----------------------------
# Resolution comparison plot (PyChop vs TOFtwin)
# -----------------------------

const _C_US = 252.777  # t(μs) = C * L(m) / sqrt(E(meV))

function _tof_us(L1_m::Float64, L2_m::Float64, Ei_meV::Float64, Ef_meV::Float64)
    return _C_US * (L1_m / sqrt(Ei_meV) + L2_m / sqrt(Ef_meV))
end

function _Ef_from_tof_s(L1_m::Float64, L2_m::Float64, Ei_meV::Float64, t_s::Float64)
    t_us = t_s * 1e6
    t1_us = _C_US * L1_m / sqrt(Ei_meV)
    t2_us = t_us - t1_us
    t2_us <= 0 && return NaN
    return (_C_US * L2_m / t2_us)^2
end

function _fwhm_monotonic_x(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    n == 0 && return NaN
    ymax = maximum(y)
    ymax <= 0 && return NaN
    half = 0.5 * ymax
    imax = argmax(y)

    xL = NaN
    for i in imax:-1:2
        y1, y2 = y[i-1], y[i]
        if y1 < half && y2 >= half
            t = (half - y1) / (y2 - y1)
            xL = float(x[i-1]) + t*(float(x[i]) - float(x[i-1]))
            break
        end
    end

    xR = NaN
    for i in imax:1:(n-1)
        y1, y2 = y[i], y[i+1]
        if y1 >= half && y2 < half
            t = (half - y1) / (y2 - y1)
            xR = float(x[i]) + t*(float(x[i+1]) - float(x[i]))
            break
        end
    end

    return (isfinite(xL) && isfinite(xR)) ? (xR - xL) : NaN
end

function _run_pychop_oracle_curve(; python::AbstractString, script::AbstractString,
    instrument::AbstractString, Ei_meV::Float64, variant::AbstractString,
    freq_hz::Vector{Float64}, tc_index::Int, use_tc_rss::Bool,
    delta_td_us::Float64, etrans_min::Float64, etrans_max::Float64, npts::Int)

    freq_str = isempty(freq_hz) ? "" : join(string.(freq_hz), ",")
    cmd = Cmd([
        python,
        script,
        "--instrument", instrument,
        "--Ei", string(Ei_meV),
        "--variant", variant,
        "--freq", freq_str,
        "--tc-index", string(tc_index),
        "--delta-td-us", string(delta_td_us),
        "--etrans-min", string(etrans_min),
        "--etrans-max", string(etrans_max),
        "--npts", string(npts),
    ]; ignorestatus=false)
    if use_tc_rss
        cmd = Cmd(vcat(collect(cmd.exec), ["--use-tc-rss"]); ignorestatus=false)
    end

    txt = read(cmd, String)
    xs = Float64[]
    ys = Float64[]
    for line in eachline(IOBuffer(txt))
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        f = split(s)
        length(f) < 2 && continue
        push!(xs, parse(Float64, f[1]))
        push!(ys, parse(Float64, f[2]))
    end
    return xs, ys
end

function plot_resolution_compare(ctx::TOFtwin.PowderCtx; tag::AbstractString="")
    (ctx.resolution isa TOFtwin.GaussianTimingCDFResolution) || return nothing
    (ctx.cdf_work !== nothing) || return nothing

    # pick representative pixels by L2 (min/mid/max)
    L2s = [TOFtwin.L2(ctx.inst, p.id) for p in ctx.pixels]
    perm = sortperm(L2s)
    picks = unique(clamp.(round.(Int, LinRange(1, length(ctx.pixels), pychop_check_plot_npix)), 1, length(ctx.pixels)))
    pix_test = [ctx.pixels[perm[i]] for i in picks]
    L2_test = [TOFtwin.L2(ctx.inst, p.id) for p in pix_test]

    # PyChop oracle curve
    etrans_max = min(ctx.Ei_meV - 1e-3, ctx.Ei_meV)
    etrans, dE_pychop = _run_pychop_oracle_curve(
        python=pychop_python,
        script=pychop_script,
        instrument=String(ctx.instr),
        Ei_meV=ctx.Ei_meV,
        variant=pychop_variant,
        freq_hz=Vector{Float64}(pychop_freq_hz),
        tc_index=pychop_tc_index,
        use_tc_rss=pychop_use_tc_rss,
        delta_td_us=pychop_delta_td_us,
        etrans_min=0.0,
        etrans_max=etrans_max,
        npts=pychop_npts,
    )

    # downsample for the TOFtwin impulse-FWHM calculation (can be expensive)
    nplot = clamp(pychop_check_plot_npts, 5, length(etrans))
    idx = unique(clamp.(round.(Int, LinRange(1, length(etrans), nplot)), 1, length(etrans)))
    ωs = etrans[idx]
    dE_py = dE_pychop[idx]

    ntof = length(ctx.tof_edges_s) - 1
    L1 = Float64(ctx.inst.L1)

    # Compute TOFtwin FWHM(ω) for each test pixel
    curves = Vector{Vector{Float64}}(undef, length(pix_test))
    relerrs = Vector{Vector{Float64}}(undef, length(pix_test))

    for (ip, p) in enumerate(pix_test)
        L2p = Float64(L2_test[ip])
        dE_tw = fill(NaN, length(ωs))

        # Precompute ω edges for this pixel
        ω_edges = similar(ctx.tof_edges_s, Float64)
        for i in eachindex(ctx.tof_edges_s)
            Ef = _Ef_from_tof_s(L1, L2p, ctx.Ei_meV, Float64(ctx.tof_edges_s[i]))
            ω_edges[i] = isfinite(Ef) ? (ctx.Ei_meV - Ef) : NaN
        end
        ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])
        dω = ω_edges[2:end] .- ω_edges[1:end-1]

        C = zeros(Float64, 1, ntof)
        for (k, ω0) in enumerate(ωs)
            Ef0 = ctx.Ei_meV - ω0
            Ef0 <= 0 && continue
            t0_s = _tof_us(L1, L2p, ctx.Ei_meV, Ef0) * 1e-6
            i0 = searchsortedlast(ctx.tof_edges_s, t0_s)
            i0 = clamp(i0, 1, ntof)

            fill!(C, 0.0)
            C[1, i0] = 1.0

            TOFtwin.apply_tof_resolution!(C, ctx.inst, [p], ctx.Ei_meV, ctx.tof_edges_s, ctx.resolution; work=ctx.cdf_work)

            yω = similar(view(C, 1, :), Float64)
            @inbounds for j in 1:ntof
                denom = dω[j]
                yω[j] = (isfinite(denom) && denom > 0) ? (C[1, j] / denom) : 0.0
            end

            # Restrict to finite x-range
            good = findall(j -> isfinite(ω_cent[j]) && isfinite(yω[j]) && yω[j] >= 0, 1:ntof)
            if !isempty(good)
                x = ω_cent[good]
                y = yω[good]
                dE_tw[k] = _fwhm_monotonic_x(x, y)
            end
        end

        curves[ip] = dE_tw
        relerrs[ip] = (dE_tw .- dE_py) ./ dE_py
    end

    # Save a TSV table for quick inspection
    if do_save
        fname = isempty(tag) ? "pychop_resolution_compare.tsv" : "pychop_resolution_compare_$(tag).tsv"
        outtsv = joinpath(outdir, fname)
        open(outtsv, "w") do io
            println(io, "# etrans_meV\tdE_pychop_FWHM_meV\t" * join(["dE_toftwin_FWHM_meV_pid$(p.id)" for p in pix_test], "\t"))
            for k in eachindex(ωs)
                vals = ["$(ωs[k])", "$(dE_py[k])"]
                append!(vals, ["$(curves[ip][k])" for ip in eachindex(pix_test)])
                println(io, join(vals, "\t"))
            end
        end
        @info "Wrote $(outtsv)"
    end

    # Plot
    fig = Figure(size=(1200, 800))
    ax1 = Axis(fig[1,1], xlabel="ω (meV)", ylabel="ΔE FWHM (meV)", title="Resolution: PyChop vs TOFtwin (impulse+smear)")
    lines!(ax1, ωs, dE_py, label="PyChop (oracle)")
    for (ip, p) in enumerate(pix_test)
        lines!(ax1, ωs, curves[ip], label="TOFtwin pid=$(p.id), L2=$(round(L2_test[ip], digits=3)) m")
    end
    axislegend(ax1; position=:rb)

    ax2 = Axis(fig[2,1], xlabel="ω (meV)", ylabel="(TOFtwin-PyChop)/PyChop", title="Relative error")
    hlines!(ax2, [0.0])
    for (ip, p) in enumerate(pix_test)
        lines!(ax2, ωs, relerrs[ip], label="pid=$(p.id)")
    end
    axislegend(ax2; position=:rb)

    if do_save
        fname = isempty(tag) ? "pychop_resolution_compare.png" : "pychop_resolution_compare_$(tag).png"
        outpng = joinpath(outdir, fname)
        save(outpng, fig)
        @info "Wrote $(outpng)"
    end
    display(fig)
    return fig
end

ctx = step("setup_powder_ctx", () -> TOFtwin.setup_powder_ctx(;
    instr=instr,
    idf_path=idf_path,
    kern_Qmin=kern0.Q[1],
    kern_Qmax=kern0.Q[end],
    kern_wmin=kern0.ω[1],
    kern_wmax=kern0.ω[end],
    rebuild_geom=rebuild_geom,
    ψstride=ψstride, ηstride=ηstride,
    grouping=grouping,
    grouping_file=isempty(grouping_file) ? nothing : grouping_file,
    mask_btp=mask_btp,
    mask_mode=mask_mode,
    angle_step=angle_step,
    Ei=Ei, nQbins=nQbins, nωbins=nωbins,
    res_mode=res_mode, σt_us=σt_us, gh_order=gh_order, nsigma=nsigma,
    σt_source=σt_source,
    pychop_python=pychop_python,
    pychop_script=pychop_script,
    pychop_variant=pychop_variant,
    pychop_freq_hz=pychop_freq_hz,
    pychop_tc_index=pychop_tc_index,
    pychop_use_tc_rss=pychop_use_tc_rss,
    pychop_delta_td_us=pychop_delta_td_us,
    pychop_npts=pychop_npts,
    pychop_sigma_q=pychop_sigma_q,
	pychop_check=pychop_check,
	pychop_check_etrans_meV=pychop_check_etrans_meV,
    ntof_env=ntof_env, ntof_alpha=ntof_alpha, ntof_beta=ntof_beta, ntof_min=ntof_min, ntof_max=ntof_max,
    disk_cache=disk_cache, cache_dir=cache_dir, cache_ver=cache_ver,
    cache_Cv=cache_Cv, cache_rmap=cache_rmap
))

@info "SETUP complete. ctx pixels=$(length(ctx.pixels)) grouping='$(grouping)' mask='$(mask_btp)' res_mode='$(res_mode)'"

if pychop_check_plot && (σt_source == "pychop") && (res_mode == "cdf")
    step("plot pychop resolution compare", () -> plot_resolution_compare(ctx; tag=lowercase(String(instr))))
end

for (i, spath) in enumerate(sunny_paths)
    kern = (i == 1) ? kern0 : step("load Sunny kernel", () -> load_sunny_powder_jld2(spath; outside=0.0))
    model = (q,w) -> kern(q,w)

    tag = "k$(i)"
    @info "EVALUATE kernel $i / $(length(sunny_paths))" path=spath

    pred = step("eval_powder", () -> TOFtwin.eval_powder(ctx, model; do_hist=do_hist))

    if do_hist
        plot_result(ctx, kern, pred; tag=tag)
    end
end

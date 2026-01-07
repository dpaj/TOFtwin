
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
# Artificial "delta-comb" kernel (no Sunny)
#
# Idea: a regular grid of peaks spaced by:
#   ΔQ = 0.5 Å⁻¹,  Δω = 0.5 meV
#
# Because eval_powder samples a continuous model, true Dirac deltas would be
# missed. We approximate deltas with narrow Gaussians in Q and ω.
#
# Env knobs:
#   TOFTWIN_KERNEL=delta|sunny
#   TOFTWIN_KERN_QMIN, TOFTWIN_KERN_QMAX
#   TOFTWIN_KERN_WMIN, TOFTWIN_KERN_WMAX
#   TOFTWIN_DELTA_DQ, TOFTWIN_DELTA_DW
#   TOFTWIN_DELTA_SIGMA_Q, TOFTWIN_DELTA_SIGMA_W
#   TOFTWIN_DELTA_FINE_DQ,  TOFTWIN_DELTA_FINE_DW
#   TOFTWIN_DELTA_AMP
# -----------------------------

function _delta_comb_value(q, w; dQ=0.5, dW=0.5, σQ=0.03, σW=0.03, amp=1.0)
    (q < 0 || w < 0) && return 0.0
    dq = abs(q - round(q/dQ) * dQ)
    dw = abs(w - round(w/dW) * dW)
    return amp * exp(-0.5*(dq/σQ)^2 - 0.5*(dw/σW)^2)
end

function make_delta_comb_kernel(; Qmin::Real=0.0, Qmax::Real=6.0, wmin::Real=0.0, wmax::Real=25.0,
                                dQ::Real=0.5, dW::Real=0.5, σQ::Real=0.03, σW::Real=0.03,
                                fine_dQ::Real=0.05, fine_dW::Real=0.05, amp::Real=1.0,
                                outside::Real=0.0)
    Q = collect(range(Float64(Qmin), Float64(Qmax); step=Float64(fine_dQ)))
    W = collect(range(Float64(wmin), Float64(wmax); step=Float64(fine_dW)))
    S = [ _delta_comb_value(q, w; dQ=dQ, dW=dW, σQ=σQ, σW=σW, amp=amp) for q in Q, w in W ]
    return TOFtwin.GridKernelPowder(Q, W, S; outside=outside)
end




# -----------------------------
# Instrument selection
# -----------------------------
function parse_instrument()
    s = uppercase(get(ENV, "TOFTWIN_INSTRUMENT", "SEQUOIA"))
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
            ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="Kernel S(|Q|,ω)")
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

# -----------------------------
# Optional: plot PyChop vs TOFtwin-implied ΔE(ω)
# -----------------------------

const pychop_check_plot = lowercase(get(ENV, "TOFTWIN_PYCHOP_CHECK_PLOT", "0")) in ("1","true","yes")
const pychop_check_plot_npts = parse(Int, get(ENV, "TOFTWIN_PYCHOP_CHECK_PLOT_NPTS", "121"))
const pychop_check_plot_npix = parse(Int, get(ENV, "TOFTWIN_PYCHOP_CHECK_PLOT_NPIX", "3"))

function _write_tsv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "pixel_id\tL2_m\tetrans_meV\tdE_pychop_FWHM_meV\tdE_toftwin_FWHM_meV\trel_err")
        for r in rows
            println(io,
                "$(r.pixel_id)\t$(r.L2_m)\t$(r.etrans_meV)\t$(r.dE_pychop_FWHM_meV)\t$(r.dE_toftwin_FWHM_meV)\t$(r.rel_err)")
        end
    end
end

function plot_pychop_resolution_compare(ctx::TOFtwin.PowderCtx; tag::AbstractString="")
    # Only meaningful for CDF timing resolution
    (ctx.resolution isa TOFtwin.GaussianTimingCDFResolution) || return nothing
    (ctx.cdf_work !== nothing) || return nothing

    # Pull a smooth ΔE(ω) curve from the PyChop oracle.
    # NOTE: `etrans_max` must be specified *before* we have `et_oracle`.
    et_oracle, dE_oracle = TOFtwin._run_pychop_oracle_dE(
        python=pychop_python,
        script=pychop_script,
        instrument=ctx.instr,
        Ei_meV=ctx.Ei_meV,
        variant=pychop_variant,
        freq_hz=pychop_freq_hz,
        tc_index=pychop_tc_index,
        use_tc_rss=pychop_use_tc_rss,
        delta_td_us=pychop_delta_td_us,
        etrans_min=0.0,
        etrans_max=max(0.0, ctx.Ei_meV - 1e-3),
        npts=pychop_npts,
    )

    meta = (etrans_meV = Vector{Float64}(et_oracle), dE_fwhm_meV = Vector{Float64}(dE_oracle))

    emax = min(ctx.Ei_meV - 1e-3, maximum(meta.etrans_meV))
    et_test = collect(range(max(0.0, meta.etrans_meV[1]), emax; length=max(5, pychop_check_plot_npts)))

    rows = TOFtwin.pychop_check_dE(
        ctx.inst, ctx.pixels, ctx.Ei_meV, ctx.tof_edges_s, ctx.resolution, ctx.cdf_work, meta;
        etrans_test_meV=et_test,
        npix_test=pychop_check_plot_npix,
    )
    isempty(rows) && return nothing

    # group rows by pixel_id
    pids = unique(r.pixel_id for r in rows)
    sort!(pids)

    fig = Figure(size=(1400, 900))
    ax1 = Axis(fig[1, 1], xlabel="Etransfer (meV)", ylabel="ΔE FWHM (meV)",
        title="Resolution comparison (PyChop oracle vs TOFtwin-implied)")
    lines!(ax1, meta.etrans_meV, meta.dE_fwhm_meV, label="PyChop oracle")

    ax2 = Axis(fig[2, 1], xlabel="Etransfer (meV)", ylabel="(TOFtwin − PyChop)/PyChop",
        title="Relative error")
    hlines!(ax2, [0.0])

    for pid in pids
        rr = [r for r in rows if r.pixel_id == pid]
        sort!(rr, by = r -> r.etrans_meV)
        et = [r.etrans_meV for r in rr]
        dE_t = [r.dE_toftwin_FWHM_meV for r in rr]
        rel = [r.rel_err for r in rr]
        L2m = rr[1].L2_m
        lines!(ax1, et, dE_t, label="TOFtwin pixel $(pid) (L2=$(round(L2m, digits=3)) m)")
        lines!(ax2, et, rel, label="pixel $(pid)")
    end

    axislegend(ax1; position=:rb)
    axislegend(ax2; position=:rb)

    if do_save
        base = isempty(tag) ? "pychop_resolution_compare_$(lowercase(String(ctx.instr)))" :
                              "pychop_resolution_compare_$(lowercase(String(ctx.instr)))_$(tag)"
        outpng = joinpath(outdir, base * ".png")
        outtsv = joinpath(outdir, base * ".tsv")
        save(outpng, fig)
        _write_tsv(outtsv, rows)
        @info "Wrote $outpng"
        @info "Wrote $outtsv"
    end

    display(fig)
    return nothing
end

# -----------------------------------------------------------------------------
# Main: build ctx once, then evaluate 1..N Sunny kernels
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main: build ctx once, then evaluate either:
#   - 1..N Sunny kernels from JLD2 files (TOFTWIN_KERNEL=sunny; default), or
#   - an artificial "delta-comb" kernel (TOFTWIN_KERNEL=delta)
#
# Examples:
#   # Sunny (unchanged behavior)
#   TOFTWIN_INSTRUMENT=CNCS julia --project=. scripts/demo_powder_ctx_setup_eval_groupmask_refac_delta_demo.jl \
#       ../sunny_powder_corh2o4.jld2
#
#   # Artificial delta-comb: peaks every 0.5 Å^-1 and 0.5 meV
#   TOFTWIN_INSTRUMENT=CNCS TOFTWIN_KERNEL=delta TOFTWIN_EI=25.0 \
#       julia --project=. scripts/demo_powder_ctx_setup_eval_groupmask_refac_delta_demo.jl
#
#   # Make spikes sharper/broader
#   TOFTWIN_KERNEL=delta TOFTWIN_DELTA_SIGMA_Q=0.02 TOFTWIN_DELTA_SIGMA_W=0.02 ...
# -----------------------------------------------------------------------------

kernel_kind = lowercase(strip(get(ENV, "TOFTWIN_KERNEL", "delta")))

instr = parse_instrument()
idf_path = instr === :CNCS ? joinpath(@__DIR__, "CNCS_Definition_2025B.xml") :
                            joinpath(@__DIR__, "SEQUOIA_Definition.xml")

default_Ei = instr === :SEQUOIA ? "30.0" : "12.0"
Ei = parse(Float64, get(ENV, "TOFTWIN_EI", default_Ei))  # meV

# Choose kernel (only used to set ctx axes + optional plotting)
sunny_paths = String[]
kern0 = nothing

if kernel_kind == "sunny"
    sunny_paths = length(ARGS) > 0 ? ARGS : [joinpath(@__DIR__, "..", "sunny_powder_corh2o4.jld2")]
    kern0 = step("load Sunny kernel (axes anchor)", () -> load_sunny_powder_jld2(sunny_paths[1]; outside=0.0))
elseif kernel_kind in ("delta", "comb", "delta-comb", "deltacomb")
    Qmin = parse(Float64, get(ENV, "TOFTWIN_KERN_QMIN", "0.0"))
    Qmax = parse(Float64, get(ENV, "TOFTWIN_KERN_QMAX", "6.0"))
    wmin = parse(Float64, get(ENV, "TOFTWIN_KERN_WMIN", "0.0"))
    wmax = parse(Float64, get(ENV, "TOFTWIN_KERN_WMAX", string(Ei)))

    dQ  = parse(Float64, get(ENV, "TOFTWIN_DELTA_DQ", "0.5"))
    dW  = parse(Float64, get(ENV, "TOFTWIN_DELTA_DW", "1.0"))
    σQ  = parse(Float64, get(ENV, "TOFTWIN_DELTA_SIGMA_Q", "0.03"))
    σW  = parse(Float64, get(ENV, "TOFTWIN_DELTA_SIGMA_W", "0.03"))
    fdQ = parse(Float64, get(ENV, "TOFTWIN_DELTA_FINE_DQ", "0.025"))
    fdW = parse(Float64, get(ENV, "TOFTWIN_DELTA_FINE_DW", "0.05"))
    amp = parse(Float64, get(ENV, "TOFTWIN_DELTA_AMP", "1.0"))

    @info "Using artificial delta-comb kernel" Qmin=Qmin Qmax=Qmax wmin=wmin wmax=wmax dQ=dQ dW=dW σQ=σQ σW=σW fine_dQ=fdQ fine_dW=fdW amp=amp
    kern0 = step("build artificial kernel (axes anchor)", () -> make_delta_comb_kernel(
        Qmin=Qmin, Qmax=Qmax, wmin=wmin, wmax=wmax,
        dQ=dQ, dW=dW, σQ=σQ, σW=σW,
        fine_dQ=fdQ, fine_dW=fdW,
        amp=amp, outside=0.0
    ))
else
    throw(ArgumentError("TOFTWIN_KERNEL must be 'sunny' or 'delta' (got '$kernel_kind')"))
end

# -----------------------------
# Binning / resolution knobs
# -----------------------------
nQbins = parse(Int, get(ENV, "TOFTWIN_NQBINS", "420"))
nωbins = parse(Int, get(ENV, "TOFTWIN_NWBINS", "440"))

ψstride = parse(Int, get(ENV, "TOFTWIN_PSI_STRIDE", "1"))
ηstride = parse(Int, get(ENV, "TOFTWIN_ETA_STRIDE", "1"))

res_mode = lowercase(get(ENV, "TOFTWIN_RES_MODE", "cdf"))
σt_us    = parse(Float64, get(ENV, "TOFTWIN_SIGMA_T_US", "10.0"))
gh_order = parse(Int, get(ENV, "TOFTWIN_GH_ORDER", "3"))
nsigma   = parse(Float64, get(ENV, "TOFTWIN_NSIGMA", "4.0"))

# Timing width source:
#   TOFTWIN_SIGMA_T_SOURCE=manual -> use TOFTWIN_SIGMA_T_US
#   TOFTWIN_SIGMA_T_SOURCE=pychop -> derive an *effective* σt(t) curve from PyChop ΔE(ω)
σt_source = lowercase(get(ENV, "TOFTWIN_SIGMA_T_SOURCE", "pychop"))

pychop_check = lowercase(get(ENV, "TOFTWIN_PYCHOP_CHECK", "1")) in ("1", "true", "yes")
pychop_check_etrans_meV = Float64[]

# PyChop oracle script (used when TOFTWIN_SIGMA_T_SOURCE=pychop)
pychop_python = get(ENV, "TOFTWIN_PYCHOP_PYTHON", get(ENV, "PYTHON", Sys.iswindows() ? raw"C:\Users\vdp\AppData\Local\Microsoft\WindowsApps\python3.11.exe" : "python3"))
pychop_script = get(ENV, "TOFTWIN_PYCHOP_SCRIPT", joinpath(@__DIR__, "pychop_oracle_dE.py"))
pychop_variant = get(ENV, "TOFTWIN_PYCHOP_VARIANT", "SEQ-100-2.0-AST") #"High Flux"
pychop_freq_str = get(ENV, "TOFTWIN_PYCHOP_FREQ_HZ", "600") #"300,60"
pychop_freq_hz = isempty(strip(pychop_freq_str)) ? Float64[] : parse.(Float64, split(pychop_freq_str, r"[^0-9eE+\-.]+"; keepempty=false))
pychop_tc_index = parse(Int, get(ENV, "TOFTWIN_PYCHOP_TC_INDEX", "0"))
pychop_use_tc_rss = lowercase(get(ENV, "TOFTWIN_PYCHOP_TC_RSS", "0")) in ("1","true","yes")
pychop_delta_td_us = parse(Float64, get(ENV, "TOFTWIN_PYCHOP_DELTA_TD_US", "0.0"))
pychop_npts = parse(Int, get(ENV, "TOFTWIN_PYCHOP_NPTS", "401"))
pychop_sigma_q = parse(Float64, get(ENV, "TOFTWIN_PYCHOP_SIGMA_Q", "0.25"))

ntof_env   = get(ENV, "TOFTWIN_NTOF", "auto")
ntof_alpha = parse(Float64, get(ENV, "TOFTWIN_NTOF_ALPHA", "0.5"))
ntof_beta  = parse(Float64, get(ENV, "TOFTWIN_NTOF_BETA",  "0.333333333333"))
ntof_min   = parse(Int, get(ENV, "TOFTWIN_NTOF_MIN", "200"))
ntof_max   = parse(Int, get(ENV, "TOFTWIN_NTOF_MAX", "2000"))

rebuild_geom = lowercase(get(ENV, "TOFTWIN_REBUILD_GEOM", "0")) in ("1","true","yes")

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

if hasproperty(ctx, :pychop_spec) && ctx.pychop_spec !== nothing
    @info "PyChop model spec" ctx.pychop_spec
end

if pychop_check_plot
    step("plot_pychop_resolution_compare", () -> plot_pychop_resolution_compare(ctx))
end

@info "SETUP complete. ctx pixels=$(length(ctx.pixels)) grouping='$(grouping)' mask='$(mask_btp)' res_mode='$(res_mode)' kernel='$(kernel_kind)'"

if kernel_kind == "sunny"
    for (i, spath) in enumerate(sunny_paths)
        kern = (i == 1) ? kern0 : step("load Sunny kernel", () -> load_sunny_powder_jld2(spath; outside=0.0))
        model = (q,w) -> kern(q,w)

        tag = "k$(i)"
        @info "EVALUATE kernel $i / $(length(sunny_paths))" path=spath

        local pred = step("eval_powder", () -> TOFtwin.eval_powder(ctx, model; do_hist=do_hist))

        if do_hist
            plot_result(ctx, kern, pred; tag=tag)
        end
    end
else
    # single evaluation for artificial kernel
    kern = kern0
    model = (q,w) -> kern(q,w)
    tag = "delta"
    @info "EVALUATE artificial kernel" tag=tag

    local pred = step("eval_powder", () -> TOFtwin.eval_powder(ctx, model; do_hist=do_hist))

    if do_hist
        plot_result(ctx, kern, pred; tag=tag)
    end
end

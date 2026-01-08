using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using TOFtwin
using JLD2
using LinearAlgebra
using TOML

# -----------------------------------------------------------------------------
# Demo: TOML-driven wrapper around:
#   demo_powder_ctx_setup_eval_groupmask_refac_delta_demo.jl
#
# Goals:
# - Keep ALL user controls visible in a config file
# - Still allow TOFTWIN_* env overrides for quick experiments
# - Keep the core workflow identical: setup_powder_ctx once, eval_powder per model
# -----------------------------------------------------------------------------

# -----------------------------
# Helpers
# -----------------------------
const _CFG_PATH = let
    # Priority: ARGS[1] if provided, else TOFTWIN_CONFIG, else default name.
    if length(ARGS) >= 1 && endswith(lowercase(ARGS[1]), ".toml")
        abspath(ARGS[1])
    else
        abspath(get(ENV, "TOFTWIN_CONFIG", joinpath(@__DIR__, "demo_powder_CNCS_HighFlux_Ei12_f300-60.toml")))
        #abspath(get(ENV, "TOFTWIN_CONFIG", joinpath(@__DIR__, "demo_powder_CNCS_HighFlux_Ei3.32_f300-60.toml")))
        #abspath(get(ENV, "TOFTWIN_CONFIG", joinpath(@__DIR__, "demo_powder_SEQUOIA_SEQ-100-2.0-AST_Ei160_f600.toml")))
    end
end

const _CFG_DIR = dirname(_CFG_PATH)
const CFG = TOML.parsefile(_CFG_PATH)

cfg_get(tbl, keys::Vector{String}, default) = begin
    t = tbl
    for k in keys[1:end-1]
        t = get(t, k, Dict{String,Any}())
        isa(t, Dict) || return default
    end
    return get(t, keys[end], default)
end

# Env override: if envkey exists, parse using `parsefn`; else use cfg value.
function cfg_get_env(envkey::String, keys::Vector{String}, default; parsefn=identity)
    if haskey(ENV, envkey)
        return parsefn(get(ENV, envkey, string(default)))
    else
        return cfg_get(CFG, keys, default)
    end
end

env_bool(s) = lowercase(strip(s)) in ("1","true","yes","y","on")
env_float_list(s) = isempty(strip(s)) ? Float64[] : parse.(Float64, split(s, r"[^0-9eE+\-.]+"; keepempty=false))

# Make relative paths resolve relative to config dir.
function resolve_path(p::AbstractString)
    isempty(strip(p)) && return ""
    isabspath(p) && return p
    return normpath(joinpath(_CFG_DIR, p))
end

function parse_instrument(s::AbstractString)
    u = uppercase(strip(s))
    if u in ("CNCS",)
        return :CNCS
    elseif u in ("SEQUOIA","SEQ")
        return :SEQUOIA
    else
        throw(ArgumentError("instrument.name must be CNCS or SEQUOIA (got '$s')"))
    end
end

# -----------------------------
# Read config (with optional env overrides)
# -----------------------------
backend = cfg_get_env("TOFTWIN_BACKEND", ["run","backend"], "gl"; parsefn=lowercase)

const DO_PROFILE = cfg_get_env("TOFTWIN_PROFILE", ["run","profile"], false; parsefn=env_bool)

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

do_hist     = cfg_get_env("TOFTWIN_DO_HIST", ["run","do_hist"], true; parsefn=env_bool)
plot_kernel = cfg_get_env("TOFTWIN_PLOT_KERNEL", ["run","plot_kernel"], true; parsefn=env_bool)
do_save     = cfg_get_env("TOFTWIN_SAVE", ["run","save"], true; parsefn=env_bool)
outdir_cfg  = cfg_get_env("TOFTWIN_OUTDIR", ["run","outdir"], "out"; parsefn=String)
outdir      = resolve_path(outdir_cfg)
do_save && mkpath(outdir)

# plotting backend
if do_hist
    if backend == "gl"
        using GLMakie
    else
        using CairoMakie
    end
end

# cache
disk_cache  = cfg_get_env("TOFTWIN_DISK_CACHE", ["cache","disk_cache"], true; parsefn=env_bool)
cache_dir   = resolve_path(cfg_get_env("TOFTWIN_CACHE_DIR", ["cache","cache_dir"], ".toftwin_cache"; parsefn=String))
cache_ver   = cfg_get_env("TOFTWIN_CACHE_VERSION", ["cache","cache_version"], "ctx_v2"; parsefn=String)
cache_Cv    = cfg_get_env("TOFTWIN_CACHE_CV", ["cache","cache_Cv"], false; parsefn=env_bool)
cache_rmap  = cfg_get_env("TOFTWIN_CACHE_RMAP", ["cache","cache_rmap"], false; parsefn=env_bool)
disk_cache && mkpath(cache_dir)

# grouping/masking
grouping      = cfg_get_env("TOFTWIN_GROUPING", ["grouping_masking","grouping"], "4x2"; parsefn=String)
grouping_file = resolve_path(cfg_get_env("TOFTWIN_GROUPING_FILE", ["grouping_masking","grouping_file"], ""; parsefn=String))
mask_btp      = cfg_get_env("TOFTWIN_MASK_BTP", ["grouping_masking","mask_btp"], "Bank=36-50"; parsefn=String)
mask_mode     = Symbol(lowercase(cfg_get_env("TOFTWIN_MASK_MODE", ["grouping_masking","mask_mode"], "drop"; parsefn=String)))
angle_step    = cfg_get_env("TOFTWIN_POWDER_ANGLESTEP", ["grouping_masking","powder_anglestep_deg"], 0.5; parsefn=x->parse(Float64,x))

# instrument
instr_name = cfg_get_env("TOFTWIN_INSTRUMENT", ["instrument","name"], "SEQUOIA"; parsefn=String)
instr = parse_instrument(instr_name)

default_Ei = instr === :SEQUOIA ? 30.0 : 12.0
Ei = cfg_get_env("TOFTWIN_EI", ["instrument","Ei_meV"], default_Ei; parsefn=x->parse(Float64,x))

rebuild_geom = cfg_get_env("TOFTWIN_REBUILD_GEOM", ["instrument","rebuild_geom"], false; parsefn=env_bool)

idf_override = cfg_get_env("TOFTWIN_IDF_PATH", ["instrument","idf_path"], ""; parsefn=String)
idf_path = if !isempty(strip(idf_override))
    resolve_path(idf_override)
else
    instr === :CNCS ? joinpath(@__DIR__, "CNCS_Definition_2025B.xml") :
                      joinpath(@__DIR__, "SEQUOIA_Definition.xml")
end

# bins/strides
nQbins = cfg_get_env("TOFTWIN_NQBINS", ["bins","nQbins"], 420; parsefn=x->parse(Int,x))
nωbins = cfg_get_env("TOFTWIN_NWBINS", ["bins","nWbins"], 440; parsefn=x->parse(Int,x))
ψstride = cfg_get_env("TOFTWIN_PSI_STRIDE", ["bins","psi_stride"], 1; parsefn=x->parse(Int,x))
ηstride = cfg_get_env("TOFTWIN_ETA_STRIDE", ["bins","eta_stride"], 1; parsefn=x->parse(Int,x))

# resolution
res_mode = lowercase(cfg_get_env("TOFTWIN_RES_MODE", ["resolution","res_mode"], "cdf"; parsefn=String))
σt_us    = cfg_get_env("TOFTWIN_SIGMA_T_US", ["resolution","sigma_t_us"], 10.0; parsefn=x->parse(Float64,x))
gh_order = cfg_get_env("TOFTWIN_GH_ORDER", ["resolution","gh_order"], 3; parsefn=x->parse(Int,x))
nsigma   = cfg_get_env("TOFTWIN_NSIGMA", ["resolution","nsigma"], 4.0; parsefn=x->parse(Float64,x))
σt_source = lowercase(cfg_get_env("TOFTWIN_SIGMA_T_SOURCE", ["resolution","sigma_t_source"], "pychop"; parsefn=String))
pychop_check = cfg_get_env("TOFTWIN_PYCHOP_CHECK", ["resolution","pychop_check"], true; parsefn=env_bool)
pychop_check_etrans_meV = Float64[]  # keep same behavior as demo

# pychop
py_py = cfg_get(CFG, ["resolution","pychop","python"], "")
pychop_python = if haskey(ENV, "TOFTWIN_PYCHOP_PYTHON")
    get(ENV, "TOFTWIN_PYCHOP_PYTHON")
elseif haskey(ENV, "PYTHON")
    get(ENV, "PYTHON")
elseif isempty(strip(py_py))
    Sys.iswindows() ? raw"C:\Users\vdp\AppData\Local\Microsoft\WindowsApps\python3.11.exe" : "python3"
else
    py_py
end

pychop_script = haskey(ENV, "TOFTWIN_PYCHOP_SCRIPT") ? get(ENV, "TOFTWIN_PYCHOP_SCRIPT") :
               resolve_path(cfg_get(CFG, ["resolution","pychop","script"], "pychop_oracle_dE.py"))
# default script location in original demo: joinpath(@__DIR__, "pychop_oracle_dE.py")
if !isabspath(pychop_script)
    pychop_script = joinpath(@__DIR__, pychop_script)
end

pychop_variant = cfg_get_env("TOFTWIN_PYCHOP_VARIANT", ["resolution","pychop","variant"], "SEQ-100-2.0-AST"; parsefn=String)

if haskey(ENV, "TOFTWIN_PYCHOP_FREQ_HZ")
    pychop_freq_hz = env_float_list(get(ENV, "TOFTWIN_PYCHOP_FREQ_HZ"))
else
    pychop_freq_hz = Float64.(cfg_get(CFG, ["resolution","pychop","freq_hz"], [600.0]))
end

pychop_tc_index = cfg_get_env("TOFTWIN_PYCHOP_TC_INDEX", ["resolution","pychop","tc_index"], 0; parsefn=x->parse(Int,x))
pychop_use_tc_rss = cfg_get_env("TOFTWIN_PYCHOP_TC_RSS", ["resolution","pychop","use_tc_rss"], false; parsefn=env_bool)
pychop_delta_td_us = cfg_get_env("TOFTWIN_PYCHOP_DELTA_TD_US", ["resolution","pychop","delta_td_us"], 0.0; parsefn=x->parse(Float64,x))
pychop_npts = cfg_get_env("TOFTWIN_PYCHOP_NPTS", ["resolution","pychop","npts"], 401; parsefn=x->parse(Int,x))
pychop_sigma_q = cfg_get_env("TOFTWIN_PYCHOP_SIGMA_Q", ["resolution","pychop","sigma_q"], 0.25; parsefn=x->parse(Float64,x))

# optional resolution check plot
pychop_check_plot = cfg_get_env("TOFTWIN_PYCHOP_CHECK_PLOT", ["resolution","check_plot","enable"], false; parsefn=env_bool)
pychop_check_plot_npts = cfg_get_env("TOFTWIN_PYCHOP_CHECK_PLOT_NPTS", ["resolution","check_plot","npts"], 121; parsefn=x->parse(Int,x))
pychop_check_plot_npix = cfg_get_env("TOFTWIN_PYCHOP_CHECK_PLOT_NPIX", ["resolution","check_plot","npix"], 3; parsefn=x->parse(Int,x))

# ntof
ntof_env   = cfg_get_env("TOFTWIN_NTOF", ["ntof","mode"], "auto"; parsefn=String)
ntof_alpha = cfg_get_env("TOFTWIN_NTOF_ALPHA", ["ntof","alpha"], 0.5; parsefn=x->parse(Float64,x))
ntof_beta  = cfg_get_env("TOFTWIN_NTOF_BETA", ["ntof","beta"], 0.333333333333; parsefn=x->parse(Float64,x))
ntof_min   = cfg_get_env("TOFTWIN_NTOF_MIN", ["ntof","min"], 200; parsefn=x->parse(Int,x))
ntof_max   = cfg_get_env("TOFTWIN_NTOF_MAX", ["ntof","max"], 2000; parsefn=x->parse(Int,x))

# kernel selection
kernel_kind = lowercase(strip(cfg_get_env("TOFTWIN_KERNEL", ["kernel","kind"], "delta"; parsefn=String)))

# -----------------------------
# Sunny loader (unchanged)
# -----------------------------
function load_sunny_powder_jld2(path::AbstractString; outside=0.0)
    radii = energies = S_Qω = nothing
    @load path radii energies S_Qω

    Q = collect(radii)
    W = collect(energies)
    S = Matrix(S_Qω)

    @info "Loaded Sunny table: size(S)=$(size(S)), len(Q)=$(length(Q)), len(ω)=$(length(W))"

    if size(S) == (length(W), length(Q))
        @info "Transposing Sunny table (ω,Q) -> (Q,ω)"
        S = permutedims(S)
    end
    @assert size(S) == (length(Q), length(W))

    return TOFtwin.GridKernelPowder(Q, W, S; outside=outside)
end

# -----------------------------
# Delta-comb kernel (unchanged)
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
# Plot (copied from original demo)
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
    (ctx.resolution isa TOFtwin.GaussianTimingCDFResolution) || return nothing
    (ctx.cdf_work !== nothing) || return nothing

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

# -----------------------------
# Choose kernel (used to set ctx axes + optional plotting)
# -----------------------------
sunny_paths = String[]
kern0 = nothing

if kernel_kind == "sunny"
    cfg_paths = String.(cfg_get(CFG, ["kernel","sunny_paths"], String[]))
    extra = [a for a in ARGS[1:end] if !endswith(lowercase(a), ".toml")]
    sunny_paths = length(extra) > 0 ? extra :
                 (length(cfg_paths) > 0 ? resolve_path.(cfg_paths) : [joinpath(@__DIR__, "..", "sunny_powder_corh2o4.jld2")])
    kern0 = step("load Sunny kernel (axes anchor)", () -> load_sunny_powder_jld2(sunny_paths[1]; outside=0.0))
elseif kernel_kind in ("delta", "comb", "delta-comb", "deltacomb")
    Qmin = cfg_get_env("TOFTWIN_KERN_QMIN", ["kernel","Qmin"], 0.0; parsefn=x->parse(Float64,x))
    Qmax = cfg_get_env("TOFTWIN_KERN_QMAX", ["kernel","Qmax"], 6.0; parsefn=x->parse(Float64,x))
    wmin = cfg_get_env("TOFTWIN_KERN_WMIN", ["kernel","wmin"], 0.0; parsefn=x->parse(Float64,x))
    wmax_default = Float64(cfg_get(CFG, ["kernel","wmax"], Ei))
    wmax = cfg_get_env("TOFTWIN_KERN_WMAX", ["kernel","wmax"], wmax_default; parsefn=x->parse(Float64,x))

    dQ  = cfg_get_env("TOFTWIN_DELTA_DQ", ["kernel","delta_dQ"], 0.5; parsefn=x->parse(Float64,x))
    dW  = cfg_get_env("TOFTWIN_DELTA_DW", ["kernel","delta_dW"], 1.0; parsefn=x->parse(Float64,x))
    σQ  = cfg_get_env("TOFTWIN_DELTA_SIGMA_Q", ["kernel","delta_sigma_Q"], 0.03; parsefn=x->parse(Float64,x))
    σW  = cfg_get_env("TOFTWIN_DELTA_SIGMA_W", ["kernel","delta_sigma_W"], 0.03; parsefn=x->parse(Float64,x))
    fdQ = cfg_get_env("TOFTWIN_DELTA_FINE_DQ", ["kernel","delta_fine_dQ"], 0.025; parsefn=x->parse(Float64,x))
    fdW = cfg_get_env("TOFTWIN_DELTA_FINE_DW", ["kernel","delta_fine_dW"], 0.05; parsefn=x->parse(Float64,x))
    amp = cfg_get_env("TOFTWIN_DELTA_AMP", ["kernel","delta_amp"], 1.0; parsefn=x->parse(Float64,x))

    @info "Using artificial delta-comb kernel" Qmin=Qmin Qmax=Qmax wmin=wmin wmax=wmax dQ=dQ dW=dW σQ=σQ σW=σW fine_dQ=fdQ fine_dW=fdW amp=amp
    kern0 = step("build artificial kernel (axes anchor)", () -> make_delta_comb_kernel(
        Qmin=Qmin, Qmax=Qmax, wmin=wmin, wmax=wmax,
        dQ=dQ, dW=dW, σQ=σQ, σW=σW,
        fine_dQ=fdQ, fine_dW=fdW,
        amp=amp, outside=0.0
    ))
else
    throw(ArgumentError("kernel.kind must be 'sunny' or 'delta' (got '$kernel_kind')"))
end

# -----------------------------
# Setup ctx
# -----------------------------
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

@info "SETUP complete" cfg=_CFG_PATH pixels=length(ctx.pixels) grouping=grouping mask=mask_btp res_mode=res_mode kernel=kernel_kind Ei_meV=Ei

# -----------------------------
# Evaluate
# -----------------------------
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
    kern = kern0
    model = (q,w) -> kern(q,w)
    tag = "delta"
    @info "EVALUATE artificial kernel" tag=tag

    local pred = step("eval_powder", () -> TOFtwin.eval_powder(ctx, model; do_hist=do_hist))

    if do_hist
        plot_result(ctx, kern, pred; tag=tag)
    end
end

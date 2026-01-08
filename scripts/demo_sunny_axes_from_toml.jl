using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using TOFtwin
using TOML
using JLD2

# Optional plotting
backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
do_plot = lowercase(get(ENV, "TOFTWIN_PLOT_COVERAGE", "true")) in ("1","true","yes","y","on")
if do_plot
    if backend == "cairo"
        using CairoMakie
    else
        using GLMakie
    end
end

# -----------------------------------------------------------------------------
# Demo: TOML-driven helper to suggest Sunny powder (|Q|, ω) axes from an
# instrument configuration.
#
# The goal is to:
#   1) load the instrument configuration from the same TOML used by the powder workflow
#   2) build a PowderCtx (so detector selection/masking, TOF binning, etc. match)
#   3) compute a reachable (|Q|, ω) point cloud and suggest Sunny radii/energies grids
# -----------------------------------------------------------------------------

# -----------------------------
# Config helpers (trimmed from demo_powder_from_toml.jl)
# -----------------------------
const _CFG_PATH = let
    if length(ARGS) >= 1 && endswith(lowercase(ARGS[1]), ".toml")
        abspath(ARGS[1])
    else
        #abspath(get(ENV, "TOFTWIN_CONFIG", joinpath(@__DIR__, "demo_powder_CNCS_HighFlux_Ei12_f300-60.toml")))
        #abspath(get(ENV, "TOFTWIN_CONFIG", joinpath(@__DIR__, "demo_powder_SEQUOIA_SEQ-100-2.0-AST_Ei160_f600.toml")))
        abspath(get(ENV, "TOFTWIN_CONFIG", joinpath(@__DIR__, "demo_powder_CNCS_HighFlux_Ei3.32_f300-60.toml")))
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

function cfg_get_env(envkey::String, keys::Vector{String}, default; parsefn=identity)
    if haskey(ENV, envkey)
        return parsefn(get(ENV, envkey, string(default)))
    else
        return cfg_get(CFG, keys, default)
    end
end

env_bool(s) = lowercase(strip(s)) in ("1","true","yes","y","on")
env_float_list(s) = isempty(strip(s)) ? Float64[] : parse.(Float64, split(s, r"[^0-9eE+\-.]+"; keepempty=false))

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
outdir_cfg = cfg_get_env("TOFTWIN_OUTDIR", ["run","outdir"], "out"; parsefn=String)
outdir = resolve_path(outdir_cfg)
mkpath(outdir)

# instrument
instr_name = cfg_get_env("TOFTWIN_INSTRUMENT", ["instrument","name"], "CNCS"; parsefn=String)
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

# grouping/masking
grouping      = cfg_get_env("TOFTWIN_GROUPING", ["grouping_masking","grouping"], "4x2"; parsefn=String)
grouping_file = resolve_path(cfg_get_env("TOFTWIN_GROUPING_FILE", ["grouping_masking","grouping_file"], ""; parsefn=String))
mask_btp      = cfg_get_env("TOFTWIN_MASK_BTP", ["grouping_masking","mask_btp"], "Bank=36-50"; parsefn=String)
mask_mode     = Symbol(lowercase(cfg_get_env("TOFTWIN_MASK_MODE", ["grouping_masking","mask_mode"], "drop"; parsefn=String)))
angle_step    = cfg_get_env("TOFTWIN_POWDER_ANGLESTEP", ["grouping_masking","powder_anglestep_deg"], 0.5; parsefn=x->parse(Float64,x))

# bins/strides (these affect which pixels are used)
ψstride = cfg_get_env("TOFTWIN_PSI_STRIDE", ["bins","psi_stride"], 1; parsefn=x->parse(Int,x))
ηstride = cfg_get_env("TOFTWIN_ETA_STRIDE", ["bins","eta_stride"], 1; parsefn=x->parse(Int,x))

# resolution + PyChop (kept consistent with powder ctx setup; TOF binning depends on ntof, not ΔE)
res_mode = lowercase(cfg_get_env("TOFTWIN_RES_MODE", ["resolution","res_mode"], "cdf"; parsefn=String))
σt_us    = cfg_get_env("TOFTWIN_SIGMA_T_US", ["resolution","sigma_t_us"], 10.0; parsefn=x->parse(Float64,x))
gh_order = cfg_get_env("TOFTWIN_GH_ORDER", ["resolution","gh_order"], 3; parsefn=x->parse(Int,x))
nsigma   = cfg_get_env("TOFTWIN_NSIGMA", ["resolution","nsigma"], 4.0; parsefn=x->parse(Float64,x))
σt_source = lowercase(cfg_get_env("TOFTWIN_SIGMA_T_SOURCE", ["resolution","sigma_t_source"], "pychop"; parsefn=String))
pychop_check = cfg_get_env("TOFTWIN_PYCHOP_CHECK", ["resolution","pychop_check"], false; parsefn=env_bool)

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
if !isabspath(pychop_script)
    pychop_script = joinpath(@__DIR__, pychop_script)
end

pychop_variant = cfg_get_env("TOFTWIN_PYCHOP_VARIANT", ["resolution","pychop","variant"],
    instr === :SEQUOIA ? "SEQ-100-2.0-AST" : "High Flux"; parsefn=String)

pychop_freq_hz = if haskey(ENV, "TOFTWIN_PYCHOP_FREQ_HZ")
    env_float_list(get(ENV, "TOFTWIN_PYCHOP_FREQ_HZ"))
else
    Float64.(cfg_get(CFG, ["resolution","pychop","freq_hz"], instr === :SEQUOIA ? [600.0] : [300.0, 60.0]))
end

pychop_tc_index = cfg_get_env("TOFTWIN_PYCHOP_TC_INDEX", ["resolution","pychop","tc_index"], 0; parsefn=x->parse(Int,x))
pychop_use_tc_rss = cfg_get_env("TOFTWIN_PYCHOP_TC_RSS", ["resolution","pychop","use_tc_rss"], false; parsefn=env_bool)
pychop_delta_td_us = cfg_get_env("TOFTWIN_PYCHOP_DELTA_TD_US", ["resolution","pychop","delta_td_us"], 0.0; parsefn=x->parse(Float64,x))
pychop_npts = cfg_get_env("TOFTWIN_PYCHOP_NPTS", ["resolution","pychop","npts"], 401; parsefn=x->parse(Int,x))
pychop_sigma_q = cfg_get_env("TOFTWIN_PYCHOP_SIGMA_Q", ["resolution","pychop","sigma_q"], 0.25; parsefn=x->parse(Float64,x))

# ntof (controls TOF binning used in coverage point cloud)
ntof_env   = cfg_get_env("TOFTWIN_NTOF", ["ntof","mode"], "auto"; parsefn=String)
ntof_alpha = cfg_get_env("TOFTWIN_NTOF_ALPHA", ["ntof","alpha"], 0.5; parsefn=x->parse(Float64,x))
ntof_beta  = cfg_get_env("TOFTWIN_NTOF_BETA", ["ntof","beta"], 0.333333333333; parsefn=x->parse(Float64,x))
ntof_min   = cfg_get_env("TOFTWIN_NTOF_MIN", ["ntof","min"], 200; parsefn=x->parse(Int,x))
ntof_max   = cfg_get_env("TOFTWIN_NTOF_MAX", ["ntof","max"], 2000; parsefn=x->parse(Int,x))

# cache (optional; helps avoid re-loading geometry repeatedly)
disk_cache  = cfg_get_env("TOFTWIN_DISK_CACHE", ["cache","disk_cache"], true; parsefn=env_bool)
cache_dir   = resolve_path(cfg_get_env("TOFTWIN_CACHE_DIR", ["cache","cache_dir"], ".toftwin_cache"; parsefn=String))
cache_ver   = cfg_get_env("TOFTWIN_CACHE_VERSION", ["cache","cache_version"], "ctx_v2"; parsefn=String)
cache_Cv    = cfg_get_env("TOFTWIN_CACHE_CV", ["cache","cache_Cv"], false; parsefn=env_bool)
cache_rmap  = cfg_get_env("TOFTWIN_CACHE_RMAP", ["cache","cache_rmap"], false; parsefn=env_bool)
disk_cache && mkpath(cache_dir)

# -----------------------------
# Sunny axis suggestion parameters
# -----------------------------
# Defaults are "pretty safe"; adjust in TOML under [sunny_axes]
dq_target = cfg_get_env("TOFTWIN_SUNNY_DQ", ["sunny_axes","dq_target"], 0.01; parsefn=x->parse(Float64,x))
dω_target = cfg_get_env("TOFTWIN_SUNNY_DW", ["sunny_axes","domega_target"], 0.05; parsefn=x->parse(Float64,x))

q_quant = Tuple(Float64.(cfg_get(CFG, ["sunny_axes","q_quant"], [0.01, 0.99])))
w_quant = Tuple(Float64.(cfg_get(CFG, ["sunny_axes","w_quant"], [0.01, 0.99])))

q_pad_frac = cfg_get_env("TOFTWIN_SUNNY_QPAD", ["sunny_axes","q_pad_frac"], 0.05; parsefn=x->parse(Float64,x))
w_pad_frac = cfg_get_env("TOFTWIN_SUNNY_WPAD", ["sunny_axes","w_pad_frac"], 0.05; parsefn=x->parse(Float64,x))

w_min_clip = cfg_get(CFG, ["sunny_axes","w_min_clip"], 0.0)
w_max_clip = cfg_get(CFG, ["sunny_axes","w_max_clip"], nothing)

# If you want to force nQ/nω rather than step-based targets, set these in TOML.
nQ_sunny = cfg_get(CFG, ["sunny_axes","nQ"], nothing)
nω_sunny = cfg_get(CFG, ["sunny_axes","nW"], nothing)

# Optional output
save_axes = cfg_get_env("TOFTWIN_SUNNY_SAVE_AXES", ["sunny_axes","save_axes"], true; parsefn=env_bool)
axes_out  = resolve_path(cfg_get_env("TOFTWIN_SUNNY_AXES_OUT", ["sunny_axes","axes_out"], "sunny_axes_suggestion.jld2"; parsefn=String))

# -----------------------------
# Build PowderCtx (we only need inst/pixels/tof_edges)
# -----------------------------
# setup_powder_ctx requires kernel axes; for this helper we provide a broad, kinematic default.
# Conversion: E[meV] ≈ 2.072 * k^2 [Å^-2]  => k = sqrt(E/2.072) [Å^-1]
ki = sqrt(max(Ei, 1e-6) / 2.072)
kern_Qmin = cfg_get_env("TOFTWIN_KERN_QMIN", ["sunny_axes","kern_Qmin"], 0.0; parsefn=x->parse(Float64,x))
kern_Qmax_default = 2.0 * ki
kern_Qmax = cfg_get_env("TOFTWIN_KERN_QMAX", ["sunny_axes","kern_Qmax"], kern_Qmax_default; parsefn=x->parse(Float64,x))
kern_wmin = cfg_get_env("TOFTWIN_KERN_WMIN", ["sunny_axes","kern_wmin"], 0.0; parsefn=x->parse(Float64,x))
kern_wmax_default = Ei
kern_wmax = cfg_get_env("TOFTWIN_KERN_WMAX", ["sunny_axes","kern_wmax"], kern_wmax_default; parsefn=x->parse(Float64,x))

# Keep histogram sizes modest here; they aren't used by the axis suggestion itself.
nQbins = cfg_get_env("TOFTWIN_NQBINS", ["bins","nQbins"], 200; parsefn=x->parse(Int,x))
nωbins = cfg_get_env("TOFTWIN_NWBINS", ["bins","nWbins"], 200; parsefn=x->parse(Int,x))

@info "Config" cfg=_CFG_PATH instr=instr Ei_meV=Ei idf_path=idf_path grouping=grouping mask=mask_btp

ctx = TOFtwin.setup_powder_ctx(;
    instr=instr,
    idf_path=idf_path,
    kern_Qmin=kern_Qmin,
    kern_Qmax=kern_Qmax,
    kern_wmin=kern_wmin,
    kern_wmax=kern_wmax,
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
    pychop_check_etrans_meV=Float64[],
    ntof_env=ntof_env, ntof_alpha=ntof_alpha, ntof_beta=ntof_beta, ntof_min=ntof_min, ntof_max=ntof_max,
    disk_cache=disk_cache, cache_dir=cache_dir, cache_ver=cache_ver,
    cache_Cv=cache_Cv, cache_rmap=cache_rmap
)

@info "PowderCtx ready" pixels=length(ctx.pixels) ntof=length(ctx.tof_edges_s)-1

# -----------------------------
# Suggest Sunny axes + optional plot
# -----------------------------
# Use TOFtwin's implementation if present; otherwise try to include a local helper file.
_suggest = isdefined(TOFtwin, :suggest_sunny_powder_axes) ? TOFtwin.suggest_sunny_powder_axes : nothing
_cover   = isdefined(TOFtwin, :coverage_points_Qω) ? TOFtwin.coverage_points_Qω : nothing

if _suggest === nothing || _cover === nothing
# Try to find a local implementation (useful during development before it is exported by TOFtwin)
    candidate_paths = [
        joinpath(@__DIR__, "powder_coverage.jl"),
        joinpath(@__DIR__, "..", "src", "powder_coverage.jl"),
        joinpath(@__DIR__, "..", "powder_coverage.jl"),
    ]
    local_path = ""
    for cp in candidate_paths
        if isfile(cp)
            local_path = cp
            break
        end
    end

    if !isempty(local_path)
        @info "TOFtwin coverage helpers not found; including local powder_coverage.jl" path=local_path
        include(local_path)
        _suggest = suggest_sunny_powder_axes
        _cover   = coverage_points_Qω
    else
        error("Need TOFtwin.suggest_sunny_powder_axes / coverage_points_Qω, or a local powder_coverage.jl (checked: $(candidate_paths)).")
    end
end

sug = _suggest(ctx.inst;
    pixels=ctx.pixels,
    Ei_meV=ctx.Ei_meV,
    tof_edges_s=ctx.tof_edges_s,
    nQ=nQ_sunny,
    nω=nω_sunny,
    dq_target=dq_target,
    dω_target=dω_target,
    q_quant=q_quant,
    w_quant=w_quant,
    q_pad_frac=q_pad_frac,
    w_pad_frac=w_pad_frac,
    w_min_clip=w_min_clip,
    w_max_clip=w_max_clip
)

@info "Suggested Sunny axes" Q_range=sug.Q_range ω_range=sug.ω_range nQ=length(sug.radii) nω=length(sug.energies)

if save_axes
    @save axes_out radii=sug.radii energies=sug.energies Q_range=sug.Q_range ω_range=sug.ω_range
    @info "Wrote Sunny axes" axes_out=axes_out
end

if do_plot
    pts = _cover(ctx.inst; pixels=ctx.pixels, Ei_meV=ctx.Ei_meV, tof_edges_s=ctx.tof_edges_s)
    fig = Figure(size=(900,700))
    ax = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)",
              title="Instrument coverage points (pixel × TOF centers)")
    scatter!(ax, pts.Qmag, pts.ω; markersize=1)
    outpng = joinpath(outdir, "coverage_points_Qw.png")
    save(outpng, fig)
    @info "Wrote coverage plot" outpng=outpng
    display(fig)
end

# Convenience: print copy/paste snippet for Sunny scripts
println()
println("# --- Copy/paste into your Sunny powder script ---")
println("radii = collect(range($(sug.Q_range[1]), $(sug.Q_range[2]); length=$(length(sug.radii))))")
println("energies = collect(range($(sug.ω_range[1]), $(sug.ω_range[2]); length=$(length(sug.energies))))")

# -----------------------------------------------------------------------------
# Powder workflow: user-facing config + env parsing + ctx wrapper
#
# Intended location in TOFtwin repo:
#   src/powder_workflow_config.jl
#
# Then in src/TOFtwin.jl (or equivalent):
#   include("powder_workflow_config.jl")
#   export PowderWorkflowConfig, powder_config_from_env
#
# This file is intentionally dependency-light (no Makie, no JLD2).
# -----------------------------------------------------------------------------

# --- tiny parsing helpers ---
_env_str(key::AbstractString, default::AbstractString="") = get(ENV, key, default)

function _env_bool(key::AbstractString, default::Bool=false)
    v = lowercase(strip(get(ENV, key, default ? "1" : "0")))
    return v in ("1","true","yes","y","on")
end

_env_int(key::AbstractString, default::Integer) = parse(Int, get(ENV, key, string(default)))
_env_float(key::AbstractString, default::Real) = parse(Float64, get(ENV, key, string(default)))

function _env_float_list(key::AbstractString, default::AbstractString="")
    s = strip(get(ENV, key, default))
    isempty(s) && return Float64[]
    return parse.(Float64, split(s, r"[^0-9eE+\-.]+"; keepempty=false))
end

function parse_instrument(s::AbstractString)
    u = uppercase(strip(s))
    if u in ("CNCS",)
        return :CNCS
    elseif u in ("SEQUOIA","SEQ")
        return :SEQUOIA
    else
        throw(ArgumentError("TOFTWIN_INSTRUMENT must be CNCS or SEQUOIA (got '$s')"))
    end
end

"""
    PowderWorkflowConfig

A single, user-facing configuration object for building a `PowderCtx` via `setup_powder_ctx`.

This is meant to:
- centralize environment-variable parsing
- provide a stable “surface area” for future analysis workflow APIs
- keep the *existing* `setup_powder_ctx(; kwargs...)` as the low-level primitive

Notes:
- `idf_path` and `pychop_script` can be set via env vars (TOFTWIN_IDF_PATH, TOFTWIN_PYCHOP_SCRIPT)
  or default to files in `base_dir` when created via `powder_config_from_env(base_dir=...)`.
"""
Base.@kwdef struct PowderWorkflowConfig
    # instrument + geometry
    instr::Symbol
    idf_path::String
    Ei_meV::Float64
    rebuild_geom::Bool = false

    # sampling / pixel decimation
    ψstride::Int = 1
    ηstride::Int = 1

    # grouping / masking
    grouping::String = "4x2"
    grouping_file::Union{Nothing,String} = nothing
    mask_btp::String = "Bank=36-50"
    mask_mode::Symbol = :drop
    angle_step_deg::Float64 = 0.5

    # histogram binning
    nQbins::Int = 420
    nωbins::Int = 440

    # resolution settings
    res_mode::String = "cdf"
    σt_us::Float64 = 10.0
    gh_order::Int = 3
    nsigma::Float64 = 4.0

    # timing width source
    σt_source::String = "pychop"  # "manual" | "pychop"

    # pychop oracle settings (used when σt_source="pychop")
    pychop_python::String = Sys.iswindows() ? raw"C:\Users\vdp\AppData\Local\Microsoft\WindowsApps\python3.11.exe" : "python3"
    pychop_script::String
    pychop_variant::String = "SEQ-100-2.0-AST"
    pychop_freq_hz::Vector{Float64} = Float64[600.0]
    pychop_tc_index::Int = 0
    pychop_use_tc_rss::Bool = false
    pychop_delta_td_us::Float64 = 0.0
    pychop_npts::Int = 401
    pychop_sigma_q::Float64 = 0.25

    pychop_check::Bool = true
    pychop_check_etrans_meV::Vector{Float64} = Float64[]

    # adaptive TOF grid knobs
    ntof_env::String = "auto"
    ntof_alpha::Float64 = 0.5
    ntof_beta::Float64 = 1/3
    ntof_min::Int = 200
    ntof_max::Int = 2000

    # caching (Cv + reduce-map only)
    disk_cache::Bool = true
    cache_dir::String
    cache_ver::String = "ctx_v2"
    cache_Cv::Bool = false
    cache_rmap::Bool = false
end

"""
    powder_config_from_env(; base_dir="")

Create a `PowderWorkflowConfig` by reading the same env vars used by the current powder demo script.

If `base_dir` is provided, it is used to resolve defaults for:
- `idf_path` (CNCS_Definition_2025B.xml / SEQUOIA_Definition.xml)
- `pychop_script` (pychop_oracle_dE.py)
- `cache_dir` (.toftwin_cache)

Env vars (subset; matches demo):
- TOFTWIN_INSTRUMENT, TOFTWIN_IDF_PATH, TOFTWIN_EI, TOFTWIN_REBUILD_GEOM
- TOFTWIN_NQBINS, TOFTWIN_NWBINS, TOFTWIN_PSI_STRIDE, TOFTWIN_ETA_STRIDE
- TOFTWIN_GROUPING, TOFTWIN_GROUPING_FILE, TOFTWIN_MASK_BTP, TOFTWIN_MASK_MODE, TOFTWIN_POWDER_ANGLESTEP
- TOFTWIN_RES_MODE, TOFTWIN_SIGMA_T_US, TOFTWIN_GH_ORDER, TOFTWIN_NSIGMA, TOFTWIN_SIGMA_T_SOURCE
- TOFTWIN_PYCHOP_* (PYTHON, SCRIPT, VARIANT, FREQ_HZ, TC_INDEX, TC_RSS, DELTA_TD_US, NPTS, SIGMA_Q, CHECK)
- TOFTWIN_NTOF* (NTOF, NTOF_ALPHA, NTOF_BETA, NTOF_MIN, NTOF_MAX)
- TOFTWIN_DISK_CACHE, TOFTWIN_CACHE_DIR, TOFTWIN_CACHE_VERSION, TOFTWIN_CACHE_CV, TOFTWIN_CACHE_RMAP
"""
function powder_config_from_env(; base_dir::AbstractString="")
    instr = parse_instrument(_env_str("TOFTWIN_INSTRUMENT", "SEQUOIA"))

    # Ei default depends on instrument (match demo behavior)
    default_Ei = instr === :SEQUOIA ? 30.0 : 12.0
    Ei = _env_float("TOFTWIN_EI", default_Ei)

    # IDF path default mirrors demo script
    idf_default = if isempty(base_dir)
        ""
    elseif instr === :CNCS
        joinpath(base_dir, "CNCS_Definition_2025B.xml")
    else
        joinpath(base_dir, "SEQUOIA_Definition.xml")
    end
    idf_path = strip(_env_str("TOFTWIN_IDF_PATH", idf_default))
    isempty(idf_path) && throw(ArgumentError("Missing IDF path. Set TOFTWIN_IDF_PATH or pass base_dir to powder_config_from_env."))

    rebuild_geom = _env_bool("TOFTWIN_REBUILD_GEOM", false)

    # grouping/masking
    grouping      = strip(_env_str("TOFTWIN_GROUPING", "4x2"))
    grouping_file = strip(_env_str("TOFTWIN_GROUPING_FILE", ""))
    grouping_file = isempty(grouping_file) ? nothing : grouping_file
    mask_btp      = _env_str("TOFTWIN_MASK_BTP", "Bank=36-50")
    mask_mode     = Symbol(lowercase(_env_str("TOFTWIN_MASK_MODE", "drop")))
    angle_step    = _env_float("TOFTWIN_POWDER_ANGLESTEP", 0.5)

    # binning + strides
    nQbins   = _env_int("TOFTWIN_NQBINS", 420)
    nωbins   = _env_int("TOFTWIN_NWBINS", 440)
    ψstride  = _env_int("TOFTWIN_PSI_STRIDE", 1)
    ηstride  = _env_int("TOFTWIN_ETA_STRIDE", 1)

    # resolution knobs
    res_mode = lowercase(_env_str("TOFTWIN_RES_MODE", "cdf"))
    σt_us    = _env_float("TOFTWIN_SIGMA_T_US", 10.0)
    gh_order = _env_int("TOFTWIN_GH_ORDER", 3)
    nsigma   = _env_float("TOFTWIN_NSIGMA", 4.0)
    σt_source = lowercase(_env_str("TOFTWIN_SIGMA_T_SOURCE", "pychop"))

    # pychop
    pychop_python = _env_str("TOFTWIN_PYCHOP_PYTHON",
                      _env_str("PYTHON", Sys.iswindows() ? raw"C:\Users\vdp\AppData\Local\Microsoft\WindowsApps\python3.11.exe" : "python3"))
    pychop_script_default = isempty(base_dir) ? "" : joinpath(base_dir, "pychop_oracle_dE.py")
    pychop_script = _env_str("TOFTWIN_PYCHOP_SCRIPT", pychop_script_default)
    pychop_variant = _env_str("TOFTWIN_PYCHOP_VARIANT", "SEQ-100-2.0-AST")
    pychop_freq_hz = _env_float_list("TOFTWIN_PYCHOP_FREQ_HZ", "600")
    pychop_tc_index = _env_int("TOFTWIN_PYCHOP_TC_INDEX", 0)
    pychop_use_tc_rss = _env_bool("TOFTWIN_PYCHOP_TC_RSS", false)
    pychop_delta_td_us = _env_float("TOFTWIN_PYCHOP_DELTA_TD_US", 0.0)
    pychop_npts = _env_int("TOFTWIN_PYCHOP_NPTS", 401)
    pychop_sigma_q = _env_float("TOFTWIN_PYCHOP_SIGMA_Q", 0.25)
    pychop_check = _env_bool("TOFTWIN_PYCHOP_CHECK", true)

    # adaptive tof
    ntof_env   = _env_str("TOFTWIN_NTOF", "auto")
    ntof_alpha = _env_float("TOFTWIN_NTOF_ALPHA", 0.5)
    ntof_beta  = _env_float("TOFTWIN_NTOF_BETA", 1/3)
    ntof_min   = _env_int("TOFTWIN_NTOF_MIN", 200)
    ntof_max   = _env_int("TOFTWIN_NTOF_MAX", 2000)

    # caching
    disk_cache  = _env_bool("TOFTWIN_DISK_CACHE", true)
    cache_dir_default = isempty(base_dir) ? ".toftwin_cache" : joinpath(base_dir, ".toftwin_cache")
    cache_dir   = _env_str("TOFTWIN_CACHE_DIR", cache_dir_default)
    cache_ver   = _env_str("TOFTWIN_CACHE_VERSION", "ctx_v2")
    cache_Cv    = _env_bool("TOFTWIN_CACHE_CV", false)
    cache_rmap  = _env_bool("TOFTWIN_CACHE_RMAP", false)

    return PowderWorkflowConfig(;
        instr=instr,
        idf_path=idf_path,
        Ei_meV=Ei,
        rebuild_geom=rebuild_geom,
        ψstride=ψstride, ηstride=ηstride,
        grouping=grouping, grouping_file=grouping_file,
        mask_btp=mask_btp, mask_mode=mask_mode,
        angle_step_deg=angle_step,
        nQbins=nQbins, nωbins=nωbins,
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
        cache_Cv=cache_Cv, cache_rmap=cache_rmap,
    )
end

"""
    setup_powder_ctx(cfg::PowderWorkflowConfig; kern_Qmin, kern_Qmax, kern_wmin, kern_wmax)

Convenience wrapper that forwards `cfg` into the existing low-level `setup_powder_ctx(; kwargs...)`.
"""
function setup_powder_ctx(cfg::PowderWorkflowConfig; kern_Qmin, kern_Qmax, kern_wmin, kern_wmax)
    return setup_powder_ctx(;
        instr=cfg.instr,
        idf_path=cfg.idf_path,
        kern_Qmin=kern_Qmin,
        kern_Qmax=kern_Qmax,
        kern_wmin=kern_wmin,
        kern_wmax=kern_wmax,
        rebuild_geom=cfg.rebuild_geom,
        ψstride=cfg.ψstride, ηstride=cfg.ηstride,
        grouping=cfg.grouping,
        grouping_file=cfg.grouping_file,
        mask_btp=cfg.mask_btp,
        mask_mode=cfg.mask_mode,
        angle_step=cfg.angle_step_deg,
        Ei=cfg.Ei_meV, nQbins=cfg.nQbins, nωbins=cfg.nωbins,
        res_mode=cfg.res_mode, σt_us=cfg.σt_us, gh_order=cfg.gh_order, nsigma=cfg.nsigma,
        σt_source=cfg.σt_source,
        pychop_python=cfg.pychop_python,
        pychop_script=cfg.pychop_script,
        pychop_variant=cfg.pychop_variant,
        pychop_freq_hz=cfg.pychop_freq_hz,
        pychop_tc_index=cfg.pychop_tc_index,
        pychop_use_tc_rss=cfg.pychop_use_tc_rss,
        pychop_delta_td_us=cfg.pychop_delta_td_us,
        pychop_npts=cfg.pychop_npts,
        pychop_sigma_q=cfg.pychop_sigma_q,
        pychop_check=cfg.pychop_check,
        pychop_check_etrans_meV=cfg.pychop_check_etrans_meV,
        ntof_env=cfg.ntof_env, ntof_alpha=cfg.ntof_alpha, ntof_beta=cfg.ntof_beta,
        ntof_min=cfg.ntof_min, ntof_max=cfg.ntof_max,
        disk_cache=cfg.disk_cache, cache_dir=cfg.cache_dir, cache_ver=cfg.cache_ver,
        cache_Cv=cfg.cache_Cv, cache_rmap=cfg.cache_rmap,
    )
end

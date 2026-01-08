using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using TOFtwin
using JLD2

# -----------------------------------------------------------------------------
# Demo (refactored): user-friendly powder workflow entrypoint
#
# Intended location in TOFtwin repo:
#   scripts/demo_powder_workflow.jl
#
# This demo stays focused on:
#   1) choose a kernel (Sunny JLD2 or artificial delta-comb)
#   2) build ctx using PowderWorkflowConfig_from_env
#   3) eval_powder once (or over multiple kernels)
#
# Plotting is optional (TOFTWIN_DO_HIST=1) and lives here (not in package core).
# -----------------------------------------------------------------------------

# -----------------------------
# Optional plotting backend
# -----------------------------
const do_hist  = lowercase(get(ENV, "TOFTWIN_DO_HIST", "1")) in ("1","true","yes")
const do_save  = lowercase(get(ENV, "TOFTWIN_SAVE", "1")) in ("1","true","yes")
const outdir   = get(ENV, "TOFTWIN_OUTDIR", joinpath(@__DIR__, "out"))
do_save && mkpath(outdir)

const backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
if do_hist
    if backend == "gl"
        using GLMakie
    else
        using CairoMakie
    end
end

# -----------------------------
# Sunny loader (still script-local for now)
# -----------------------------
function load_sunny_powder_jld2(path::AbstractString; outside=0.0)
    radii = energies = S_Qω = nothing
    @load path radii energies S_Qω

    Q = collect(radii)
    W = collect(energies)
    S = Matrix(S_Qω)

    @info "Loaded Sunny table" sizeS=size(S) lenQ=length(Q) lenW=length(W)

    # Sunny often stores as (ω, Q). Fix to (Q, ω).
    if size(S) == (length(W), length(Q))
        @info "Transposing Sunny table (ω,Q) -> (Q,ω)"
        S = permutedims(S)
    end
    @assert size(S) == (length(Q), length(W))

    return TOFtwin.GridKernelPowder(Q, W, S; outside=outside)
end

# -----------------------------
# Artificial "delta-comb" kernel
# -----------------------------
function _delta_comb_value(q, w; dQ=0.5, dW=0.5, σQ=0.03, σW=0.03, amp=1.0)
    (q < 0 || w < 0) && return 0.0
    dq = abs(q - round(q/dQ) * dQ)
    dw = abs(w - round(w/dW) * dW)
    return amp * exp(-0.5*(dq/σQ)^2 - 0.5*(dw/σW)^2)
end

function make_delta_comb_kernel(; Qmin=0.0, Qmax=6.0, wmin=0.0, wmax=25.0,
                                dQ=0.5, dW=0.5, σQ=0.03, σW=0.03,
                                fine_dQ=0.025, fine_dW=0.05, amp=1.0,
                                outside=0.0)
    Q = collect(range(Float64(Qmin), Float64(Qmax); step=Float64(fine_dQ)))
    W = collect(range(Float64(wmin), Float64(wmax); step=Float64(fine_dW)))
    S = [ _delta_comb_value(q, w; dQ=dQ, dW=dW, σQ=σQ, σW=σW, amp=amp) for q in Q, w in W ]
    return TOFtwin.GridKernelPowder(Q, W, S; outside=outside)
end

# -----------------------------
# Plot helper
# -----------------------------
function plot_result(ctx::TOFtwin.PowderCtx, kern::TOFtwin.GridKernelPowder, pred; tag::AbstractString="")
    Q_cent = 0.5 .* (ctx.Q_edges[1:end-1] .+ ctx.Q_edges[2:end])
    ω_cent = 0.5 .* (ctx.ω_edges[1:end-1] .+ ctx.ω_edges[2:end])

    kernel_grid = [kern(q,w) for q in Q_cent, w in ω_cent]
    wt_log = log10.(pred.Hwt.counts .+ 1.0)

    fig = Figure(size=(1400, 900))

    ax1 = Axis(fig[1,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="Kernel S(|Q|,ω)")
    heatmap!(ax1, Q_cent, ω_cent, kernel_grid)

    ax2 = Axis(fig[1,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="TOFtwin predicted (raw SUM)")
    heatmap!(ax2, Q_cent, ω_cent, pred.Hraw.counts)

    ax3 = Axis(fig[2,1], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="weights log10(N+1)")
    heatmap!(ax3, Q_cent, ω_cent, wt_log)

    ax4 = Axis(fig[2,2], xlabel="|Q| (Å⁻¹)", ylabel="ω (meV)", title="vanadium-normalized MEAN")
    heatmap!(ax4, Q_cent, ω_cent, pred.Hmean.counts)

    if do_save
        fname = isempty(tag) ? "demo_powder_$(lowercase(String(ctx.instr))).png" :
                               "demo_powder_$(lowercase(String(ctx.instr)))_$(tag).png"
        outpng = joinpath(outdir, fname)
        save(outpng, fig)
        @info "Wrote $outpng"
    end
    display(fig)
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
kernel_kind = lowercase(strip(get(ENV, "TOFTWIN_KERNEL", "delta")))

# Build cfg from env (requires that TOFtwin includes/export powder_workflow_config.jl)
cfg = TOFtwin.powder_config_from_env(base_dir=@__DIR__)

# Choose a kernel (only used for ctx axes + optional plotting)
sunny_paths = String[]
kern0 = nothing

if kernel_kind == "sunny"
    sunny_paths = length(ARGS) > 0 ? ARGS : [joinpath(@__DIR__, "..", "sunny_powder_corh2o4.jld2")]
    kern0 = load_sunny_powder_jld2(sunny_paths[1]; outside=0.0)
elseif kernel_kind in ("delta", "comb", "delta-comb", "deltacomb")
    Qmin = parse(Float64, get(ENV, "TOFTWIN_KERN_QMIN", "0.0"))
    Qmax = parse(Float64, get(ENV, "TOFTWIN_KERN_QMAX", "6.0"))
    wmin = parse(Float64, get(ENV, "TOFTWIN_KERN_WMIN", "0.0"))
    wmax = parse(Float64, get(ENV, "TOFTWIN_KERN_WMAX", string(cfg.Ei_meV)))

    dQ  = parse(Float64, get(ENV, "TOFTWIN_DELTA_DQ", "0.5"))
    dW  = parse(Float64, get(ENV, "TOFTWIN_DELTA_DW", "1.0"))
    σQ  = parse(Float64, get(ENV, "TOFTWIN_DELTA_SIGMA_Q", "0.03"))
    σW  = parse(Float64, get(ENV, "TOFTWIN_DELTA_SIGMA_W", "0.03"))
    fdQ = parse(Float64, get(ENV, "TOFTWIN_DELTA_FINE_DQ", "0.025"))
    fdW = parse(Float64, get(ENV, "TOFTWIN_DELTA_FINE_DW", "0.05"))
    amp = parse(Float64, get(ENV, "TOFTWIN_DELTA_AMP", "1.0"))

    @info "Using artificial delta-comb kernel" Qmin=Qmin Qmax=Qmax wmin=wmin wmax=wmax dQ=dQ dW=dW σQ=σQ σW=σW fine_dQ=fdQ fine_dW=fdW amp=amp

    kern0 = make_delta_comb_kernel(Qmin=Qmin, Qmax=Qmax, wmin=wmin, wmax=wmax,
                                   dQ=dQ, dW=dW, σQ=σQ, σW=σW,
                                   fine_dQ=fdQ, fine_dW=fdW,
                                   amp=amp, outside=0.0)
else
    throw(ArgumentError("TOFTWIN_KERNEL must be 'sunny' or 'delta' (got '$kernel_kind')"))
end

# Build ctx using the cfg wrapper
ctx = TOFtwin.setup_powder_ctx(cfg;
    kern_Qmin=kern0.Q[1], kern_Qmax=kern0.Q[end],
    kern_wmin=kern0.ω[1], kern_wmax=kern0.ω[end],
)

@info "SETUP complete" instr=ctx.instr Ei_meV=ctx.Ei_meV npix=length(ctx.pixels) grouping=cfg.grouping mask=cfg.mask_btp res_mode=cfg.res_mode

# Evaluate
if kernel_kind == "sunny"
    for (i, spath) in enumerate(sunny_paths)
        kern = (i == 1) ? kern0 : load_sunny_powder_jld2(spath; outside=0.0)
        model = (q,w) -> kern(q,w)

        tag = "k$(i)"
        @info "EVALUATE kernel" i=i n=length(sunny_paths) path=spath
        pred = TOFtwin.eval_powder(ctx, model; do_hist=do_hist)

        do_hist && plot_result(ctx, kern, pred; tag=tag)
    end
else
    kern = kern0
    model = (q,w) -> kern(q,w)
    pred = TOFtwin.eval_powder(ctx, model; do_hist=do_hist)

    do_hist && plot_result(ctx, kern, pred; tag="delta")
end

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using TOFtwin
using LinearAlgebra
using Base.Threads

# -----------------------------------------------------------------------------
# Demo: workflow-style single-crystal evaluation
#
#   ctx = setup_singlecrystal_ctx(...)
#   pred = eval_singlecrystal!(ctx, model_hkl)
# -----------------------------------------------------------------------------

# plotting backend (optional)
do_plot = lowercase(get(ENV, "TOFTWIN_PLOT", "true")) in ("1","true","yes","y")
backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))

if do_plot
    if backend == "gl"
        using GLMakie
    else
        using CairoMakie
    end
end

# -----------------------------------------------------------------------------
# Geometry: use a synthetic detector bank (so this demo runs without an IDF).
# Swap this for your real IDF-loaded instrument once wiring is in place.
# -----------------------------------------------------------------------------
bank = bank_from_coverage(name="example", L2=3.5, surface=:cylinder)
inst = Instrument(name="demo", L1=36.262, pixels=bank.pixels)

# Use a sparse subset for speed in the demo
pix_used = sample_pixels(bank, AngularDecimate(3, 2))
@info "pixels used" npix=length(pix_used)

# -----------------------------------------------------------------------------
# Scan definition: simple 1-axis goniometer about y with a small scan range
# -----------------------------------------------------------------------------
g = Goniometer(axis=:y, zero_offset_deg=0.0)
scan = R_SL_scan(g; start_deg=-20, stop_deg=20, step_deg=1.0)
R_SL_list = scan.R_SL_list
@info "orientations" norient=length(R_SL_list)

# -----------------------------------------------------------------------------
# Lattice + alignment: choose u,v in HKL and build a SampleAlignment
# -----------------------------------------------------------------------------
# Example reciprocal lattice (cubic). Replace with your real lattice.
recip = ReciprocalLattice(a=3.0, b=3.0, c=3.0, α=90.0, β=90.0, γ=90.0)

u_hkl = Vec3(1.0, 0.0, 0.0)
v_hkl = Vec3(0.0, 1.0, 0.0)
aln = alignment_from_uv(recip; u_hkl=u_hkl, v_hkl=v_hkl)

# -----------------------------------------------------------------------------
# Binning (H, ω) and TOF
# -----------------------------------------------------------------------------
Ei = 30.0
tof_edges = collect(range(0.001, 0.020; length=700))  # seconds
H_edges  = collect(range(-2.0, 2.0; length=401))      # r.l.u.
ω_edges  = collect(range(0.0, Ei; length=501))        # meV

K_center, K_hw = 0.0, 0.10
L_center, L_hw = 0.0, 0.10

# -----------------------------------------------------------------------------
# Kernel model in HKL space: toy Gaussian “mode”
# -----------------------------------------------------------------------------
function toy_model_hkl(hkl, ω)
    H, K, L = hkl
    # peak near (H=1,K=0,L=0, ω=10 meV)
    return exp(-((H - 1.0)^2)/(2*0.08^2)) *
           exp(-((K - 0.0)^2)/(2*0.08^2)) *
           exp(-((L - 0.0)^2)/(2*0.08^2)) *
           exp(-((ω - 10.0)^2)/(2*1.5^2))
end

# -----------------------------------------------------------------------------
# Setup + evaluate (this is the pattern we want to mirror from powders)
# -----------------------------------------------------------------------------
ctx = setup_singlecrystal_ctx(;
    inst=inst,
    pixels=pix_used,
    Ei_meV=Ei,
    tof_edges_s=tof_edges,
    H_edges=H_edges,
    ω_edges=ω_edges,
    aln=aln,
    R_SL_list=R_SL_list,
    K_center=K_center,
    K_halfwidth=K_hw,
    L_center=L_center,
    L_halfwidth=L_hw,
)

t_setup = @elapsed begin
    # (setup is already done above; keep this var if you want to time setup)
end

t_eval1 = @elapsed pred = eval_singlecrystal!(ctx, toy_model_hkl)
t_eval2 = @elapsed pred2 = eval_singlecrystal!(ctx, toy_model_hkl)  # same model; should reuse preallocs

@info "timings (s)" setup=t_setup eval1=t_eval1 eval2=t_eval2

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
if do_plot
    H_cent = 0.5 .* (H_edges[1:end-1] .+ H_edges[2:end])
    ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

    fig = Figure(size=(1100, 500))
    ax1 = Axis(fig[1,1], xlabel="H (r.l.u.)", ylabel="ω (meV)", title="I(H,ω) weighted mean")
    heatmap!(ax1, H_cent, ω_cent, pred.Hmean.counts)

    ax2 = Axis(fig[1,2], xlabel="H (r.l.u.)", ylabel="ω (meV)", title="coverage weights log10(w+1)")
    heatmap!(ax2, H_cent, ω_cent, log10.(pred.Hwt.counts .+ 1.0))

    mkpath("out")
    save("out/demo_singlecrystal_from_ctx.png", fig)
    display(fig)
end

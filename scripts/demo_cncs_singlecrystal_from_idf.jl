using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin
using LinearAlgebra
using Statistics
using Base.Threads

backend = lowercase(get(ENV, "TOFTWIN_BACKEND", "gl"))
if backend == "gl"
    using GLMakie
else
    using CairoMakie
end

const TVec3 = TOFtwin.Vec3

idf_path = joinpath(@__DIR__, "CNCS_Definition_2025B.xml")

# --- Load CNCS geometry from IDF ---
geo = TOFtwin.MantidIDF.load_mantid_idf_diskcached(idf_path; ψbins=720, ηbins=256)
inst = geo.inst
pixels_all = geo.pixels
@info "CNCS loaded" geo.meta

# Decimate aggressively to keep it fast while iterating
pix_used = TOFtwin.decimate_pixels_angular(pixels_all; ψstride=1, ηstride=1)
@info "pixels used = $(length(pix_used)) / $(length(pixels_all))"
@info "Threads = $(nthreads())"

# --- Ei / TOF bins ---
Ei = 12.0
L2min, L2max = minimum(inst.L2), maximum(inst.L2)
tmin = TOFtwin.tof_from_EiEf(inst.L1, L2min, Ei, Ei)
tmax = TOFtwin.tof_from_EiEf(inst.L1, L2max, Ei, 1.0)
tof_edges = collect(range(tmin, tmax; length=600+1))
@info "TOF window (ms) = $(1e3*tmin) .. $(1e3*tmax)"

# --- Lattice / alignment ---
lp = TOFtwin.LatticeParams(8.5031, 8.5031, 8.5031, 90.0, 90.0, 90.0)
recip = TOFtwin.reciprocal_lattice(lp)

u_hkl = TVec3(0.5, 0.5, 0.1)
v_hkl = TVec3(0.0, 1.0, -0.1)
aln = TOFtwin.alignment_from_uv(recip; u_hkl=u_hkl, v_hkl=v_hkl)

# --- Goniometer scan ---
angles_deg = -180:1.0:180
angles = deg2rad.(collect(angles_deg))
zero_offset_deg = 0.0
R_SL_list = TOFtwin.goniometer_scan_RSL_y(angles; zero_offset_rad=deg2rad(zero_offset_deg))

# --- Cut definition ---
H_edges = collect(range(-2.0, 2.0; length=260))
ω_edges = collect(range(-2.0, Ei; length=260))
H_cent = 0.5 .* (H_edges[1:end-1] .+ H_edges[2:end])
ω_cent = 0.5 .* (ω_edges[1:end-1] .+ ω_edges[2:end])

Kc, Kh = 0.0, 0.10
Lc, Lh = 0.0, 0.10

# --- Model (use your existing ToyCosineHKL callable) ---
model = TOFtwin.ToyCosineHKL(; q0=TVec3(1,0,0), Δ=2.0, Jh=3.0, Jk=1.5, Jl=0.8, σE=0.25, σQ=0.35, amp=1.0)

@info "Scanning CNCS WITH u,v alignment..."
pred = TOFtwin.predict_cut_mean_Hω_hkl_scan_aligned(inst;
    pixels=pix_used, Ei_meV=Ei, tof_edges_s=tof_edges,
    H_edges=H_edges, ω_edges=ω_edges,
    aln=aln, R_SL_list=R_SL_list, model_hkl=model,
    K_center=Kc, K_halfwidth=Kh, L_center=Lc, L_halfwidth=Lh,
    threaded=true
)

# --- Plot ---
fig = Figure(size=(1100, 800))
ax1 = Axis(fig[1,1], xlabel="H (r.l.u.)", ylabel="ω (meV)",
           title="CNCS IDF: aligned weighted mean   (zero_offset=$(zero_offset_deg)°)")
heatmap!(ax1, H_cent, ω_cent, pred.Hmean.counts)

ax2 = Axis(fig[1,2], xlabel="H (r.l.u.)", ylabel="ω (meV)",
           title="CNCS IDF: coverage weights log10(w+1)")
heatmap!(ax2, H_cent, ω_cent, log10.(pred.Hwt.counts .+ 1.0))

mkpath("out")
save("out/demo_cncs_singlecrystal_from_idf.png", fig)
display(fig)

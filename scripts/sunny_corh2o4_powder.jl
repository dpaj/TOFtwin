#!/usr/bin/env julia
# Powder LSWT for CoRh2O4, based on Sunny tutorial:
# https://sunnysuite.github.io/Sunny.jl/stable/examples/01_LSWT_CoRh2O4.html

using Sunny
using LinearAlgebra
using JLD2

# ---------------- Crystal (conventional cubic cell) ----------------
a = 8.5031 # Å  (tutorial value) :contentReference[oaicite:1]{index=1}
latvecs = lattice_vectors(a, a, a, 90, 90, 90)

# Co belongs to Wyckoff 8a; tutorial uses just one representative position :contentReference[oaicite:2]{index=2}
positions = [[1/8, 1/8, 1/8]]
cryst = Crystal(latvecs, positions, 227; types=["Co"])  # spacegroup 227 (Fd-3m) :contentReference[oaicite:3]{index=3}

# ---------------- Spin system + exchange ----------------
# Co2+ : s=3/2, g=2 (tutorial) :contentReference[oaicite:4]{index=4}
sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole)

# AF nearest-neighbor exchange J=0.63 meV; bond (2,3,[0,0,0]) per tutorial :contentReference[oaicite:5]{index=5}
J = +0.63
set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))

# ---------------- Ground state ----------------
randomize_spins!(sys)
minimize_energy!(sys)

# ---------------- Primitive magnetic cell ----------------
shape = primitive_cell(cryst)                  # tutorial step :contentReference[oaicite:6]{index=6}
sys_prim = reshape_supercell(sys, shape)

# ---------------- Spin-wave theory measurement ----------------
formfactors = [1 => FormFactor("Co2")]         # tutorial step :contentReference[oaicite:7]{index=7}
measure = ssf_perp(sys_prim; formfactors)
swt = SpinWaveTheory(sys_prim; measure)

# Intrinsic broadening kernel (tutorial uses Lorentzian fwhm=0.8 meV) :contentReference[oaicite:8]{index=8}
kernel = lorentzian(fwhm=0.8)

# ---------------- Powder grid to compute ----------------
# You can overwrite these from TOFtwin-suggested ranges later; for now pick something reasonable.
Qmin, Qmax = 0.0, 5.0
ωmin, ωmax = 0.0, 12.0

nQ = 300          # number of |Q| radii
nω = 350          # number of energy points
radii   = collect(range(Qmin, Qmax; length=nQ))
energies = collect(range(ωmin, ωmax; length=nω))

# Powder average sampling per spherical shell (tutorial uses 2000) :contentReference[oaicite:9]{index=9}
nsamp = 2000

# ---------------- Compute powder average ----------------
res_pow = powder_average(cryst, radii, nsamp) do qs
    intensities(swt, qs; energies, kernel)
end

# ---------------- Extract numeric matrix robustly ----------------
function intensities_matrix(res)
    # Many Sunny "intensities-like" objects act like arrays; try that first.
    try
        return Array(res)
    catch
    end
    # Fallback: try common field names.
    for nm in (:data, :I, :intensities, :vals, :S)
        if hasproperty(res, nm)
            return Array(getproperty(res, nm))
        end
    end
    error("Can't extract matrix from $(typeof(res)); fields=$(fieldnames(typeof(res)))")
end

S_Qω = intensities_matrix(res_pow)

@info "powder table size = $(size(S_Qω)) (expect nQ×nω = $nQ×$nω)"
@info "Q range = ($(first(radii)), $(last(radii))) Å^-1"
@info "ω range = ($(first(energies)), $(last(energies))) meV"
@info "nsamp per radius = $nsamp"

# ---------------- Save for TOFtwin ----------------
out = "sunny_powder_corh2o4.jld2"
@save out radii energies S_Qω
@info "wrote $out"

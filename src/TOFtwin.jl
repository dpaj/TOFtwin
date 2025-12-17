module TOFtwin

include("detectors_coverage.jl")
include("detectors_sampling.jl")

include("frames.jl")
include("pose.jl")

include("kinematics.jl")

include("instrument.jl")
include("events.jl")
include("histograms.jl")
include("forward_predict.jl")

include("models_toy.jl")

include("powder_coverage.jl")

include("lattice.jl")
include("single_crystal.jl")


export Vec3, DetectorPixel, pixels_from_coverage, summarize_pixels
export DetectorBank, bank_from_coverage
export PixelSampler, AllPixels, RandomSubset, Stride, StratifiedByEta, AngularDecimate, ByBank
export sample_pixels

export Rigid, apply_point, apply_vec, inv, rigid, Rx, Ry, Rz
export Pose, T_LS, T_SL, sample_orientation_euler

export k_from_EmeV, EmeV_from_k, v_from_EmeV, tof_from_EiEf, Ef_from_tof, Qω_from_pixel
export dω_dt
export Instrument, pixel, L2
export Event, Qω_from_event
export Hist2D, hist_Qω_powder, hist_pixel_tof

export predict_hist_Qω_powder
export predict_pixel_tof, normalize_by_vanadium, reduce_pixel_tof_to_Qω_powder

export ToyModePowder, ToyGaussianDispHKL, ToyCosineHKL

export predict_powder_mean_Qω

export coverage_points_Qω, suggest_sunny_powder_axes
export GridKernelPowder, kernel_from_function

export LatticeParams, ReciprocalLattice, reciprocal_lattice, hkl_from_Q
export reduce_pixel_tof_to_Hω_cut, predict_cut_mean_Hω

export Q_L_from_pixel, Q_S_from_pixel

export predict_cut_weightedmean_Hω_hkl_scan

end

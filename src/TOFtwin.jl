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

export Vec3, DetectorPixel, pixels_from_coverage, summarize_pixels
export DetectorBank, bank_from_coverage
export PixelSampler, AllPixels, RandomSubset, Stride, StratifiedByEta, AngularDecimate, ByBank
export sample_pixels

export Rigid, apply_point, apply_vec, inv, rigid, Rx, Ry, Rz
export Pose, T_LS, T_SL, sample_orientation_euler

export k_from_EmeV, EmeV_from_k, v_from_EmeV, tof_from_EiEf, Ef_from_tof, Qω_from_pixel
export Instrument, pixel, L2
export Event, Qω_from_event
export Hist2D, hist_Qω_powder, hist_pixel_tof

export predict_hist_Qω_powder

export ToyModePowder

end

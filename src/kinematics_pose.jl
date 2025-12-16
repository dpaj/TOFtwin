
"""
Compute Q in sample frame using a Pose.

Assumes sample position is at origin in sample frame; you can add sample offsets later.
"""
function Q_S_from_pixel(r_pix_L::Vec3, Ei::Float64, Ef::Float64, pose::Pose)
    Q_L = Q_L_from_pixel(r_pix_L, Ei, Ef)
    # Q is a vector, so transform with rotation only:
    R_SL = T_SL(pose).R
    return R_SL * Q_L
end

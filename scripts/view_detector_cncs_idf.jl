using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using TOFtwin

CFG = Dict(
    :backend   => :gl,   # :gl (interactive) or :cairo (static)
    :idf_src   => joinpath(@__DIR__, "CNCS_Definition_2025B.xml"),

    :views     => (:xz, :xyz),
    :layout    => :h,    # :h (side-by-side) or :v (stacked)

    :title     => "CNCS from IDF",
    :bank_regex=> nothing,
    :ψstride   => 1,
    :ηstride   => 1,

    :cached    => true,
    :rebuild   => false,

    # XZ handedness (set true if you want +y toward the viewer)
    :xz_y_toward_viewer => true,

    # Camera controls (only affects :xyz)
    :camera        => :ki,        # :ki or :default (skip)
    :cam_ki        => (0, 0, 1),   # +z (forward)
    :cam_up        => (0, 1, 0),   # +y up
    :cam_dist      => nothing,     # set Float64 to override
    :cam_side_frac => 0.22,
    :cam_up_frac   => 0.12,

    :fig_size  => (1500, 750),
    :ms_2d     => 2.0,
    :ms_3d     => 2.0,

    :out_path  => "out/view_detector_cncs_views.png",
    :save      => true,
)

if CFG[:backend] == :cairo
    using CairoMakie
else
    using GLMakie
end

include(joinpath(@__DIR__, "view_detector_idf_common.jl"))

fig = view_detector_idf(; cfg=CFG)
display(fig)

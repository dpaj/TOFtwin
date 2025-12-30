using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using TOFtwin

CFG = Dict(
    :backend   => :gl,
    :idf_src   => :SEQUOIA,

    :views     => (:xz, :xyz),
    :layout    => :h,

    :title     => "SEQUOIA from IDF",
    :bank_regex=> nothing,

    :ψstride   => 1,
    :ηstride   => 1,

    # NEW: grouping/masking
    :grouping  => "4x2",
    :grouping_file => nothing,
    :mask_btp  => "",
    :mask_mode => :drop,
    :powder_angle_step => 0.5,
    :outdir    => "out",

    :cached    => true,
    :rebuild   => false,

    :xz_y_toward_viewer => true,

    :camera        => :ki,
    :cam_ki        => (0, 0, 1),
    :cam_up        => (0, 1, 0),
    :cam_dist      => nothing,
    :cam_side_frac => 0.18,
    :cam_up_frac   => 0.10,

    :fig_size  => (1500, 750),
    :ms_2d     => 1.2,
    :ms_3d     => 1.2,

    :out_path  => "out/view_detector_sequoia_views.png",
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

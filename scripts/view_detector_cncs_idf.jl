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

    # NEW: grouping/masking (Mantid-like)
    :grouping  => "8x2",          # "" | "2x1" | "4x1" | "8x1" | "4x2" | "8x2" | "powder"
    :grouping_file => nothing, # override with explicit path if desired
    :mask_btp  => "Bank=36-50;Mode=drop",          # e.g. "Bank=40-50;Mode=drop" or Dict(...)
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

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using TOFtwin

# Simple detector-view script for CNCS Mantid IDF.
# Produces a 2D X–Z projection (lab frame) and saves a PNG.

CFG = Dict(
    :backend    => :gl,   # :gl or :cairo
    :idf_path   => joinpath(@__DIR__, "CNCS_Definition_2025B.xml"),
    :bank_regex => nothing,  # e.g. r"bank29" to focus
    :ψstride    => 1,
    :ηstride    => 1,
    :out_path   => "out/view_detector_cncs.png",
    :save       => true,
    :rebuild    => false,
)

if CFG[:backend] == :cairo
    using CairoMakie
else
    using GLMakie
end

function view_cncs(; cfg=CFG)
    out = TOFtwin.detector_cloud_from_idf(cfg[:idf_path];
        bank_regex = cfg[:bank_regex],
        ψstride    = cfg[:ψstride],
        ηstride    = cfg[:ηstride],
        cached     = true,
        rebuild    = cfg[:rebuild],
    )

    cloud = out.cloud

    fig = Figure(size=(1000, 650))
    ax  = Axis(fig[1,1];
        xlabel = "x (m)",
        ylabel = "z (m)",
        title  = "CNCS detector coverage (X–Z projection)",
    )

    scatter!(ax, cloud.xs[cloud.idxL], cloud.zs[cloud.idxL], markersize=2, label="x < x_samp")
    scatter!(ax, cloud.xs[cloud.idxR], cloud.zs[cloud.idxR], markersize=2, label="x ≥ x_samp")

    axislegend(ax; position=:rb)

    n_used = length(cloud.xs)
    Label(fig[2,1], "pixels used = $n_used   (ψstride=$(cfg[:ψstride]), ηstride=$(cfg[:ηstride]))"; tellwidth=false)

    if cfg[:save]
        mkpath(dirname(cfg[:out_path]))
        save(cfg[:out_path], fig)
        @info "Wrote $(cfg[:out_path])"
    end

    return fig
end

fig = view_cncs()
display(fig)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin

const CFG = Dict(
    :backend    => :gl,   # :gl or :cairo
    :idf_path   => joinpath(@__DIR__, "CNCS_Definition_2025B.xml"),
    :bank_regex => nothing,   # e.g. r"bank29" to focus
    :ψstride    => 1,
    :ηstride    => 1,
    :out_path   => "out/view_detector_cncs.png",
    :save       => true,
)

if CFG[:backend] == :cairo
    using CairoMakie
else
    using GLMakie
end

function view_cncs(; cfg=CFG)
    out = TOFtwin.load_mantid_idf(cfg[:idf_path])
    inst = out.inst
    r_samp = inst.r_samp_L

    cloud = TOFtwin.detector_cloud(inst.pixels;
        r_samp=r_samp,
        bank_regex=cfg[:bank_regex],
        ψstride=cfg[:ψstride],
        ηstride=cfg[:ηstride],
    )

    fig = Figure(size=(1100, 850))
    ax  = Axis3(fig[1,1], aspect=:data, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
                title="CNCS from IDF (N=$(length(cloud.pixels_used)))")

    scatter!(ax, [r_samp[1]], [r_samp[2]], [r_samp[3]], markersize=18)

    ms = 2.5
    scatter!(ax, cloud.xs[cloud.idxL], cloud.ys[cloud.idxL], cloud.zs[cloud.idxL];
             markersize=ms, marker=:circle, label="x < x_sample")
    scatter!(ax, cloud.xs[cloud.idxR], cloud.ys[cloud.idxR], cloud.zs[cloud.idxR];
             markersize=ms, marker=:rect, label="x ≥ x_sample")

    # beam (+z)
    lines!(ax, [r_samp[1], r_samp[1]],
              [r_samp[2], r_samp[2]],
              [r_samp[3], r_samp[3] + max(4.0, 1.2*cloud.ringR)])

    # ring at y = sample y
    θ = range(-pi, pi; length=361)
    ringx = r_samp[1] .+ cloud.ringR .* sin.(θ)
    ringy = r_samp[2] .+ 0.0 .* θ
    ringz = r_samp[3] .+ cloud.ringR .* cos.(θ)
    lines!(ax, ringx, ringy, ringz)

    axislegend(ax)

    if cfg[:save]
        mkpath("out")
        save(cfg[:out_path], fig)
        @info "Wrote $(cfg[:out_path])"
    end

    return fig
end

fig = view_cncs()
display(fig)

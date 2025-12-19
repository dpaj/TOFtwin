using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using TOFtwin

# -------- live knobs --------
const CFG = Dict(
    :backend    => :gl,          # :gl or :cairo
    :L2         => 3.5,
    :surface    => :cylinder,    # :cylinder or :sphere
    :ψstride    => 1,
    :ηstride    => 1,
    :out_path   => "out/view_detector_toy.png",
    :save       => true,
)

# backend
if CFG[:backend] == :cairo
    using CairoMakie
else
    using GLMakie
end

function view_toy(; cfg=CFG)
    pix = TOFtwin.pixels_from_coverage(L2=cfg[:L2], surface=cfg[:surface])
    cloud = TOFtwin.detector_cloud(pix; ψstride=cfg[:ψstride], ηstride=cfg[:ηstride])

    fig = Figure(size=(1100, 850))
    ax  = Axis3(fig[1,1], aspect=:data, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
                title="Toy coverage (L2=$(cfg[:L2]) m, $(cfg[:surface]))")

    # sample at origin
    scatter!(ax, [0.0], [0.0], [0.0], markersize=18)

    ms = 3
    scatter!(ax, cloud.xs[cloud.idxL], cloud.ys[cloud.idxL], cloud.zs[cloud.idxL];
             markersize=ms, marker=:circle, label="x < 0")
    scatter!(ax, cloud.xs[cloud.idxR], cloud.ys[cloud.idxR], cloud.zs[cloud.idxR];
             markersize=ms, marker=:rect, label="x ≥ 0")

    # beam (+z)
    lines!(ax, [0.0, 0.0], [0.0, 0.0], [0.0, max(4.0, 1.2*cloud.ringR)])

    # ring at y=0
    θ = range(-pi, pi; length=361)
    ringx = cloud.ringR .* sin.(θ)
    ringy = 0.0 .* θ
    ringz = cloud.ringR .* cos.(θ)
    lines!(ax, ringx, ringy, ringz)

    axislegend(ax)

    if cfg[:save]
        mkpath("out")
        save(cfg[:out_path], fig)
        @info "Wrote $(cfg[:out_path])"
    end

    return fig
end

fig = view_toy()
display(fig)

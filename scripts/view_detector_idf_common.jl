# scripts/view_detector_idf_common.jl

using TOFtwin
using LinearAlgebra
using GLMakie
import GeometryBasics

const _Vec3fT   = isdefined(GeometryBasics, :Vec3f0)   ? GeometryBasics.Vec3f0   : GeometryBasics.Vec3f
const _Point3fT = isdefined(GeometryBasics, :Point3f0) ? GeometryBasics.Point3f0 : GeometryBasics.Point3f

# -----------------------
# Small helpers
# -----------------------

"""Return cfg value under any of the provided keys."""
_getany(cfg::AbstractDict, keys::Tuple, default=nothing) =
    (k = findfirst(key -> haskey(cfg, key), keys); k === nothing ? default : cfg[keys[k]])

"""Ensure we always have a Tuple of view Symbols."""
function _views_tuple(v)
    v === nothing && return (:xz, :xyz)
    v isa Symbol && return (v,)
    v isa AbstractVector{<:Symbol} && return Tuple(v)
    v isa Tuple && return v
    return (:xz, :xyz)
end

# -----------------------
# Camera: y-up, mostly along ki
# -----------------------

function _default_ki_camera(r_samp, ringR;
    ki=(0,0,1), up=(0,1,0), dist=nothing,
    side_frac=0.22, up_frac=0.12,
)
    # Work in Float32; Makie likes Float32 camera vectors/points.
    ki_v = _Vec3fT(Float32(ki[1]), Float32(ki[2]), Float32(ki[3]))
    up_v = _Vec3fT(Float32(up[1]), Float32(up[2]), Float32(up[3]))

    look = _Point3fT(Float32(r_samp[1]), Float32(r_samp[2]), Float32(r_samp[3]))

    ki_n = ki_v / max(eps(Float32), norm(ki_v))
    up_n = up_v / max(eps(Float32), norm(up_v))

    # Right-handed basis: right = ki × up
    right = cross(ki_n, up_n)
    nr = norm(right)
    if nr < 1f-6
        # If up is nearly collinear with ki, pick a fallback up.
        up_n = _Vec3fT(0f0, 1f0, 0f0)
        right = cross(ki_n, up_n)
        nr = norm(right)
        if nr < 1f-6
            up_n = _Vec3fT(1f0, 0f0, 0f0)
            right = cross(ki_n, up_n)
            nr = norm(right)
        end
    end
    right_n = right / max(1f-6, nr)

    # Make an orthonormal up consistent with ki/right
    up2 = cross(right_n, ki_n)
    up2_n = up2 / max(1f-6, norm(up2))

    d = dist === nothing ? Float32(max(4.0, 1.6 * ringR)) : Float32(dist)

    # Eye: behind the sample along -ki, with small side/up offsets
    eye = look - d * ki_n + (side_frac * d) * right_n + (up_frac * d) * up2_n

    return eye, look, up2_n
end

"""Set a free camera on a Scene (from LScene)."""
function _set_scene_camera!(scene, eye, lookat, up)
    # Ensure a 3D camera exists
    try
        Makie.cam3d_cad!(scene)
    catch
        try Makie.cam3d!(scene) catch end
    end

    # Prevent auto-recentering from fighting us (if available)
    try
        camc = Makie.cameracontrols(scene)
        if hasproperty(camc, :center)
            camc.center[] = false
        end
        # Prefer signature with cam controls
        try
            Makie.update_cam!(scene, camc, eye, lookat, up)
            return
        catch
            Makie.update_cam!(scene, eye, lookat, up)
            return
        end
    catch e
        @warn "Couldn't set scene camera" exception=(e, catch_backtrace())
    end
end

# -----------------------
# Main viewer
# -----------------------

"""
Unified detector viewer.

Required cfg keys (one of):
- :idf_src  (Symbol like :SEQUOIA or String path to an IDF)
- :idf_path (legacy alias for :idf_src)

Optional cfg keys:
- :views   => (:xz, :xyz) by default. Any subset of (:xz, :xyz, :xy, :yz)
- :layout  => :h (default) or :v
- :title   => String
- :bank_regex, :ψstride, :ηstride, :cached, :rebuild
- :fig_size, :ms_2d, :ms_3d

XZ handedness:
- :xz_y_toward_viewer => Bool (default false). If true, flips x-axis in XZ so +y is out-of-page.

Camera cfg (applies when :xyz in :views):
- :camera  => :ki (default) or :default (skip)
- :cam_ki  => (0,0,1)
- :cam_up  => (0,1,0)
- :cam_dist, :cam_side_frac, :cam_up_frac

Saving:
- :save, :out_path
"""
function view_detector_idf(; cfg::Dict)
    idf_src = _getany(cfg, (:idf_src, :idf_path), nothing)
    idf_src === nothing && error("cfg must include :idf_src (or legacy :idf_path)")

    cached  = get(cfg, :cached, true)
    rebuild = get(cfg, :rebuild, false)

    # Load instrument (tolerate old/new load_instrument_idf kw set)
    function _load()
        try
            return TOFtwin.load_instrument_idf(idf_src; cached=cached, rebuild=rebuild)
        catch
            return TOFtwin.load_instrument_idf(idf_src; cached=cached)
        end
    end

    out    = _load()
    inst   = out.inst
    r_samp = inst.r_samp_L
    pixels = hasproperty(inst, :pixels) ? inst.pixels : out.pixels

    cloud = TOFtwin.detector_cloud(pixels;
        r_samp     = r_samp,
        bank_regex = get(cfg, :bank_regex, nothing),
        ψstride    = get(cfg, :ψstride, 1),
        ηstride    = get(cfg, :ηstride, 1),
    )

    views   = _views_tuple(get(cfg, :views, (:xz, :xyz)))
    layout  = get(cfg, :layout, :h)
    figsize = get(cfg, :fig_size, (1500, 750))
    ms2d    = get(cfg, :ms_2d, 2.0)
    ms3d    = get(cfg, :ms_3d, 2.0)

    n_used = length(cloud.pixels_used)
    title_base = get(cfg, :title, "Detectors from IDF")

    fig = Figure(size=figsize)

    # Layout slots
    n = length(views)
    slot(i) = (layout == :v) ? fig[i, 1] : fig[1, i]

    # Keep reference to 3D scene if created (for camera init)
    scene3 = nothing

    for (i, v) in enumerate(views)
        if v == :xz
            ax = Axis(slot(i);
                aspect = DataAspect(),
                xlabel = "x (m)", ylabel = "z (m)",
                title  = "$title_base — X–Z (N=$n_used)",
                xreversed = get(cfg, :xz_y_toward_viewer, false),
            )

            scatter!(ax, [r_samp[1]], [r_samp[3]]; markersize=12)
            scatter!(ax, cloud.xs[cloud.idxL], cloud.zs[cloud.idxL]; markersize=ms2d, marker=:circle, label="x < x_samp")
            scatter!(ax, cloud.xs[cloud.idxR], cloud.zs[cloud.idxR]; markersize=ms2d, marker=:rect,   label="x ≥ x_samp")
            axislegend(ax; position=:rb)

            # beam (+z)
            zmax = r_samp[3] + max(4.0, 1.2 * cloud.ringR)
            lines!(ax, [r_samp[1], r_samp[1]], [r_samp[3], zmax])

            # ring in XZ
            θ = range(-pi, pi; length=361)
            ringx = r_samp[1] .+ cloud.ringR .* sin.(θ)
            ringz = r_samp[3] .+ cloud.ringR .* cos.(θ)
            lines!(ax, ringx, ringz)

        elseif v == :xy
            ax = Axis(slot(i);
                aspect=DataAspect(),
                xlabel="x (m)", ylabel="y (m)",
                title="$title_base — X–Y (N=$n_used)",
            )
            scatter!(ax, [r_samp[1]], [r_samp[2]]; markersize=12)
            scatter!(ax, cloud.xs[cloud.idxL], cloud.ys[cloud.idxL]; markersize=ms2d, marker=:circle, label="x < x_samp")
            scatter!(ax, cloud.xs[cloud.idxR], cloud.ys[cloud.idxR]; markersize=ms2d, marker=:rect,   label="x ≥ x_samp")
            axislegend(ax; position=:rb)

        elseif v == :yz
            ax = Axis(slot(i);
                aspect=DataAspect(),
                xlabel="y (m)", ylabel="z (m)",
                title="$title_base — Y–Z (N=$n_used)",
            )
            scatter!(ax, [r_samp[2]], [r_samp[3]]; markersize=12)
            scatter!(ax, cloud.ys[cloud.idxL], cloud.zs[cloud.idxL]; markersize=ms2d, marker=:circle, label="x < x_samp")
            scatter!(ax, cloud.ys[cloud.idxR], cloud.zs[cloud.idxR]; markersize=ms2d, marker=:rect,   label="x ≥ x_samp")
            axislegend(ax; position=:rb)

        elseif v == :xyz
            # 3×1 sublayout: title, big 3D scene, legend below
            sub = GridLayout(3, 1)
            slot(i)[] = sub

            rowsize!(sub, 1, Auto())         # title
            rowsize!(sub, 2, Relative(1))    # scene gets the bulk
            rowsize!(sub, 3, Auto())         # legend

            Label(sub[1, 1], "$title_base — 3D (N=$n_used)"; tellwidth=false)

            lsc = LScene(sub[2, 1]; show_axis=true, scenekw=(camera=Makie.cam3d_cad!,))
            scene3 = lsc.scene

            scatter!(lsc, [r_samp[1]], [r_samp[2]], [r_samp[3]]; markersize=18)

            pL = scatter!(lsc, cloud.xs[cloud.idxL], cloud.ys[cloud.idxL], cloud.zs[cloud.idxL];
                markersize=ms3d, marker=:circle)
            pR = scatter!(lsc, cloud.xs[cloud.idxR], cloud.ys[cloud.idxR], cloud.zs[cloud.idxR];
                markersize=ms3d, marker=:rect)

            # beam (+z)
            zmax = r_samp[3] + max(4.0, 1.2 * cloud.ringR)
            lines!(lsc, [r_samp[1], r_samp[1]],
                    [r_samp[2], r_samp[2]],
                    [r_samp[3], zmax])

            # ring at y = sample y
            θ = range(-pi, pi; length=361)
            ringx = r_samp[1] .+ cloud.ringR .* sin.(θ)
            ringy = r_samp[2] .+ 0.0 .* θ
            ringz = r_samp[3] .+ cloud.ringR .* cos.(θ)
            lines!(lsc, ringx, ringy, ringz)

            Legend(sub[3, 1], [pL, pR], ["x < x_samp", "x ≥ x_samp"]; tellwidth=false)
        else
            @warn "Unknown view '$v' (supported: :xz, :xyz, :xy, :yz)"
        end
    end

    # Footer label
    footer = "pixels used = $n_used   (ψstride=$(get(cfg,:ψstride,1)), ηstride=$(get(cfg,:ηstride,1)))"
    if layout == :v
        Label(fig[n+1, 1], footer; tellwidth=false)
    else
        Label(fig[2, 1:n], footer; tellwidth=false)
    end

    # Camera init (set once, after all plots added)
    if scene3 !== nothing && get(cfg, :camera, :ki) != :default
        ki = get(cfg, :cam_ki, (0,0,1))
        up = get(cfg, :cam_up, (0,1,0))
        dist = get(cfg, :cam_dist, nothing)
        side_frac = get(cfg, :cam_side_frac, 0.22)
        up_frac   = get(cfg, :cam_up_frac, 0.12)

        eye, look, upv = _default_ki_camera(r_samp, cloud.ringR;
            ki=ki, up=up, dist=dist, side_frac=side_frac, up_frac=up_frac
        )
        _set_scene_camera!(scene3, eye, look, upv)
    end

    if get(cfg, :save, false)
        out_path = get(cfg, :out_path, "out/view_detector.png")
        mkpath(dirname(out_path))
        save(out_path, fig)
        @info "Wrote $out_path"
    end

    return fig
end

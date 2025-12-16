"""
Minimal event representation.

- pixel_id: 1..N
- tof_s: seconds
- Ei_meV: incident energy (meV) for that frame/run
- weight: typically 1.0, or monitor-normalized, etc.
"""
struct Event
    pixel_id::Int
    tof_s::Float64
    Ei_meV::Float64
    weight::Float64
end

Event(pixel_id::Int, tof_s::Float64, Ei_meV::Float64; weight::Float64=1.0) =
    Event(pixel_id, tof_s, Ei_meV, weight)

"""
Map an event -> (Q, ω, Ef).

frame=:lab returns Q in lab; frame=:sample rotates Q into sample frame using Pose.
"""
function Qω_from_event(ev::Event, inst::Instrument; pose::Pose=Pose(), frame::Symbol=:sample)
    pid = ev.pixel_id
    p   = pixel(inst, pid)

    # Ef from TOF
    Ef = try
        Ef_from_tof(inst.L1, L2(inst, pid), ev.Ei_meV, ev.tof_s)
    catch
        return nothing
    end
    (Ef <= 0 || Ef > ev.Ei_meV) && return nothing

    Q_L, ω = Qω_from_pixel(p.r_L, ev.Ei_meV, Ef; r_samp_L=inst.r_samp_L)

    if frame === :lab
        return (Q=Q_L, ω=ω, Ef=Ef)
    elseif frame === :sample
        Q_S = T_SL(pose).R * Q_L
        return (Q=Q_S, ω=ω, Ef=Ef)
    else
        throw(ArgumentError("frame must be :lab or :sample"))
    end
end

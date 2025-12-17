"""
A simple powder toy model: one dispersing mode with Gaussian energy profile.

ω0(Q) = Δ + v*Q
S(Q,ω) = amp * exp(-(ω-ω0)^2 / (2σ^2))
"""
struct ToyModePowder
    Δ::Float64
    v::Float64
    σ::Float64
    amp::Float64
end

# keyword constructor (convenience)
ToyModePowder(; Δ::Float64, v::Float64, σ::Float64, amp::Float64=1.0) =
    ToyModePowder(Δ, v, σ, amp)

(m::ToyModePowder)(Qmag::Float64, ω::Float64) =
    m.amp * exp(-0.5*((ω - (m.Δ + m.v*Qmag))/m.σ)^2)


"""
Toy single-crystal kernel in HKL space (r.l.u.) with one dispersing mode near q0.

dq = ||hkl - q0|| (in r.l.u.)
ω0(dq) = Δ + v*dq
S(hkl,ω) = amp * exp(-dq^2/(2σQ^2)) * exp(-(ω-ω0)^2/(2σE^2))
"""
struct ToyGaussianDispHKL
    q0::Vec3
    Δ::Float64
    v::Float64
    σE::Float64
    σQ::Float64
    amp::Float64
end

ToyGaussianDispHKL(; q0::Vec3=Vec3(1.0,0.0,0.0),
    Δ::Float64=2.0, v::Float64=2.0,
    σE::Float64=0.2, σQ::Float64=0.15,
    amp::Float64=1.0) = ToyGaussianDispHKL(q0, Δ, v, σE, σQ, amp)

function (m::ToyGaussianDispHKL)(hkl::Vec3, ω::Float64)
    dq = norm(hkl - m.q0)
    ω0 = m.Δ + m.v * dq
    return m.amp * exp(-0.5*(dq/m.σQ)^2) * exp(-0.5*((ω - ω0)/m.σE)^2)
end

"""
Periodic toy single-crystal kernel in HKL (r.l.u.) with cosine dispersion.

ω0(h,k,l) = Δ + Jh*(1-cos(2πh)) + Jk*(1-cos(2πk)) + Jl*(1-cos(2πl))

S(hkl,ω) = amp * envelope(hkl) * exp(-(ω-ω0)^2/(2σE^2))

If σQ is provided, envelope is a Gaussian around q0 to keep intensity localized.
Set σQ = Inf to make it uniform in HKL.
"""
struct ToyCosineHKL
    q0::Vec3
    Δ::Float64
    Jh::Float64
    Jk::Float64
    Jl::Float64
    σE::Float64
    σQ::Float64
    amp::Float64
end

ToyCosineHKL(; q0::Vec3=Vec3(0.0,0.0,0.0),
    Δ::Float64=2.0,
    Jh::Float64=3.0, Jk::Float64=1.5, Jl::Float64=0.8,
    σE::Float64=0.25,
    σQ::Float64=0.3,
    amp::Float64=1.0) = ToyCosineHKL(q0, Δ, Jh, Jk, Jl, σE, σQ, amp)

@inline function (m::ToyCosineHKL)(hkl::Vec3, ω::Float64)
    h,k,l = hkl
    ω0 = m.Δ +
         m.Jh*(1 - cos(2π*h)) +
         m.Jk*(1 - cos(2π*k)) +
         m.Jl*(1 - cos(2π*l))

    dq = norm(hkl - m.q0)
    env = isfinite(m.σQ) ? exp(-0.5*(dq/m.σQ)^2) : 1.0

    return m.amp * env * exp(-0.5*((ω - ω0)/m.σE)^2)
end

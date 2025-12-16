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

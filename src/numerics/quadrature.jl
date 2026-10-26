## DIFFERENCES OF SPECIFIC ENTHALPY BY QUADRATURE

"""
    Δh = enthalpy_difference(fluid, p, Δp, ϑ, q)

h(p + Δp, ϑ, q) - h(p, ϑ, q) [J kg⁻¹] at fixed conserved variable ϑ and composition q, from p to p + Δp [Pa]: the
integral of the specific volume, ∂h/∂p = v, by three-point Gauss–Legendre quadrature,
    Δh = ∫ v(p′, ϑ, q) dp′ = (Δp/2) [5 v(p̄ - c) + 8 v(p̄) + 5 v(p̄ + c)]/9,   p̄ = p + Δp/2,   c = √(3/5) Δp/2,
which is exact when v is a polynomial of degree five or less in p (it is linear for `NonlinearBinaryFluid`). The
difference of two values of h (about 10⁶ J kg⁻¹) loses the differences that decide the reference state (10⁻⁴ J kg⁻¹ or
less) in Float32; the integral keeps the relative precision of the float type when Δp is small.
"""
@inline function enthalpy_difference(fluid, p, Δp, ϑ, q)
    volume = fluid(:p, :consvar, :q).specific_volume
    half = Δp / 2                                   # half the interval
    middle = p + half                               # its middle, p̄
    offset = sqrt(3 * one(half) / 5) * half          # distance c of the outer nodes from the middle
    return half * (5 * (volume(middle - offset, ϑ, q) + volume(middle + offset, ϑ, q)) + 8 * volume(middle, ϑ, q)) / 9
end

"""
    Π₁ = available_compressible_energy(fluid, P, Δp, ϑ, q)

The available compressible energy (ACE) Π₁ = h(p, ϑ, q) - h(P, ϑ, q) - (p - P) v(p, ϑ, q) [J kg⁻¹] of a parcel at the
pressure p = P + Δp [Pa], relative to the pressure P, at fixed conserved variable ϑ and composition q: the energy of
its compression from P to p, nonnegative for ∂v/∂p < 0. As h(p) - h(P) = ∫ v(s) ds from P to
p, and v(s) - v(p) = -∫ ∂v/∂p(s′) ds′ from s to p, exchanging the order of integration gives
    Π₁ = ∫ (v(s) - v(p)) ds = -∫ (s - P) ∂v/∂p(s) ds   from P to p,
here by three-point Gauss–Legendre quadrature, with s - P = t ∈ [0, Δp],
    Π₁ = -(Δp/2) [5 t₋ ∂v/∂p(P + t₋) + 8 t₀ ∂v/∂p(P + t₀) + 5 t₊ ∂v/∂p(P + t₊)]/9,
    t₀ = Δp/2,   t± = t₀ ± √(3/5) Δp/2,
which is exact when ∂v/∂p is a polynomial of degree four or less in p (a constant for `NonlinearBinaryFluid`, where
Π₁ = -∂v/∂p Δp²/2). Π₁ is of order Δp² ∂v/∂p, so the difference of the three terms h(p) - h(P), (p - P) v(p), each of
order Δp v, loses most of its digits (all of them in Float32 for Δp of a few Pa); the integral keeps the relative
precision of the float type.
"""
@inline function available_compressible_energy(fluid, P, Δp, ϑ, q)
    volume_functions = fluid(:p, :consvar, :q).volume_functions
    half = Δp / 2                                   # half the interval, t₀
    offset = sqrt(3 * one(half) / 5) * half          # distance of the outer nodes from the middle
    t₋, t₊ = half - offset, half + offset           # the outer nodes, from P
    ∂v₋ = volume_functions(P + t₋, ϑ, q)[2]          # ∂v/∂p at the three nodes
    ∂v₀ = volume_functions(P + half, ϑ, q)[2]
    ∂v₊ = volume_functions(P + t₊, ϑ, q)[2]
    return -half * (5 * (t₋ * ∂v₋ + t₊ * ∂v₊) + 8 * half * ∂v₀) / 9
end

## C¹ PIECEWISE-CUBIC HERMITE INTERPOLATION

"""
    profile = HermiteProfile(F, n)
    (y, dy/dz, d²y/dz²) = hermite(profile, z)

C¹ piecewise-cubic Hermite interpolant through `n` nodes of increasing heights `profile.z` [m], with the values
`profile.y` and the slopes `profile.d` (dy/dz) at the nodes, all of element type `F` and updated in place. Beyond the
end nodes it is extended linearly with the end slopes (zero curvature). At an interior node, where the curvature of a
C¹ cubic jumps, `hermite` returns the mean of its values on either side. With one node, it is the line through it.
"""
struct HermiteProfile{F}
    z::Vector{F}    # heights of the nodes, increasing [m]
    y::Vector{F}    # values at the nodes
    d::Vector{F}    # slopes dy/dz at the nodes
end
HermiteProfile(F::Type, n::Integer) = HermiteProfile(zeros(F, n), zeros(F, n), zeros(F, n))

@inline function hermite((; z, y, d)::HermiteProfile{F}, ζ) where F
    n = length(z)
    # outside the nodes: the tangent line at the end node, y = y_end + d_end (ζ - z_end), with zero curvature
    (n == 1 || ζ < z[1]) && return (y[1] + d[1] * (ζ - z[1]), d[1], zero(F))
    ζ > z[n] && return (y[n] + d[n] * (ζ - z[n]), d[n], zero(F))
    # segment k, z_k ≤ ζ ≤ z_{k+1}: the last one for ζ = z_n
    k = min(searchsortedlast(z, ζ), n - 1)
    value, slope, curvature = hermite_segment(z, y, d, k, ζ)
    # at an interior node the curvature jumps: the mean of the values at the end of segment k-1 and the start of segment k
    if ζ == z[k] && k > 1
        curvature = (curvature + hermite_segment(z, y, d, k - 1, ζ)[3]) / 2
    end
    return value, slope, curvature
end

"""
    (y, dy/dz, d²y/dz²) = hermite_segment(z, y, d, k, ζ)

The cubic of the Hermite interpolant on segment k, z_k ≤ ζ ≤ z_{k+1}, with t = (ζ - z_k)/h ∈ [0, 1],
h = z_{k+1} - z_k, Δy = y_{k+1} - y_k:
    y   = y_k + Δy (3t² - 2t³) + h d_k (t³ - 2t² + t) + h d_{k+1} (t³ - t²)
    y'  = Δy (6t - 6t²)/h + d_k (3t² - 4t + 1) + d_{k+1} (3t² - 2t)
    y'' = Δy (6 - 12t)/h² + d_k (6t - 4)/h + d_{k+1} (6t - 2)/h
which take the values y_k, y_{k+1} and the slopes d_k, d_{k+1} at the ends.
"""
@inline function hermite_segment(z, y, d, k, ζ)
    h = z[k+1] - z[k]
    t = (ζ - z[k]) / h
    Δy = y[k+1] - y[k]
    value = y[k] + Δy * (3t^2 - 2t^3) + h * (d[k] * (t^3 - 2t^2 + t) + d[k+1] * (t^3 - t^2))
    slope = Δy * (6t - 6t^2) / h + d[k] * (3t^2 - 4t + 1) + d[k+1] * (3t^2 - 2t)
    curvature = (Δy * (6 - 12t) / h + d[k] * (6t - 4) + d[k+1] * (6t - 2)) / h
    return value, slope, curvature
end

"""
    d = monotone_slopes!(d, z, secant)

Slopes at the nodes `z` (increasing) of a monotone C¹ piecewise-cubic interpolant (Fritsch–Carlson with Brodlie's
weights, as in PCHIP), from the secant slopes `secant[k]` = (y_{k+1} - y_k)/(z_{k+1} - z_k) of its segments: at an
interior node, the weighted harmonic mean
    d_k = (w₁ + w₂)/(w₁/s_{k-1} + w₂/s_k),   w₁ = 2h_k + h_{k-1},   w₂ = h_k + 2h_{k-1},
of the secants on either side (zero if they differ in sign). At an end node it is the three-point, second-order
estimate from the two end segments, d₁ = ((2h₁ + h₂) s₁ - h₁ s₂)/(h₁ + h₂), limited so as to keep the shape: zero if its
sign differs from that of s₁, and 3s₁ if s₁ and s₂ differ in sign and |d₁| > 3|s₁|. With one segment, the secant.
"""
function monotone_slopes!(d, z, secant)
    n = length(z)
    if n == 2
        d[1] = d[2] = secant[1]
        return d
    end
    d[1] = end_slope(z[2] - z[1], z[3] - z[2], secant[1], secant[2])
    d[n] = end_slope(z[n] - z[n-1], z[n-1] - z[n-2], secant[n-1], secant[n-2])
    for k in 2:n-1
        h₋, h₊ = z[k] - z[k-1], z[k+1] - z[k]
        s₋, s₊ = secant[k-1], secant[k]
        w₁, w₂ = 2h₊ + h₋, h₊ + 2h₋
        d[k] = s₋ * s₊ > 0 ? (w₁ + w₂) / (w₁ / s₋ + w₂ / s₊) : zero(eltype(d))
    end
    return d
end

"""
    d = end_slope(h₁, h₂, s₁, s₂)

Slope at an end node of a monotone interpolant (`monotone_slopes!`), from the end segment (width h₁, secant s₁) and
the next one (h₂, s₂).
"""
@inline function end_slope(h₁, h₂, s₁, s₂)
    d = ((2h₁ + h₂) * s₁ - h₁ * s₂) / (h₁ + h₂)
    sign(d) != sign(s₁) && return zero(d)
    sign(s₁) != sign(s₂) && abs(d) > 3abs(s₁) && return 3s₁
    return d
end

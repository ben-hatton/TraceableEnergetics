## MINIMUM OF A FUNCTION OF ONE VARIABLE, REFINED FROM A DISCRETE MINIMUM

"""
    (x, f(x)) = refine_minimum(f, r, levels, c, f_c, xatol)

Refines the minimum of a function f of one variable, with r = -df/dx, from its smallest value f_c = f(x_c) over the
increasing `levels`, at x_c = `levels[c]`:
- if r > 0 at x_c, f decreases upwards: the minimum is the root of r between x_c and the level above, if r < 0 there;
- if r < 0 at x_c, f increases upwards: the minimum is the root of r between the level below and x_c, if r > 0 there;
- otherwise (r = 0 at x_c, an end of the levels with r of the outward sign, or no change of sign) it is x_c.
The root is found to within `xatol` by the bracketing method of Alefeld, Potra and Shi (Roots.A42; Brent's method
stalls in BigFloat), and kept only if f is lower there than at x_c. Near a minimum f changes by about r'/2 times the
square of the error, so `xatol` of order sqrt(eps) times the scale of x leaves f(x) at round-off, in about half the
evaluations of r needed to find x itself to round-off.
"""
function refine_minimum(f::Objective, r::Residual, levels, c, f_c, xatol) where {Objective, Residual}
    # the function arguments are typed so that Julia specialises on them, which it does not for a function only passed
    # on to another one
    x_c = levels[c]
    r_c = r(x_c)
    if r_c > 0 && c < length(levels) && r(levels[c+1]) < 0
        x = find_zero(r, (x_c, levels[c+1]), A42(); xatol, xrtol = zero(xatol))
    elseif r_c < 0 && c > 1 && r(levels[c-1]) > 0
        x = find_zero(r, (levels[c-1], x_c), A42(); xatol, xrtol = zero(xatol))
    else
        return x_c, f_c
    end
    y = f(x)
    return y < f_c ? (x, y) : (x_c, f_c)
end

## useful functions:
scale_depth(space, fluid) = sound_speed2(fluid, (; p = fluid.p0, T = fluid.T0, q = fluid.q0)) / space.g
exp_profile(z, space, fluid) = exp( - z / scale_depth(space, fluid))
lin_profile(z, space, fluid) = 1 - z / scale_depth(space, fluid)
const_profile(z, space, fluid) = 1.
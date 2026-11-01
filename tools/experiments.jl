# Helpers shared by the experiment scripts of examples/ (notes/experiments/experiment_plan.md lists them):
#     include(joinpath(@__DIR__, "..", "..", "tools", "experiments.jl"))
# after tools/setup.jl. Each script reads `quick = quick_run()` in its parameter block and writes its files only to
# `output_dir(@__DIR__, quick)`. The analysis helpers (discrete wavenumbers, fits of growth rates and of orders of
# convergence, the Jacobian of the tendencies, the model recipe) are included from here as they are written.

"""
    quick = quick_run()

Whether the script runs its quick variant (small grid, short duration) rather than the full experiment: `true` if the
environment variable `TE_QUICK` is `"true"`, `false` if it is `"false"` or unset. Any other value is an error, so that
a misspelt value never starts a long run by accident. The test suite sets `TE_QUICK=true` when it includes the scripts.

    TE_QUICK=true julia -t 4 --project=. examples/<script>/run.jl
"""
function quick_run()
    value = get(ENV, "TE_QUICK", "false")         # the full experiment by default
    value == "true" && return true
    value == "false" && return false
    error("TE_QUICK must be \"true\" or \"false\", got \"$value\"")
end

"""
    dir = output_dir(script_dir, quick)

The folder, created if needed, to which the script in `script_dir` writes its data files and figures:
`script_dir/output` for the full experiment and `script_dir/output/quick` for the quick variant, so that a quick run
never overwrites the files of a full one. Both are untracked (`examples/*/output/` in .gitignore).
"""
output_dir(script_dir, quick::Bool) = mkpath(quick ? joinpath(script_dir, "output", "quick") : joinpath(script_dir, "output"))

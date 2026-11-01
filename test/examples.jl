# The experiment scripts of examples/, run in their quick variant as smoke tests. Opt-in, as they take minutes:
#     TE_EXAMPLES=true julia -t 4 --project=. test/runtests.jl
# Each script ends with its own @testset of checks against reference values, which then becomes part of the suite.

"""
    test_examples(scripts; examples = joinpath(@__DIR__, "..", "examples"))

Include each script, `examples/<script>/run.jl` (or a path to a `.jl` file), with `TE_QUICK=true`, each in a fresh
module of its own (the scripts define globals with the same names) and in a @testset named after it. The module has
its own `include`, so that the script's includes relative to `@__DIR__` work as they do when it is run on its own.
"""
function test_examples(scripts; examples = joinpath(@__DIR__, "..", "examples"))
    withenv("TE_QUICK" => "true") do                                  # every script runs its quick variant
        for script in scripts
            # the script's path: a .jl file as given, otherwise examples/<script>/run.jl
            path = endswith(script, ".jl") ? abspath(script) : abspath(joinpath(examples, script, "run.jl"))
            @testset "$script" begin
                mod = Module(Symbol(basename(dirname(path))))              # a fresh module, with Base and Core
                Core.eval(mod, :(include(p) = Base.include($mod, p)))      # include into this module, as at top level
                Base.include(mod, path)
            end
        end
    end
end

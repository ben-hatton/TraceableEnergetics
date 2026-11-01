# Helpers of tools/experiments.jl: the quick flag and the output folders of the experiment scripts

@testset "experiment helpers" begin
    # TE_QUICK: unset or "false" gives the full run, "true" the quick one, anything else is an error
    withenv("TE_QUICK" => nothing) do
        @test quick_run() === false
    end
    withenv("TE_QUICK" => "false") do
        @test quick_run() === false
    end
    withenv("TE_QUICK" => "true") do
        @test quick_run() === true
    end
    for typo in ("1", "yes", "True", "")
        withenv("TE_QUICK" => typo) do
            @test_throws ErrorException quick_run()
        end
    end
    # output folders: created on demand, the quick one inside the full one
    mktempdir() do dir
        full, quick = output_dir(dir, false), output_dir(dir, true)
        @test full == joinpath(dir, "output") && isdir(full)
        @test quick == joinpath(dir, "output", "quick") && isdir(quick)
        @test output_dir(dir, true) == quick                         # a second call reuses the folder
    end
end

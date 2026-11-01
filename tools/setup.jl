# Common setup for experiment and analysis scripts:  include("<relative path>/tools/setup.jl")
import Pkg

# activate the TraceableEnergetics environment, unless it is already active (as when the test suite, started with
# --project=., includes a script); CFAnelastics and CFCompressible come from their GitHub forks via [sources]
let project = realpath(joinpath(@__DIR__, "..", "Project.toml"))                 # this repository's environment
    active = Base.active_project()                                                # nothing if no environment is active
    (active !== nothing && isfile(active) && realpath(active) == project) || Pkg.activate(dirname(project))
end

# allow packages to automatically update when edited, in an interactive session only (a batch run or the test suite
# has no use for it)
if isinteractive()
    using Revise
end

# these define function for a computing backend
using LoopManagers

# defining functions rather than const variables ensures that we re-initialize backends each time we re-run
plain(args...)      = LoopManagers.PlainCPU();
omp(args...)        = LoopManagers.MainThread(plain());
threads(args...)    = LoopManagers.MultiThread(plain());
SIMD(args...)       = LoopManagers.VectorizedCPU(args...);
ompSIMD(args...)    = LoopManagers.MainThread(SIMD(args...));
tSIMD(args...)      = LoopManagers.MultiThread(SIMD(args...));
fake(args...)       = LoopManagers.FakeGPU(tSIMD(args...));

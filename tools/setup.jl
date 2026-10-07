# Common setup for experiment and analysis scripts:  include("<relative path>/tools/setup.jl")
import Pkg

# activate the TraceableEnergetics environment; the work-in-progress packages come from packages/ via [sources]
Pkg.activate(joinpath(@__DIR__, ".."))

# allow packages to automatically update when edited
using Revise

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

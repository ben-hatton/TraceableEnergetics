import Pkg

# activate climflows-energetics
Pkg.activate("$(@__DIR__)/..")

# add modules to path
let path=(@__DIR__)*"/../CFAnelastic"; path in LOAD_PATH || push!(LOAD_PATH, path) end;
# let path=(@__DIR__)*"/../CFCompressible"; path in LOAD_PATH || push!(LOAD_PATH, path) end;
# let path=(@__DIR__)*"/../Loops"; path in LOAD_PATH || push!(LOAD_PATH, path) end;

# allow modules to automatically update when edited
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
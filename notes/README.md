# Notes

Specifications and plans for TraceableEnergetics, one subfolder per subject.

| Folder | Contents |
| --- | --- |
| models/ | `traceable_energetics.tex` (and PDF): the AN2D and FC2D models, continuous and discrete |
| energy_budget/ | `energy.tex`: energy budgets of the compressible and anelastic fluids, available energy (the "energy notes") |
| reference_state/ | `ape_reference_state.tex` (and PDF): algorithms for the Lorenz reference state; `reference_state_plan.md`: the history of its implementation; `prototypes/`: Python prototypes that give reference results for the Julia tests |
| experiments/ | `experiment_plan.md`: tutorials, verification tests and validation experiments to implement in examples/ |

LaTeX build files (`.aux`, `.log`, `.toc`, `.fls`, `.fdb_latexmk`) are untracked; build each document in its own folder,
e.g. `cd notes/models && latexmk -pdf traceable_energetics.tex`.

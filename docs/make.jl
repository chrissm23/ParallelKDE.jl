using ParallelKDE
using Documenter

DocMeta.setdocmeta!(ParallelKDE, :DocTestSetup, :(using ParallelKDE); recursive=true)

makedocs(;
  modules=[ParallelKDE],
  authors="Christian Sustay (christian.sustay@tum.de)",
  sitename="ParallelKDE.jl",
  format=Documenter.HTML(;
    canonical="https://chrissm23.github.io/ParallelKDE.jl",
    edit_link="main",
    assets=String[],
  ),
  pages=[
    "Home" => "index.md",
    "Installation" => "installation.md",
    "Estimators" => Any[
      "estimators/parallel_estimator.md",
      "estimators/rot_estimator.md",
    ],
    "Development" => Any[
      "development/adding_estimators.md",
      "development/current_tools.md",
    ],
    "API Reference" => Any[
      "api_reference/parallel_kde.md",
      "api_reference/estimators.md",
      "api_reference/grids.md",
      "api_reference/kdes.md",
      "api_reference/direct_space.md",
      "api_reference/fourier_space.md",
      "api_reference/devices.md"
    ],
  ],
  remotes=nothing,
)

deploydocs(;
  repo="github.com/chrissm23/ParallelKDE.jl",
  devbranch="dev",
)

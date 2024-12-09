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
    ],
)

deploydocs(;
    repo="github.com/chrissm23/ParallelKDE.jl",
    devbranch="main",
)

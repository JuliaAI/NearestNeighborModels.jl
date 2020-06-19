using MLJNearestNeighborsInterface
using Documenter

makedocs(;
    modules=[MLJNearestNeighborsInterface],
    authors="Sebastian Vollmer <s.vollmer.4@warwick.ac.uk> and contributors",
    repo="https://github.com/vollmersj/MLJNearestNeighborsInterface.jl/blob/{commit}{path}#L{line}",
    sitename="MLJNearestNeighborsInterface.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://vollmersj.github.io/MLJNearestNeighborsInterface.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/vollmersj/MLJNearestNeighborsInterface.jl",
)

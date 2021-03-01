using Documenter, NearestNeighborModels

makedocs(;
    authors = """
        Anthony D. Blaom <anthony.blaom@gmail.com>, 
        Sebastian Vollmer <s.vollmer.4@warwick.ac.uk>, 
        Thibaut Lienart <thibaut.lienart@gmail.com> and 
        Okon Samuel <okonsamuel50@gmail.com>
        """,
    format = Documenter.HTML(;
        prettyurls= get(ENV, "CI", "false") == "true"
    ),
    modules = [NearestNeighborModels],
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ],
    repo = "https://github.com/alan-turing-institute/NearestNeighborModels.jl/blob/{commit}{path}#L{line}",
    sitename = "NearestNeighborModels.jl",
)

deploydocs(;
    repo="github.com/alan-turing-institute/NearestNeighborModels.jl.git",
)


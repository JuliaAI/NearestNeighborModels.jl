using Documenter, MLJBase, NearestNeighborModels

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
    doctest = false, # don't runt doctest as doctests are automatically run separately in ci.
    repo = "https://github.com/JuliaAI/NearestNeighborModels.jl/blob/{commit}{path}#L{line}",
    sitename = "NearestNeighborModels.jl",
)

# By default Documenter does not deploy docs just for PR
# this causes issues with how we're doing things and ends
# up choking the deployment of the docs, so  here we
# force the environment to ignore this so that Documenter
# does indeed deploy the docs
#ENV["GITHUB_EVENT_NAME"] = "pull_request"

deploydocs(;
    deploy_config = Documenter.GitHubActions(),
    repo="github.com/JuliaAI/NearestNeighborModels.jl.git",
    push_preview=true
)

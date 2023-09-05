using TaijaPlotting
using Documenter

DocMeta.setdocmeta!(TaijaPlotting, :DocTestSetup, :(using TaijaPlotting); recursive=true)

makedocs(;
    modules=[TaijaPlotting],
    authors="Patrick Altmeyer",
    repo="https://github.com/JuliaTrustworthyAI/TaijaPlotting.jl/blob/{commit}{path}#{line}",
    sitename="TaijaPlotting.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaTrustworthyAI.github.io/TaijaPlotting.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaTrustworthyAI/TaijaPlotting.jl",
    devbranch="main",
)

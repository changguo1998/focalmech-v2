dr = abspath(@__DIR__, "../out/build/x64-Debug/")
for f in filter(endswith("exe"), readdir(dr))
    cp(joinpath(dr, f), joinpath(@__DIR__, f); force=true)
end

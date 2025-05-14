exefile = ARGS[1]
# exefile = "fm"

@info "test program: $exefile"

efname = let
    t = filter(endswith("exe"), readdir(@__DIR__))
    s = filter(startswith(exefile), t)
    if isempty(s)
        _x = nothing
    else
        _x = s[1]
    end
    _x
end

@info "update exe files"
run(`julia copy_exe_here.jl`)

@info "run program: $efname"
run(Cmd(Cmd([efname]); dir=pwd()))

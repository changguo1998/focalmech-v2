include("../../julia/FocalMechv2/src/FocalMechv2.jl")

function gauss(x, μ, σ)
    return exp(-((x - μ) / σ)^2)
end

nr = 2
ne = 2
np = nr
npts = 101
dt = 0.01
dstrike = 10.0
ddip = 10.0
drake = 10.0
nstrike = round(Int, 360.0/dstrike)
ndip = floor(Int, 90.0/ddip) + 1
nrake = floor(Int, 180.0/drake) + 1

gs = FocalMechv2.GlobalSetting("abc", nr, ne, np, [0.1], [0.5], 0.0, 0.0, nstrike, ndip, nrake, dstrike, ddip, drake)
gs_c = FocalMechv2.GlobalSetting_C(gs)

rs = map(1:nr) do i
    data = randn(npts, 1).*0.01
    data[:] .+= gauss.((0:npts-1).*dt, 0.5, 0.1)
    # data = FocalMechv2.pp_records_in_different_frequency(w, gs.low_frequency, gs.high_frequency, dt)
    FocalMechv2.Record_C(i+100, data)
end

Gidx = CartesianIndices((nr, ne))
gfs = Vector{FocalMechv2.GreenFunction_C}(undef, nr * ne)

for i = eachindex(gfs)
    I = Gidx[i]
    tw = gauss.((0:npts-1).*dt, 0.5, 0.1)
    tm = zeros(npts, gs_c.n_frequency_pair)
    for c = eachcol(tm)
        c[:] .= tw[:]
    end
    gfs[i] = FocalMechv2.GreenFunction_C(I[1]+100, I[2], tm, tm, tm, tm, tm, tm)
end

# ps = Vector{FocalMechv2.Phase_C}(undef, np)
# for i = eachindex(ps)
#     ps[i] = FocalMechv2.Phase_C(rand(1:nr), rand(1:ne), rand(1:4), rand(0:4), rand(0:4), rand(4:6), rand([true, false]))
# end

ps = Vector{FocalMechv2.Phase_C}(undef, np)
for i = eachindex(ps)
    ps[i] = FocalMechv2.Phase_C(i+100, 1, 1, 25, 25, 50, true)
end


open("input_db.bin", "w") do io
    FocalMechv2.write_to_database(io, gs_c)
    FocalMechv2.write_to_database.(io, rs)
    FocalMechv2.write_to_database.(io, gfs)
    FocalMechv2.write_to_database.(io, ps)
    return nothing
end

exit(0)

run(Cmd(`database_io.exe`; dir=pwd()))

# io = open("output.bin", "r")
io = open("input_db.bin", "r")
gs_t = FocalMechv2.GlobalSetting_C(io);
rs_t = map(x -> FocalMechv2.Record_C(io, gs_t), 1:gs_t.n_record);
gf_t = map(x -> FocalMechv2.GreenFunction_C(io, rs_t, gs_t), 1:gs_t.n_event_location*gs_t.n_record);
ps_t = map(x -> FocalMechv2.Phase_C(io), 1:gs_t.n_phase);
close(io)

@test gs_c == gs_t
@test all(map((x, y)->(x.id == y.id) && (x.data == y.data), rs, rs_t))
@test begin
    map(gfs, gf_t) do x, y
        flag = true
        for f in fieldnames(FocalMechv2.GreenFun_C)
            flag &=  (getfield(x, f) == getfield(y, f))
        end
        return flag
    end |> all
end
@test ps == ps_t

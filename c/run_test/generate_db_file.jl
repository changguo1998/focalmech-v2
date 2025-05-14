using Test
include("../../julia/FocalMechv2/src/FocalMechv2.jl")

nr = 2
ne = 3
np = 4
npts = 5
dt = 0.1
dstrike = 1.0
ddip = 1.0
drake = 1.0
nstrike = round(Int, 360.0/dstrike)
ndip = round(Int, 90.0/ddip) + 1
nrake = round(Int, 360.0/drake)

gs = FocalMechv2.GlobalSetting("abc", nr, ne, np, [0.1], [0.5, 1.0], 0.0, 0.0, nstrike, ndip, nrake, dstrike, ddip, drake)
gs_c = FocalMechv2.GlobalSetting_C(gs)

rs = map(1:nr) do i
    w = randn(npts)
    data = FocalMechv2.pp_records_in_different_frequency(w, gs.low_frequency, gs.high_frequency, dt)
    FocalMechv2.Record_C(i, data)
end

Gidx = CartesianIndices((nr, ne))
gfs = Vector{FocalMechv2.GreenFun_C}(undef, nr * ne)

for i = eachindex(gfs)
    I = Gidx[i]
    tm = randn(npts, gs_c.n_frequency_pair)
    gfs[i] = FocalMechv2.GreenFun_C(I[1], I[2], tm, tm, tm, tm, tm, tm)
end

ps = Vector{FocalMechv2.Phase_C}(undef, np)
for i = eachindex(ps)
    ps[i] = FocalMechv2.Phase_C(rand(1:nr), rand(1:ne), rand(1:4), rand(0:4), rand(0:4), rand(4:6), rand([true, false]))
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

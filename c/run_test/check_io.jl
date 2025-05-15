using Test
include("../../julia/FocalMechv2/src/FocalMechv2.jl")

io = open("output.bin", "r")
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

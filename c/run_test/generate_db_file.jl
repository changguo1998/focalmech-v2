include("../../julia/FocalMechv2/src/FocalMechv2.jl")

nr = 2
ne = 3
npts = 10
dt = 0.1

gs = FocalMechv2.GlobalSetting("abc", nr, ne, [0.1], [0.5, 1.0], 0.0, 0.0, 1.0, 1.0, 1.0)
gs_c = FocalMechv2.GlobalSetting_C(gs)

rs = map(1:nr) do i
    w = randn(npts)
    data = FocalMechv2.pp_records_in_different_frequency(w, gs.low_frequency, gs.high_frequency, dt)
    FocalMechv2.Record_C(i, data, [(1, 3, true)])
end

Gidx = CartesianIndices((nr, ne))
gfs = Vector{FocalMechv2.GreenFun_C}(undef, nr*ne)

for i = eachindex(gfs)
    I = Gidx[i]
    tm = randn(npts, gs_c.n_frequency_pairs)
    gfs[i] = FocalMechv2.GreenFun_C(I[1], I[2], tm, tm, tm, tm, tm, tm)
end

open("input_db.bin", "w") do io
    FocalMechv2.write_to_database(io, gs_c)
    FocalMechv2.write_to_database.(io, rs)
    FocalMechv2.write_to_database.(io, gfs)
end

io = open("output.bin", "r")
gs_t = FocalMechv2.read_global_setting_from_database(io)
rs_t = map(x->FocalMechv2.read_record_from_database(io, gs_t), 1:gs_t.n_record);
gf_t = map(x->FocalMechv2.read_green_fun_from_database(io, rs_t, gs_t), 1:gs_t.n_event_location*gs_t.n_record)
close(io)

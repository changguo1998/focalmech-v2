using GLMakie
include("../../julia/FocalMechv2/src/FocalMechv2.jl")

io = open("input_db.bin", "r")
gs = FocalMechv2.GlobalSetting_C(io);
rs = map(x -> FocalMechv2.Record_C(io, gs), 1:gs.n_record);
gf = map(x -> FocalMechv2.GreenFunction_C(io, rs, gs), 1:gs.n_event_location*gs.n_record);
ps = map(x -> FocalMechv2.Phase_C(io), 1:gs.n_phase);
close(io)

res_gpu = open(io->FocalMechv2.Result_C(io), "result_gpu.bin", "r");
res_omp = open(io->FocalMechv2.Result_C(io), "result_omp.bin", "r");


fig = Figure()
ax = Axis(fig[1,1])
heatmap!(res.waveform[1,:,:])

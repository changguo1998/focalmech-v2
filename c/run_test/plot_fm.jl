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

foreach(fieldnames(FocalMechv2.Result_C)) do f
    println(f, " ", getfield(res_gpu, f) == getfield(res_omp, f));
end;

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(rs[1].data[:])
ax2 = Axis(fig[2,1])
lines!(gf[1].g11[:].+5.0)
lines!(gf[1].g22[:].+4.0)
lines!(gf[1].g33[:].+3.0)
lines!(gf[1].g12[:].+2.0)
lines!(gf[1].g13[:].+1.0)
lines!(gf[1].g23[:])
ax3 = Axis(fig[1:2,2])
heatmap!(res_gpu.waveform[1,:,:])

using Pkg
Pkg.activate("/home/vddiazz/phd/w/code/jl/Solitons1D.jl/Solitons1D")

using Solitons1D

using ProgressMeter
using NPZ

###

dx = 0.1
dt = 0.0005

type = "fr"
model = "phi4"
moduli = "aB"

space = collect(-15:dx:15)
time = [100,dt]

a0 	= 10.
v0 	= -0.3
b0 	= 0.392699*v0^2 + 0.0924209*v0^4 - 0.00654868*v0^6 - 0.00994437*v0^8
db0 	= 0.
incs = [a0,v0,b0,db0]

path = "/home/vddiazz/Desktop/temp_scc"

########## results
#==
a2,v2,b2,db2 = moduli_RK4_nm2(model,moduli,incs,space,time)
npzwrite(path*"/a2_v=$(v0)_dt=$(dt).npy", a2)
npzwrite(path*"/v2_v=$(v0)_dt=$(dt).npy", v2)
npzwrite(path*"/b2_v=$(v0)_dt=$(dt).npy", b2)
npzwrite(path*"/db2_v=$(v0)_dt=$(dt).npy", db2)
==#
###

c0 = 0.
dc0 = 2.

push!(incs,c0)
push!(incs,dc0)

a3,v3,b3,db3,c3,dc3 = moduli_RK4_nm3(type,model,moduli,incs,space,time)
npzwrite(path*"/a_m3_v=$(v0)_dt=$(dt).npy", a3)
npzwrite(path*"/da_m3_v=$(v0)_dt=$(dt).npy", v3)
npzwrite(path*"/b_m3_v=$(v0)_dt=$(dt).npy", b3)
npzwrite(path*"/db_m3_v=$(v0)_dt=$(dt).npy", db3)
npzwrite(path*"/c_m3_v=$(v0)_dt=$(dt).npy", c3)
npzwrite(path*"/dc_m3_v=$(v0)_dt=$(dt).npy", dc3)


#--------- pkg

using Pkg
Pkg.activate("/home/velni/phd/w/code/jl/Solitons1D.jl/Solitons1D")

using Solitons1D

using BenchmarkTools

#---------- moduli_RK4_nm2

a0 = 6.
v0 = -0.241
b0 = 0.392699*v0^2 + 0.0924209*v0^4 - 0.00654868*v0^6 - 0.00994437*v0^8
db0 = 0.

incs = [a0,v0,b0,db0]
time = [175000,0.0005]
out = "/home/velni/Escritorio"

@btime moduli_RK4_nm2(incs,time,out,"npy")


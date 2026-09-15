using Pkg
Pkg.activate("/home/vddiazz/phd/w/code/jl/Solitons1D.jl/Solitons1D")

using Solitons1D

###

model = "phi4"
space = collect(-15:0.01:15)

a0 = 10.
v0 = 0.1
c1_0 = 4.86254539e-07 + 4.99995057e-01*v0^2 + 3.75161008e-01*v0^4 + 2.59169274e-01*v0^6 + 1.95964711e-01*v0^8
c2_0 = 9.21466047e-07 - 7.97457659e-06*v0^2 + 2.50193910e-01*v0^4 + 2.32627308e-01*v0^6 + 1.97771320e-01*v0^8

gamma = 1/sqrt(1-v0^2)

###

m3_step(model, "pR2", gamma, collect(space), [a0, c1_0, c2_0], [v0, 0., 0.])

using Pkg
Pkg.activate("/home/vddiazz/phd/w/code/jl/Solitons1D.jl/Solitons1D")

using Solitons1D

###

model = "phi4"
moduli = "aB"
space = collect(-15:0.01:15)

a0 = 10.
v0 = -0.3
b0 = 0.392699*v0^2 + 0.0924209*v0^4 - 0.00654868*v0^6 - 0.00994437*v0^8
db0 = 0.

gamma = 0.

###

dt = 0.0005
t = dt*1

x1 = a0
dx1 = v0
x2 = b0
dx2 = db0

### m2

ddot_step_m2 = m2_step(model, moduli, gamma, space, [a0, b0], [v0, db0])

println()
println("ddot_step_m2: $(ddot_step_m2)")

### m3

ddot_step_m3 = m3_step(model, moduli, gamma, space, [a0, b0, 0.], [v0, db0, 0.])

println()
println("ddot_step_m3: $(ddot_step_m3)")



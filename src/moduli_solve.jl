using LinearAlgebra
using JLD2
using SpecialFunctions

### import moduli equations 

function eq_import(path::String)
    # path : PATH : path to .txt with expression
    
    # read file
    eq_str = read(path, String)

    # parse
    eq = Meta.parse(eq_str)

    # anonymous function
    eq_f = eval(:((t,a,da,b,db)->$eq))

    return eq_f
end

### 4th-order Runge-Kutta

function moduli_RK4_m2(f1,f2,incs,time,out)
    # eq1 : Function : ddm_1 = f1
    # eq2 : Function : ddm_2 = f2
    # incs : Vector{Float64} : initial conditions : [m1_0, dm1_0, m2_0, dm2_0]
    # out : PATH : path to output folder

    # unpacking
    N = time[1]
    dt = time[2]

    x1 = incs[1]
    dx1 = incs[2]
    x2 = incs[3]
    dx2 = incs[4]

    # initialization
    l1 = Float64[]
    ld1 = Float64[]
    l2 = Float64[]
    ld2 = Float64[]
    
    t = 0.

    #---------- RK4
    for n in 1:1:N
		# save data
        push!(l1,x1)
        push!(ld1,dx1)
        push!(l2,x2)
        push!(ld2,dx2)
	
		# compute next step
		t = t+dt
		
		k1_1 = dt*dx1
		k1_d1 = dt*f1(t,x1,dx1,x2,dx2)
		k1_2 = dt*dx2
		k1_d2 = dt*f2(t,x1,dx1,x2,dx2)

		k2_1 = dt*(dx1 + k1_d1/2.)
		k2_d1 = dt*f1(t, x1+k1_1/2., dx1+k1_d1/2., x2+k1_2/2., dx2+k1_d2/2.)
		k2_2 = dt*(dx2 + k1_d2/2.)
		k2_d2 = dt*f2(t, x1+k1_1/2., dx1+k1_d1/2., x2+k1_2/2., dx2+k1_d2/2.)

		k3_1 = dt*(dx1 + k2_d1/2.)
		k3_d1 = dt*f1(t, x1+k2_1/2., dx1+k2_d1/2., x2+k2_2/2., dx2+k2_d2/2.)
		k3_2 = dt*(dx2 + k2_d2/2.)
		k3_d2 = dt*f2(t, x1+k2_1/2., dx1+k2_d1/2., x2+k2_2/2., dx2+k2_d2/2.)

		k4_1 = dt*(dx1 + k3_d1)
		k4_d1 = dt*f1(t, x1+k3_1/2., dx1+k3_d1/2., x2+k3_2/2., dx2+k3_d2/2.)
		k4_2 = dt*(dx2 + k3_d2/2)
		k4_d2 = dt*f2(t, x1+k3_1/2., dx1+k3_d1/2., x2+k3_2/2., dx2+k3_d2/2.)		

		x1n = x1 + k1_1/6. + k2_1/3. + k3_1/3. + k4_1/6.
		dx1n = dx1 + k1_d1/6. + k2_d1/3. + k3_d1/3. + k4_d1/6.
		x2n = x2 + k1_2/6. + k2_2/3. + k3_2/3. + k4_2/6.
		dx2n = dx2 + k1_d2/6. + k2_d2/3. + k3_d2/3. + k4_d2/6.

		# update variables
		x1 = x1n
		dx1 = dx1n
		x2 = x2n
		dx2 = dx2n
	
        println("done: t = $(round(t,digits=6))")
    end

    #----------- data saving

    path = out*"kak_moduli_v=$(dx1).jld2"
    @save path l1 ld1 l2 ld2
    println("data saved at "*path)

end

### numerical integration
#=
using ForwardDiff

function moduli_dynamics(profile,incs,time,out)

    # params

    space = -5000:0.01:5000

    N = time[1]
    dt = time[2]

    X = [a,b]

    # coefficient functions

    G = ForwardDiff.gradient

    e = zeros(Float64, N)

    for (j,x) in enumerate(Jarr)
        e[j] = G(X -> profile(x,X), X)

    # RK4

    moduli_RK4_m2(f1,f2,incs,time,out)
=#

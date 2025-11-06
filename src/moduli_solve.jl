using LinearAlgebra
using JLD2
using SpecialFunctions
using ProgressMeter
using Base.Threads

### import moduli equations # UNUSED

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

### 4th-order Runge-Kutta (analytical) # UNUSED

function moduli_RK4_am2(f1,f2,incs,time,out)
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

### moduli equations of motion

using ForwardDiff

# MOVE TO AUX.JL
function F_kak(model,moduli,x, M, gamma)
    if model == "phi4"
        if moduli == "aB"
            f = tanh(x+M[1]) - tanh(x-M[1]) - 1 + (M[2]/tanh(M[1]))*( sinh(x+M[1])/(cosh(x+M[1]))^2 - sinh(x-M[1])/(cosh(x-M[1]))^2 )
        elseif moduli == "maB"
            f = tanh(gamma*(x+M[1])) - tanh(gamma*(x-M[1])) - 1 + (M[2]/tanh(M[1]))*( sinh(gamma*(x+M[1]))/(cosh(gamma*(x+M[1])))^2 - sinh(gamma*(x-M[1]))/(cosh(gamma*(x-M[1])))^2 )
        elseif moduli == "mpR"
            f = tanh(gamma*(x+M[1])) - tanh(gamma*(x-M[1])) - 1 + (M[2]/tanh(M[1]))*( gamma*(x+M[1])*sech(gamma*(x+M[1]))^2 - gamma*(x-M[1])*sech(gamma*(x+M[1]))^2 )
        end
    end
    return f
end

# MOVE TO AUX.JL
function U_kak(model,moduli, x, M,gamma)
    if model == "phi4"
        U = 0.5*(1-F_kak(model,moduli,x,M,gamma)^2)^2
    end
    return U
end

# MOVE TO AUX.JL
function W_kak(model,moduli,x, M,gamma)

    if model == "phi4"
        if moduli == "aB"
            deriv = -sech(M[1]-x)^2 + sech(M[1]+x)^2 + M[2]*coth(M[1])*(-sech(M[1]-x)^3 + sech(M[1]+x)^3 + sech(M[1]-x)*tanh(M[1]-x)^2 - sech(M[1]+x)*tanh(M[1]+x)^2 )
            W = 0.5*(deriv)^2 + U_kak(model,moduli,x,M,gamma)
        elseif moduli == "maB"
            deriv = -gamma*sech((-M[1]+x)*gamma)^2 + gamma*sech((M[1]+x)*gamma)^2 + M[2]*coth(M[1])*(-gamma*sech((-M[1]+x)*gamma)^3 + gamma*sech((M[1]+x)*gamma)^3 + gamma*sech((-M[1]+x)*gamma)*tanh((-M[1]+x)*gamma)^2 - gamma*sech((M[1]+x)*gamma)*tanh((M[1]+x)*gamma)^2 )
            W = 0.5*(deriv)^2 + U_kak(model,moduli,x,M,gamma)
        end
    end
    return W
end

function m2_step(model::String,moduli::String,gamma::Float64,x::Vector{Float64}, M0::Vector{Float64}, dM0::Vector{Float64})
    
    # params
    dx = x[2]-x[1]

    # coefficient functions
    e = zeros(Float64, length(M0),length(x))
    H = zeros(Float64, length(M0),length(M0),length(x))
    dW = zeros(Float64, length(M0),length(x))

    @threads for idx in 1:length(x)
        e[:,idx] .= ForwardDiff.gradient(M -> F_kak(model,moduli,x[idx],M,gamma), M0)
        
        H[:,:,idx] .= ForwardDiff.hessian(M -> F_kak(model,moduli,x[idx],M,gamma), M0)
        
        dW[:,idx] .= ForwardDiff.gradient(M -> W_kak(model,moduli,x[idx],M,gamma), M0)
    end

    #--- numerical integrals
    ddot = zeros(Float64, length(M0))

    # terms
    ee_11 = sum(e[1,:] .* e[1,:])*dx
    ee_12 = sum(e[1,:] .* e[2,:])*dx
    ee_21 = sum(e[2,:] .* e[1,:])*dx
    ee_22 = sum(e[2,:] .* e[2,:])*dx

    He_1 = sum(e[1,:] .* (H[1,1,:]*dM0[1]*dM0[1] + H[1,2,:]*dM0[1]*dM0[2] + H[2,1,:]*dM0[2]*dM0[1] + H[2,2,:]*dM0[2]*dM0[2]) )*dx
    He_2 = sum(e[2,:] .* (H[1,1,:]*dM0[1]*dM0[1] + H[1,2,:]*dM0[1]*dM0[2] + H[2,1,:]*dM0[2]*dM0[1] + H[2,2,:]*dM0[2]*dM0[2]) )*dx

    pW_1 = -sum(dW[1,:])*dx
    pW_2 = -sum(dW[2,:])*dx

    #

    D1 = pW_1 - He_1
    D2 = pW_2 - He_2
    M = ee_11*ee_22 - ee_21*ee_12

    # eqs
    ddot[1] = (D1 - D2*ee_21/ee_22)/(ee_11-(ee_21*ee_12)/ee_22)
    ddot[2] = D2*(1/ee_22 + ee_12*ee_21/ee_22/M) - D1*ee_12/M

    return ddot
end

# 4th-order Runge-Kutta (numerical)

function moduli_RK4_nm2(model::String,moduli::String,incs::Array{Float64},time::Array{Float64},out::String,output_format::String)
    # incs : Vector{Float64} : initial conditions : [m1_0, dm1_0, m2_0, dm2_0]
    # out : PATH : path to output folder

    if (output_format != "jld2") && (output_format != "npy")
        println("invalid output data type")
        return
    end

    # unpacking
    N = time[1]
    dt = time[2]

    x1 = incs[1]
    dx1 = incs[2]
    x2 = incs[3]
    dx2 = incs[4]
    gamma = 1/sqrt(1-incs[2]^2)

    # initialization
    l1 = Float64[]
    ld1 = Float64[]
    l2 = Float64[]
    ld2 = Float64[]
    
    t = 0.
    space = collect(-15:0.1:15)

    println()
    println("#--------------------------------------------------#")
    println()
    println("KAK collision: a0=$(x1), v0=$(dx1), B0=$(x2), dB0=$(dx2)")
    println()

    #---------- RK4
    @showprogress 1 "Computing..." for n in 1:1:N 
        @inbounds @fastmath begin
		    # save data
            push!(l1,x1)
            push!(ld1,dx1)
            push!(l2,x2)
            push!(ld2,dx2)
	
            # compute next step
            t = t+dt
            
            ddot_step_1 = m2_step(model,moduli,gamma,space, [x1,x2], [dx1,dx2])
            k1_1 = dt*dx1
            k1_d1 = dt*ddot_step_1[1]
            k1_2 = dt*dx2
            k1_d2 = dt*ddot_step_1[2]

            ddot_step_2 = m2_step(model,moduli,gamma, space, [x1+k1_1/2., x2+k1_2/2.], [dx1+k1_d1/2., dx2+k1_d2/2.])
            k2_1 = dt*(dx1 + k1_d1/2.)
            k2_d1 = dt*ddot_step_2[1]
            k2_2 = dt*(dx2 + k1_d2/2.)
            k2_d2 = dt*ddot_step_2[2]

            ddot_step_3 = m2_step(model,moduli,gamma, space, [x1+k2_1/2., x2+k2_2/2.], [dx1+k2_d1/2., dx2+k2_d2/2.])
            k3_1 = dt*(dx1 + k2_d1/2.)
            k3_d1 = dt*ddot_step_3[1]
            k3_2 = dt*(dx2 + k2_d2/2.)
            k3_d2 = dt*ddot_step_3[2]

            ddot_step_4 = m2_step(model,moduli,gamma, space, [x1+k3_1/2., x2+k3_2/2.], [dx1+k3_d1/2., dx2+k3_d2/2.])
            k4_1 = dt*(dx1 + k3_d1)
            k4_d1 = dt*ddot_step_4[1]
            k4_2 = dt*(dx2 + k3_d2/2)
            k4_d2 = dt*ddot_step_4[2]

            x1n = x1 + k1_1/6. + k2_1/3. + k3_1/3. + k4_1/6.
            dx1n = dx1 + k1_d1/6. + k2_d1/3. + k3_d1/3. + k4_d1/6.
            x2n = x2 + k1_2/6. + k2_2/3. + k3_2/3. + k4_2/6.
            dx2n = dx2 + k1_d2/6. + k2_d2/3. + k3_d2/3. + k4_d2/6.

            # update variables
            x1 = x1n
            dx1 = dx1n
            x2 = x2n
            dx2 = dx2n
        end
    end

    #----------- data saving
    
    if output_format == "jld2"
        path = out*"/kak_moduli_v=$(ld1[1])_dt=$(dt).jld2"
        @save path l1 ld1 l2 ld2
  
    elseif output_format == "npy"
        npzwrite(out*"/a_v=$(ld1[1])_dt=$(dt).npy", l1)
        npzwrite(out*"/da_v=$(ld1[1])_dt=$(dt).npy", ld1)
        npzwrite(out*"/b_v=$(ld1[1])_dt=$(dt).npy", l2)
        npzwrite(out*"/db_v=$(ld1[1])_dt=$(dt).npy", ld2)
    end

    println()
    println("Data saved at "*out )
    println()
    println("#--------------------------------------------------#")
    print()

    return l1,ld1,l2,ld2
end

using LinearAlgebra
using Serialization
using JLD2
using SpecialFunctions
using ProgressMeter
using Base.Threads
using LoopVectorization
using Interpolations

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

function moduli_RK4_nm2(type::String,model::String,moduli::String,incs::Array{Float64},time::Array{Float64},out::String,output_format::String)
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
    
    if moduli == "aB" # necessary for grid selection
        gamma = 0.
    elseif moduli == "maB"
        gamma = 1/sqrt(1-incs[2]^2)
    end

    # initialization
    l1 = Float64[]
    ld1 = Float64[]
    l2 = Float64[]
    ld2 = Float64[]

    if type == "interp"
        #=
        Ch_grid = open("/home/velni/phd/w/scc/1d/kak_moduli/data/grid/phi4/aB/Ch_model=$(model)_moduli=$(moduli).jls") do io; deserialize(io); end
        gpV_grid = open("/home/velni/phd/w/scc/1d/kak_moduli/data/grid/phi4/aB/gpV_model=$(model)_moduli=$(moduli).jls") do io; deserialize(io); end
        =#
        Ch_grid = npzread("/home/velni/phd/w/scc/1d/kak_moduli/data/grid/phi4/aB/Ch_model=$(model)_moduli=$(moduli)_gamma=$(gamma).npy")
        dV_grid = npzread("/home/velni/phd/w/scc/1d/kak_moduli/data/grid/phi4/aB/dV_model=$(model)_moduli=$(moduli)_gamma=$(gamma).npy")
    
        X1 = npzread("/home/velni/phd/w/scc/1d/kak_moduli/data/grid/phi4/aB/X1_model=$(model)_moduli=$(moduli).npy")
        X2 = npzread("/home/velni/phd/w/scc/1d/kak_moduli/data/grid/phi4/aB/X2_model=$(model)_moduli=$(moduli).npy")
    end
    
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
            
            if type == "full-res"
                ddot_step_1 = m2_step(model,moduli,gamma, space, [x1,x2], [dx1,dx2])
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

            elseif type == "interp"
                ddot_step_1 = m2_step_interp(Ch_grid,dV_grid,X1,X2, [x1,x2], [dx1,dx2])
                k1_1 = dt*dx1
                k1_d1 = dt*ddot_step_1[1]
                k1_2 = dt*dx2
                k1_d2 = dt*ddot_step_1[2]

                ddot_step_2 = m2_step_interp(Ch_grid,dV_grid,X1,X2, [x1+k1_1/2., x2+k1_2/2.], [dx1+k1_d1/2., dx2+k1_d2/2.])
                k2_1 = dt*(dx1 + k1_d1/2.)
                k2_d1 = dt*ddot_step_2[1]
                k2_2 = dt*(dx2 + k1_d2/2.)
                k2_d2 = dt*ddot_step_2[2]

                ddot_step_3 = m2_step_interp(Ch_grid,dV_grid,X1,X2, [x1+k2_1/2., x2+k2_2/2.], [dx1+k2_d1/2., dx2+k2_d2/2.])
                k3_1 = dt*(dx1 + k2_d1/2.)
                k3_d1 = dt*ddot_step_3[1]
                k3_2 = dt*(dx2 + k2_d2/2.)
                k3_d2 = dt*ddot_step_3[2]

                ddot_step_4 = m2_step_interp(Ch_grid,dV_grid,X1,X2, [x1+k3_1/2., x2+k3_2/2.], [dx1+k3_d1/2., dx2+k3_d2/2.])
                k4_1 = dt*(dx1 + k3_d1)
                k4_d1 = dt*ddot_step_4[1]
                k4_2 = dt*(dx2 + k3_d2/2)
                k4_d2 = dt*ddot_step_4[2]

            end

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

### moduli grid interpolation

function mkgrid_m2(model::String,moduli::String,gamma::Float64,X1::Array{Float64},X2::Array{Float64},x::Array{Float64},out::String,output_format::String)

    # params
    dx = x[2]-x[1]

    l1 = length(X1)
    l2 = length(X2)
    lx = length(x)

    # coefficient grids
    ChS = zeros(Float64, 2,2,2,l1,l2) # (nº of indices, nº of moduli)
    gpV = zeros(Float64, 2,l1,l2)
    g = zeros(Float64, 2,2,l1,l2)
    gc = zeros(Float64, 2,2,l1,l2)

    e = zeros(Float64, 2,lx)
    H = zeros(Float64, 2,2,lx)
    dW = zeros(Float64, 2,lx)

    println()
    println("#--------------------------------------------------#")
    println()
    println("e.o.m. coefficients grid")
    println()
   
    M0 = zeros(Float64, 2)

    @showprogress 1 "Computing grid..." for idx1 in 1:l1
        @inbounds @fastmath for idx2 in 1:l2, idx in 1:lx
            M0[1] = X1[idx1]; M0[2] = X2[idx2]

            e[:,idx] .= ForwardDiff.gradient(M -> F_kak(model,moduli,x[idx],M,gamma), M0)
            H[:,:,idx] .= ForwardDiff.hessian(M -> F_kak(model,moduli,x[idx],M,gamma), M0)
            dW[:,idx] .= ForwardDiff.gradient(M -> W_kak(model,moduli,x[idx],M,gamma), M0)
    
            #--- numerical integrals
    
            g_11 = sum(e[1,:] .* e[1,:])*dx
            g_12 = sum(e[1,:] .* e[2,:])*dx
            g_21 = sum(e[2,:] .* e[1,:])*dx
            g_22 = sum(e[2,:] .* e[2,:])*dx

            # contravariant metric
            det_g = g_11*g_22-g_12*g_21
            gc_11 = g_22/det_g
            gc_12 = -g_21/det_g
            gc_21 = -g_12/det_g
            gc_22 = g_11/det_g

            # Levi-Civita connection
            C_1_11 = gc_11*sum(e[1,:].*H[1,1,:])*dx + gc_12*sum(e[2,:].*H[1,1,:])*dx
            C_1_12 = gc_11*sum(e[1,:].*H[1,2,:])*dx + gc_12*sum(e[2,:].*H[1,2,:])*dx
            C_1_21 = gc_11*sum(e[1,:].*H[2,1,:])*dx + gc_12*sum(e[2,:].*H[2,1,:])*dx
            C_1_22 = gc_11*sum(e[1,:].*H[2,2,:])*dx + gc_12*sum(e[2,:].*H[2,2,:])*dx

            C_2_11 = gc_21*sum(e[1,:].*H[1,1,:])*dx + gc_22*sum(e[2,:].*H[1,1,:])*dx
            C_2_12 = gc_21*sum(e[1,:].*H[1,2,:])*dx + gc_22*sum(e[2,:].*H[1,2,:])*dx
            C_2_21 = gc_21*sum(e[1,:].*H[2,1,:])*dx + gc_22*sum(e[2,:].*H[2,1,:])*dx
            C_2_22 = gc_21*sum(e[1,:].*H[2,2,:])*dx + gc_22*sum(e[2,:].*H[2,2,:])*dx 

            # contravariant derivative of potential
            dV_1 = sum(dW[1,:])*dx
            dV_2 = sum(dW[2,:])*dx

            dcV_1 = gc_11*dV_1 + gc_12*dV_2
            dcV_2 = gc_21*dV_1 + gc_22*dV_2

            # return 
            ChS[1,1,1,idx1,idx2] = C_1_11
            ChS[1,1,2,idx1,idx2] = C_1_12
            ChS[1,2,1,idx1,idx2] = C_1_21
            ChS[1,2,2,idx1,idx2] = C_1_22

            ChS[2,1,1,idx1,idx2] = C_2_11
            ChS[2,1,2,idx1,idx2] = C_2_12
            ChS[2,2,1,idx1,idx2] = C_2_21
            ChS[2,2,2,idx1,idx2] = C_2_22

            gpV[1,idx1,idx2] = dcV_1
            gpV[2,idx1,idx2] = dcV_2

            g[1,1,idx1,idx2] = g_11
            g[1,2,idx1,idx2] = g_12
            g[2,1,idx1,idx2] = g_21
            g[2,2,idx1,idx2] = g_22

            gc[1,1,idx1,idx2] = gc_11
            gc[1,2,idx1,idx2] = gc_12
            gc[2,1,idx1,idx2] = gc_21
            gc[2,2,idx1,idx2] = gc_22

        end
    end
   
    #---------- point interpolation at X_1 = 0

    bad = findall(isnan, g[1,1,:,1])

    @showprogress 1 "Fixing nan values..." for bad_idx in bad
        xs = [X1[bad_idx-2],X1[bad_idx-1],X1[bad_idx+1]]

        @inbounds @fastmath for idx2 in 1:l2    
            ys_ChS_1_11 = [ChS[1,1,1,bad_idx-2,idx2],ChS[1,1,1,bad_idx-1,idx2],ChS[1,1,1,bad_idx+1,idx2]]
            ys_ChS_1_12 = [ChS[1,1,2,bad_idx-2,idx2],ChS[1,1,2,bad_idx-1,idx2],ChS[1,1,2,bad_idx+1,idx2]]
            ys_ChS_1_21 = [ChS[1,2,1,bad_idx-2,idx2],ChS[1,2,1,bad_idx-1,idx2],ChS[1,2,1,bad_idx+1,idx2]]
            ys_ChS_1_22 = [ChS[1,2,2,bad_idx-2,idx2],ChS[1,2,2,bad_idx-1,idx2],ChS[1,2,2,bad_idx+1,idx2]]
            ys_ChS_2_11 = [ChS[2,1,1,bad_idx-2,idx2],ChS[2,1,1,bad_idx-1,idx2],ChS[2,1,1,bad_idx+1,idx2]]
            ys_ChS_2_12 = [ChS[2,1,2,bad_idx-2,idx2],ChS[2,1,2,bad_idx-1,idx2],ChS[2,1,2,bad_idx+1,idx2]]
            ys_ChS_2_21 = [ChS[2,2,1,bad_idx-2,idx2],ChS[2,2,1,bad_idx-1,idx2],ChS[2,2,1,bad_idx+1,idx2]]
            ys_ChS_2_22 = [ChS[2,2,2,bad_idx-2,idx2],ChS[2,2,2,bad_idx-1,idx2],ChS[2,2,2,bad_idx+1,idx2]]
            
            ys_gpV_1 = [gpV[1,bad_idx-2,idx2],gpV[1,bad_idx-1,idx2],gpV[1,bad_idx+1,idx2]]
            ys_gpV_2 = [gpV[2,bad_idx-2,idx2],gpV[2,bad_idx-1,idx2],gpV[2,bad_idx+1,idx2]]

            ChS[1,1,1,bad_idx,idx2] = lip(2,X1[bad_idx],xs,ys_ChS_1_11)
            ChS[1,1,2,bad_idx,idx2] = lip(2,X1[bad_idx],xs,ys_ChS_1_12)
            ChS[1,2,1,bad_idx,idx2] = lip(2,X1[bad_idx],xs,ys_ChS_1_21)
            ChS[1,2,2,bad_idx,idx2] = lip(2,X1[bad_idx],xs,ys_ChS_1_22)
            ChS[2,1,1,bad_idx,idx2] = lip(2,X1[bad_idx],xs,ys_ChS_2_11)
            ChS[2,1,2,bad_idx,idx2] = lip(2,X1[bad_idx],xs,ys_ChS_2_12)
            ChS[2,2,1,bad_idx,idx2] = lip(2,X1[bad_idx],xs,ys_ChS_2_21)
            ChS[2,2,2,bad_idx,idx2] = lip(2,X1[bad_idx],xs,ys_ChS_2_22)

            gpV[1,bad_idx,idx2] = lip(2,X1[bad_idx],xs,ys_gpV_1)
            gpV[2,bad_idx,idx2] = lip(2,X1[bad_idx],xs,ys_gpV_2)
        end
    end

    #----------- data saving
    
    if output_format == "jls"
        path_c = out*"/Ch_model=$(model)_moduli=$(moduli)_gamma=$(gamma).jls"
        path_v = out*"/dV_model=$(model)_moduli=$(moduli)_gamma=$(gamma).jls"
        path_g = out*"/g_model=$(model)_moduli=$(moduli)_gamma=$(gamma).jls"
        path_gc = out*"/gc_model=$(model)_moduli=$(moduli)_gamma=$(gamma).jls"

        open(path_c, "w") do io; serialize(io, ChS); end
        open(path_v, "w") do io; serialize(io, gpV); end
        open(path_g, "w") do io; serialize(io, g); end
        open(path_gc, "w") do io; serialize(io, gc); end
        
    elseif output_format == "npy"
        npzwrite(out*"/Ch_model=$(model)_moduli=$(moduli)_gamma=$(gamma).npy", ChS)
        npzwrite(out*"/dV_model=$(model)_moduli=$(moduli)_gamma=$(gamma).npy", gpV)
        npzwrite(out*"/g_model=$(model)_moduli=$(moduli)_gamma=$(gamma).npy", g)
        npzwrite(out*"/gc_model=$(model)_moduli=$(moduli)_gamma=$(gamma).npy", gc)

    end

    println()
    println("Data saved at "*out )
    println()
    println("#--------------------------------------------------#")
    println()

end

function m2_step_interp(Ch_grid::Array{Float64},dV_grid::Array{Float64},X1::Array{Float64},X2::Array{Float64},M0::Array{Float64}, dM0::Array{Float64})
    
    #--- interpolation
    l1 = length(X1)
    l2 = length(X2)

    Ch_itp = zeros(Float64, 2,2,2)
    dV_itp = zeros(Float64, 2)

    #----- bilinear interpolation # BAD
    #= 
    # find indices of cell that contains M0
    idx1 = searchsortedlast(X1,M0[1])
    idx2 = searchsortedlast(X2,M0[2]) 

    # clamp values
    idx1 = clamp(idx1,1,l1-1)
    idx2 = clamp(idx2,1,l2-1)

    # cell corners
    X1_B = X1[idx1]; X1_T = X1[idx1+1]
    X2_B = X2[idx2]; X2_T = X2[idx2+1]
    t = (M0[1] - X1_B)/(X1_T - X1_B)
    u = (M0[2] - X2_B)/(X2_T - X2_B)
    
    @tturbo for k in 1:2, j in 1:2, i in 1:2
        # cell corner values
        Q11 = Ch_grid[i,j,k,idx1,idx2]
        Q12 = Ch_grid[i,j,k,idx1,idx2+1]
        Q21 = Ch_grid[i,j,k,idx1+1,idx2]
        Q22 = Ch_grid[i,j,k,idx1+1,idx2+1]

        # bilinear interpolation        
        Ch_itp[i,j,k] = (1-t)*(1-u)*Q11 + (1-t)*u*Q12 + t*(1-u)*Q21 + t*u*Q22
    end

    @tturbo for i in 1:2
        # cell corner values (dV)
        Q11 = dV_grid[i,idx1,idx2]
        Q12 = dV_grid[i,idx1,idx2+1]
        Q21 = dV_grid[i,idx1+1,idx2]
        Q22 = dV_grid[i,idx1+1,idx2+1]

        dV_itp[i] = (1-t)*(1-u)*Q11 + (1-t)*u*Q12 + t*(1-u)*Q21 + t*u*Q22
    
    end
    =#
    #----- quadratic interpolation # ALMOST GOOD
    #=
    M1 = M0[1]; M2= M0[2]

    idx1 = searchsortedlast(X1, M1)
    idx2 = searchsortedlast(X2, M2)

    idx1 = clamp.(idx1-1:idx1+1, 1,l1)
    idx2 = clamp.(idx2-1:idx2+1, 1,l2)

    g = zeros(Float64, 3)

    @inbounds @fastmath for i in 1:2
        dV = dV_grid[i,:,:]

        @tturbo for l in 1:3
            L0 = ((M1-X1[idx1[2]])*(M1-X1[idx1[3]]))/((X1[idx1[1]]-X1[idx1[2]])*(X1[idx1[1]]-X1[idx1[3]]))
            L1 = ((M1-X1[idx1[1]])*(M1-X1[idx1[3]]))/((X1[idx1[2]]-X1[idx1[1]])*(X1[idx1[2]]-X1[idx1[3]]))
            L2 = ((M1-X1[idx1[1]])*(M1-X1[idx1[2]]))/((X1[idx1[3]]-X1[idx1[1]])*(X1[idx1[3]]-X1[idx1[2]]))

            # interpolation along X1
            g[l] = dV[idx1[1],idx2[l]]*L0 + dV[idx1[2],idx2[l]]*L1 + dV[idx1[3],idx2[l]]*L2
        end

        # interpolation along X2 
        L0 = ((M2-X2[idx2[2]])*(M2-X2[idx2[3]]))/((X2[idx2[1]]-X2[idx2[2]])*(X2[idx2[1]]-X2[idx2[3]]))
        L1 = ((M2-X2[idx2[1]])*(M2-X2[idx2[3]]))/((X2[idx2[2]]-X2[idx2[1]])*(X2[idx2[2]]-X2[idx2[3]]))
        L2 = ((M2-X2[idx2[1]])*(M2-X2[idx2[2]]))/((X2[idx2[3]]-X2[idx2[1]])*(X2[idx2[3]]-X2[idx2[2]]))

        #
        
        dV_itp[i] = g[1]*L0 + g[2]*L1 + g[3]*L2

        @inbounds @fastmath for k in 1:2, j in 1:2
            Ch = Ch_grid[i,j,k,:,:]

            @tturbo for l in 1:3
                L0 = ((M1-X1[idx1[2]])*(M1-X1[idx1[3]]))/((X1[idx1[1]]-X1[idx1[2]])*(X1[idx1[1]]-X1[idx1[3]]))
                L1 = ((M1-X1[idx1[1]])*(M1-X1[idx1[3]]))/((X1[idx1[2]]-X1[idx1[1]])*(X1[idx1[2]]-X1[idx1[3]]))
                L2 = ((M1-X1[idx1[1]])*(M1-X1[idx1[2]]))/((X1[idx1[3]]-X1[idx1[1]])*(X1[idx1[3]]-X1[idx1[2]]))

                # interpolation along X1
                g[l] = Ch[idx1[1],idx2[l]]*L0 + Ch[idx1[2],idx2[l]]*L1 + Ch[idx1[3],idx2[l]]*L2
            end

            # interpolation along X2
            L0 = ((M2-X2[idx2[2]])*(M2-X2[idx2[3]]))/((X2[idx2[1]]-X2[idx2[2]])*(X2[idx2[1]]-X2[idx2[3]]))
            L1 = ((M2-X2[idx2[1]])*(M2-X2[idx2[3]]))/((X2[idx2[2]]-X2[idx2[1]])*(X2[idx2[2]]-X2[idx2[3]]))
            L2 = ((M2-X2[idx2[1]])*(M2-X2[idx2[2]]))/((X2[idx2[3]]-X2[idx2[1]])*(X2[idx2[3]]-X2[idx2[2]]))

            #
        
            Ch_itp[i,j,k] = g[1]*L0 + g[2]*L1 + g[3]*L2
    
        end
    end
    =#
    #----- spline interpolation (Interpolations.jl)

    X1_r = LinRange(X1[1], X1[end], length(X1))
    X2_r = LinRange(X2[1], X2[end], length(X2))

    @inbounds for i in 1:2
        M1 = M0[1]; M2 = M0[2]       
        
        Y_dV = dV_grid[i,:,:]
        p_itp_dV = interpolate(Y_dV, BSpline(Quadratic(Line())), OnGrid())
        itp_dV = extrapolate(scale(p_itp_dV, X1_r, X2_r),Line())
        dV_itp[i] = itp_dV[M1,M2]

        @inbounds for k in 1:2, j in 1:2
            Y_Ch = Ch_grid[i,j,k,:,:]
            p_itp_Ch = interpolate(Y_Ch, BSpline(Quadratic(Line())), OnGrid())
            itp_Ch = extrapolate(scale(p_itp_Ch, X1_r, X2_r),Line())
            Ch_itp[i,j,k] = itp_Ch[M1,M2]
        end
    end

    #----- coefficients

    i1 = -( Ch_itp[1,1,1]*dM0[1]*dM0[1] + Ch_itp[1,1,2]*dM0[1]*dM0[2] + Ch_itp[1,2,1]*dM0[2]*dM0[1] + Ch_itp[1,2,2]*dM0[2]*dM0[2] ) - dV_itp[1]
    i2 = -( Ch_itp[2,1,1]*dM0[1]*dM0[1] + Ch_itp[2,1,2]*dM0[1]*dM0[2] + Ch_itp[2,2,1]*dM0[2]*dM0[1] + Ch_itp[2,2,2]*dM0[2]*dM0[2] ) - dV_itp[2]

    return [i1,i2]
end

function FAST_moduli_RK4_nm2(Ch_grid::Array{Float64},dV_grid::Array{Float64},X1::Array{Float64},X2::Array{Float64},model::String,moduli::String,incs::Array{Float64},time::Array{Float64},out::String,output_format::String)
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
    
    # initialization
    l1 = Float64[]
    ld1 = Float64[]
    l2 = Float64[]
    ld2 = Float64[]

    t = 0.

    #---------- RK4

    println()
    @showprogress 1 "Computing: KAK collision" for n in 1:1:N 
        @inbounds @fastmath begin 
            # save data
            push!(l1,x1)
            push!(ld1,dx1)
            push!(l2,x2)
            push!(ld2,dx2)

            # compute next step
            t = t+dt
                
            ddot_step_1 = m2_step_interp(Ch_grid,dV_grid,X1,X2, [x1,x2], [dx1,dx2])
            k1_1 = dt*dx1
            k1_d1 = dt*ddot_step_1[1]
            k1_2 = dt*dx2
            k1_d2 = dt*ddot_step_1[2]

            ddot_step_2 = m2_step_interp(Ch_grid,dV_grid,X1,X2, [x1+k1_1/2., x2+k1_2/2.], [dx1+k1_d1/2., dx2+k1_d2/2.])
            k2_1 = dt*(dx1 + k1_d1/2.)
            k2_d1 = dt*ddot_step_2[1]
            k2_2 = dt*(dx2 + k1_d2/2.)
            k2_d2 = dt*ddot_step_2[2]

            ddot_step_3 = m2_step_interp(Ch_grid,dV_grid,X1,X2, [x1+k2_1/2., x2+k2_2/2.], [dx1+k2_d1/2., dx2+k2_d2/2.])
            k3_1 = dt*(dx1 + k2_d1/2.)
            k3_d1 = dt*ddot_step_3[1]
            k3_2 = dt*(dx2 + k2_d2/2.)
            k3_d2 = dt*ddot_step_3[2]

            ddot_step_4 = m2_step_interp(Ch_grid,dV_grid,X1,X2, [x1+k3_1/2., x2+k3_2/2.], [dx1+k3_d1/2., dx2+k3_d2/2.])
            k4_1 = dt*(dx1 + k3_d1)
            k4_d1 = dt*ddot_step_4[1]
            k4_2 = dt*(dx2 + k3_d2/2)
            k4_d2 = dt*ddot_step_4[2]

            # compute new variables
                
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
    println()

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

    return l1,ld1,l2,ld2
end

    

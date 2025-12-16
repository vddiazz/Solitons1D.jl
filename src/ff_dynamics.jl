using LinearAlgebra
using JLD2
using NPZ
using ProgressMeter

#---------------- kak dynamics

function ff_kak(model::String,ff_space,ff_time,incs,bcs::String,out::String,output_format::String)
    # ff_space : Vector{Float64} : spatial dimension : [xi,xf,dx]
    # ff_time : Vector{Float64} : temporal dimension : [tf,dt]
    # incs : Vector{Float64} : initial conditions : [a0,v0]
    # out : PATH : path of output folder

    # PENDING: Does not work with sG

    if (output_format != "jld2") && (output_format != "npy")
        println("invalid output data type")
        return
    end

    # unpacking
    xi = ff_space[1]
    xf = ff_space[2]
    dx = ff_space[3]

    tf = ff_time[1]
    dt = ff_time[2]

    J = convert(Int64,(xf-xi)/dx)
    N = convert(Int64,tf/dt)

    Jarr = range(xi,xf-dx; step=dx)
    Narr = range(0.0,tf-dt; step=dt)

    # initialization
    F = zeros(Float64, J,N)

    a0 = incs[1]
    v0 = incs[2]
    gamma = 1.0/sqrt(1.0-(v0^2))

    #---------- initial conditions
    
    # temporal ic: initial configuration
    for j in 1:1:J
        x = xi + j*dx

        if model == "phi4"
            F[j,1] = tanh(gamma*(x+a0)) - tanh(gamma*(x-a0)) - 1 # field
            F[j,2] = F[j,1] - ( ((gamma*v0)/(cosh(gamma*(x+a0)))^2 + (gamma*v0)/(cosh(gamma*(x-a0)))^2 )*dt ) # derivative
        end
        if model == "phi6" # PENDING
            F[j,1] = 0
            F[j,2] = F[j,1] + 0
        end
        if model == "sG"
            F[j,1] = 4*( atan(exp(gamma*(x+a0))) - atan(exp(gamma*(x-a0))) )
            F[j,2] = F[j,1] + 4*( -(exp(gamma*(x+a0)*gamma*v0))/(1+exp(2*gamma*(x+a0))) - (exp(gamma*(x-a0)*gamma*v0))/(1+exp(2*gamma*(x-a0)))  )*dt 
        end
    end

    # spatial ic: assymptotic value
    if model == "phi4"
        F[1,:] .= -1.0
        F[end,:] .= -1.0
    end
    if model == "phi6" # PENDING
        F[1,:] .= 0
        F[end,:] .= 0
    end
    if model == "sG"
        F[1,:] .= 0.0
        F[end,:] .= 0.0
    end

    #---------- time evolution
   
    println()
    println("#--------------------------------------------------#")
    println()
    println("Full field dynamics: model=$(model), v0=$(incs[2])")
    println()

    @showprogress 1 "Computing..." for n in 3:1:N
        for j in 2:1:J-1
            # models
            if model == "phi4"
                F[j,n] = ((F[j+1,n-1]-2*F[j,n-1]+F[j-1,n-1])/(dx^2) + 2*(1-F[j,n-1]^2)*F[j,n-1])*dt^2 + 2*F[j,n-1] - F[j,n-2]
            elseif model == "phi6"
                F[j,n] = ((F[j+1,n-1]-2*F[j,n-1]+F[j-1,n-1])/(dx^2) - F[j,n-1]*(1-F[j,n-1]^2)^2 + 2*(1-F[j,n-1]^2)*F[j,n-1]^3 )*dt^2 + 2*F[j,n-1] - F[j,n-2]
            elseif model == "sG"
                F[j,n] = ((F[j+1,n-1]-2*F[j,n-1]+F[j-1,n-1])/(dx^2) - sin(F[j,n-1]) )*dt^2 + 2*F[j,n-1] - F[j,n-2]
            end
        
            if bcs == "absorbent"
                F[1,n] = (1/(dx+dt))*(dx*F[1,n-1] + dt*F[2,n])
                F[end,n] = (1/(dx+dt))*(dt*F[end-1,n] + dx*F[end,n-1])
            elseif bcs == "periodic"
                F[1,n] = F[end-1,n]
                F[end,n] = F[2,n]
            end
        end
    end

    #----------- data saving

    if output_format == "jld2"
        path = out*"/kak_ff_$(model)_v=$(v0).jld2"
        @save path F Jarr Narr v0

    elseif output_format == "npy"
        npzwrite(out*"/kak_ff_$(model)_v=$(v0).npy", F)
    end

    println()
    println("Data saved at "*out )
    println()
    println("#--------------------------------------------------#")
    print()

end

function ff_kakkak(model::String,ff_space,ff_time,incs,bcs::String,out::String,output_format::String)
    # ff_space : Vector{Float64} : spatial dimension : [xi,xf,dx]
    # ff_time : Vector{Float64} : temporal dimension : [tf,dt]
    # incs : Vector{Float64} : initial conditions : [a0,v0]
    # out : PATH : path of output folder

    # PENDING: Does not work with sG

    if (output_format != "jld2") && (output_format != "npy")
        println("invalid output data type")
        return
    end

    # unpacking
    xi = ff_space[1]
    xf = ff_space[2]
    dx = ff_space[3]

    tf = ff_time[1]
    dt = ff_time[2]

    J = convert(Int64,(xf-xi)/dx)
    N = convert(Int64,tf/dt)

    Jarr = range(xi,xf-dx; step=dx)
    Narr = range(0.0,tf-dt; step=dt)

    # initialization
    F = zeros(Float64, J,N)

    a0 = incs[1]
    v0 = incs[2]
    gamma = 1.0/sqrt(1.0-(v0^2))
    v02 = v0
    gamma2 = 1.0/sqrt(1.0-(v02)^2)

    #---------- initial conditions
    
    # temporal ic: initial configuration
    for j in 1:1:J
        x = xi + j*dx

        if model == "phi4"
            F[j,1] = tanh(gamma*(x+a0)) + tanh(gamma2*(x-a0/2.)) - tanh(gamma*(x-a0)) - tanh(gamma2*(x+a0/2)) - 1 
            F[j,2] = F[j,1] - ( ((gamma*v0)/(cosh(gamma*(x+a0)))^2 - (gamma2*v02)/(cosh(gamma2*(x+a0/2)))^2 + (gamma*v0)/(cosh(gamma*(x-a0)))^2 - (gamma2*v02)/(cosh(gamma2*(x-a0/2)))^2)*dt )
        end
        if model == "phi6" # PENDING
            F[j,1] = 0
            F[j,2] = F[j,1] + 0
        end
        if model == "sG"
            F[j,1] = 4*( atan(exp(gamma*(x+a0))) - atan(exp(gamma*(x-a0))) )
            F[j,2] = F[j,1] + 4*( -(exp(gamma*(x+a0)*gamma*v0))/(1+exp(2*gamma*(x+a0))) - (exp(gamma*(x-a0)*gamma*v0))/(1+exp(2*gamma*(x-a0)))  )*dt 
        end
    end

    # spatial ic: assymptotic value
    if model == "phi4"
        F[1,:] .= -1.0
        F[end,:] .= -1.0
    end
    if model == "phi6" # PENDING
        F[1,:] .= 0
        F[end,:] .= 0
    end
    if model == "sG"
        F[1,:] .= 0.0
        F[end,:] .= 0.0
    end

    #---------- time evolution
   
    println()
    println("#--------------------------------------------------#")
    println()
    println("Full field dynamics: model=$(model), v0=$(incs[2])")
    println()

    @showprogress 1 "Computing..." for n in 3:1:N
        for j in 2:1:J-1
            # models
            if model == "phi4"
                F[j,n] = ((F[j+1,n-1]-2*F[j,n-1]+F[j-1,n-1])/(dx^2) + 2*(1-F[j,n-1]^2)*F[j,n-1])*dt^2 + 2*F[j,n-1] - F[j,n-2]
            elseif model == "phi6"
                F[j,n] = ((F[j+1,n-1]-2*F[j,n-1]+F[j-1,n-1])/(dx^2) - F[j,n-1]*(1-F[j,n-1]^2)^2 + 2*(1-F[j,n-1]^2)*F[j,n-1]^3 )*dt^2 + 2*F[j,n-1] - F[j,n-2]
            elseif model == "sG"
                F[j,n] = ((F[j+1,n-1]-2*F[j,n-1]+F[j-1,n-1])/(dx^2) - sin(F[j,n-1]) )*dt^2 + 2*F[j,n-1] - F[j,n-2]
            end
        
            if bcs == "absorbent"
                F[1,n] = (1/(dx+dt))*(dx*F[1,n-1] + dt*F[2,n])
                F[end,n] = (1/(dx+dt))*(dt*F[end-1,n] + dx*F[end,n-1])
            elseif bcs == "periodic"
                F[1,n] = F[end-1,n]
                F[end,n] = F[2,n]
            end
        end
    end

    #----------- data saving

    if output_format == "jld2"
        path = out*"/kak_ff_$(model)_v=$(v0).jld2"
        @save path F Jarr Narr v0

    elseif output_format == "npy"
        npzwrite(out*"/kak_ff_$(model)_v=$(v0).npy", F)
    end

    println()
    println("Data saved at "*out )
    println()
    println("#--------------------------------------------------#")
    print()

end



#---------------- x=0 dynamics

function ff_origin(model,ff_space,ff_time,bcs::String,incs,out::String)
    # ff_space : Vector{Float64} : spatial dimension : [xi,xf,dx]
    # ff_time : Vector{Float64} : temporal dimension : [tf,dt]
    # incs : Vector{Float64} : initial conditions : [a0,v0]
    # out : PATH : path of output folder

    # unpacking
    xi = ff_space[1]
    xf = ff_space[2]
    dx = ff_space[3]

    tf = ff_time[1]
    dt = ff_time[2]

    J = convert(Int64,(xf-xi)/dx)
    N = convert(Int64,tf/dt)

    Jarr = range(xi,xf-dx; step=dx)
    Narr = range(0.0,tf-dt; step=dt)

    # initialization
    F = zeros(Float64, J,N)

    a0 = incs[1]
    v0 = incs[2]
    gamma = 1.0/sqrt(1.0-(v0^2))

    #---------- initial conditions
    
    # temporal ic: initial configuration
    for j in 1:1:J
        x = xi + j*dx
        F[j,1] = tanh(gamma*(x+a0)) - tanh(gamma*(x-a0)) - 1 # field
        F[j,2] = F[j,1] - ( ((gamma*v0)/(cosh(gamma*(x+a0)))^2 + (gamma*v0)/(cosh(gamma*(x-a0)))^2 )*dt ) # derivative
    end

    # spatial ic: assymptotic value
    F[1,:] .= -1.0
    F[end,:] .= -1.0

    #---------- time evolution
    
    for n in 3:1:N
        for j in 2:1:J-1
            
            # models
            if model == "phi4"
                F[j,n] = ((F[j+1,n-1]-2*F[j,n-1]+F[j-1,n-1])/(dx^2) + 2*(1-F[j,n-1]^2)*F[j,n-1])*dt^2 + 2*F[j,n-1] - F[j,n-2]
            end
            if model == "phi6"
                F[j,n] = 0
            end
            
            if bcs == "absorbent"
                F[1,n] = (1/(dx+dt))*(dx*F[1,n-1] + dt*F[2,n])
                F[end,n] = (1/(dx+dt))*(dt*F[end-1,n] + dx*F[end,n-1])
            elseif bcs == "periodic"
                F[1,n] = F[end-1,n]
                F[end,n] = F[2,n]
            end
        end    
    end 

    #---------- x=0 extraction

    x0 = convert(Int64, J/2)

    F0 = zeros(Float64, N)
    for t in 1:1:N
        F0[t] = F[x0,t]
    end

    #---------- data saving (jdl2)
    #=
    path = out*"/kak_ff_origin_v=$(v0).jld2"
    @save path F0 x0 v0
    println("data saved at "*path )
    =#
    #---------- data saving (npy)
 
    path = out*"/kak_ff_origin_v=$(v0).npy"
    npzwrite(path, F0)
    #println("data saved at "*path )
end

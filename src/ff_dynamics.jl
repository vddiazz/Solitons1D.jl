using LinearAlgebra
using JLD2

### kak dynamics

function ff_kak(model,ff_space,ff_time,incs,out)
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
    v0 = -incs[2]
    gamma = 1.0/sqrt(1.0-(v0^2))

    #---------- initial conditions
    
    # temporal ic: initial configuration
    for j in 1:1:J
        x = xi + j*dx
        F[j,1] = tanh(gamma*(x+a0)) - tanh(gamma*(x-a0)) - 1 # field
        F[j,2] = F[j,1] +( ((gamma*v0)/(cosh(gamma*(x+a0)))^2 + (gamma*v0)/(cosh(gamma*(x-a0)))^2 )*dt ) # derivative
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
            
            # absorbent boundary conditions
            F[1,n] = F[2,n-1] + (dt-dx)/(dt+dx)*(F[2,n]-F[1,n-1])
            F[end,n] = F[end-1,n-1] + (dt-dx)/(dt+dx)*(F[end-1,n]-F[end,n-1])
        end

        println("done: t = $(round(n*dt,digits=3))")
    end

    #---------- data saving

    path = out*"/kak_ff_v=$(-v0).jld2"
    @save path F Jarr Narr v0
    println("data saved at "*path )

end

### collision plot

using LinearAlgebra
using JLD2
using Plots
using SpecialFunctions

#-------------------- moduli kak field

function kak_phi4(x,X)

    a = X[1]; b = X[2]

    F = tanh(x+a) - tanh(x-a) - 1 + (b/tanh(a))*(sinh(x+a)/(cosh(x+a)^2) - sinh(x-a)/(cosh(x-a)^2))

    return F
end

function profile_kak_m2(space,model,a,b)
    
    Jarr = space
    
    J = length(Jarr)

    F = zeros(Float64, J)

    if model == "phi4"
        for j in 1:1:J
            F[j] = tanh(Jarr[j]+a) - tanh(Jarr[j]-a) - 1 + (b/tanh(a))*(sinh(Jarr[j]+a)/(cosh(Jarr[j]+a)^2) - sinh(Jarr[j]-a)/(cosh(Jarr[j]-a)^2))
        end
    end

    return F

end

#-------------------- energy

function energy_m2(field_m,field,field_p,out)
    # field_m : field profile at t-1
    # field : field profile at t
    # field_p : field profile at t+1

    dt = 0.0001
    dx = 0.01

    Dx = zeros(Float64, length(field)-2)
    for j in 1:1:length(Dx)
        Dx[j] = (field[j+2]-field[j])/(2*dx)
     end

    Dt = (field_p - field_m)/(2*dt)
    deleteat!(Dt,1)
    deleteat!(Dt,length(Dt))
    
    Edens = zeros(Float64, length(Dx))

    for j in 1:1:length(Dx)
        Edens[j] = 0.5*Dt[j]^2 - 0.5*Dx[j]^2 - (1-field[j]^2)^2
    end

    E = sum(Edens)*dx

    return E
end

#-------------------- Lagrange interpolation

function lip(degree::Int64, x::Float64, xs::Array{Float64}, ys::Array{Float64})::Float64

    lx = length(xs)
    ly = length(ys)

    @assert lx == ly "xs and ys must have same length"

    P = 0
    for j in 1:degree+1
        lj = 1.0
        for m in 1:degree+1
            if m == j
                continue
            end
            denom = xs[j] - xs[m]
            lj = lj*(x-xs[m])/denom
         end
         P = P + ys[j]*lj
    end

    return P
end



    



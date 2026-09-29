using LinearAlgebra
using JLD2
using Plots
using SpecialFunctions
using LoopVectorization

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

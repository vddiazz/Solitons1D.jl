using LinearAlgebra
using JLD2
using Plots
using SpecialFunctions
using LoopVectorization


function lip(degree::Int64, x::Float64, xs::Array{Float64}, ys::Array{Float64})::Float64
    # Lagrange quadratic interpolation

    lx = length(xs)
    ly = length(ys)

    @assert lx == ly "xs and ys must have same length"

    P = 0
    @inbounds @fastmath for j in 1:degree+1
        lj = 1.0
        for m in 1:degree+1
            if m == j
                continue
            end
            D = xs[j] - xs[m]
            lj = lj*(x-xs[m])/D
         end
         P = P + ys[j]*lj
    end

    return P
end

function spline_1d(x::Array{Float64},y::Array{Float64}) 
    # x : moduli direction
    # y : coefficient slice y(x) 

    lx = length(x)
    h = diff(x)

    slope = zeros(Float64, lx)
    @tturbo for i in 2:lx-1
        slope[i] = 3*( (y[i+1]-y[i])/h[i] - (y[i] - y[i-1])/h[i-1] )
    end

    l = ones(Float64, lx)
    mu = zeros(Float64, lx)
    z = zeros(Float64, lx)

    @tturbo for i in 2:lx-1
        l[i] = 2*(x[i+1] - x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (slope[i] - h[i-1]*z[i-1])/l[i]
    end

    c1 = zeros(Float64, lx-1)
    c2 = zeros(Float64, lx)
    c3 = zeros(Float64, lx-1)

    for j in lx-1:-1:1
        c2[j] = z[j] - mu[j]*c2[j+1]
        c1[j] = (y[j+1]-y[j])/h[j] - h[j]*(2*c2[j] + c2[j+1])/3
        c3[j] = (c2[j+1]-c2[j]) / (3*h[j])
    end

    return c1,c2,c3
end

function spline_eval(x::Array{Float64},y::Array{Float64},c1::Array{Float64},c2::Array{Float64},c3::Array{Float64},x0) 

    lx = length(x)

    idx = searchsortedlast(x,x0)
    idx = clamp(idx,1,lx-1)

    dx = x0-x[idx]
    
    return y[idx] + c1[idx]*dx + c2[idx]*dx^2 + c3[idx]* dx^3
end

function spline_2d(X1::Array{Float64}, X2::Array{Float64}, coef::Array{Float64}, M0::Array{Float64})

    M1 = M0[1]; M2 = M0[2]

    l2 = length(X2)
    g = zeros(Float64, l2)

    # spline in X1 for each column j
    @inbounds @fastmath for j in 1:l2
        y = coef[:,j]
        c1,c2,c3 = spline_1d(X1,y)
        g[j] = spline_eval(X1,y,c1,c2,c3,M1)
    end

    # spline in X2 for the intermediate results
    c1_2, c2_2, c3_2 = spline_1d(X2,g)
    return spline_eval(X2,g,c1_2,c2_2,c3_2,M2)
end

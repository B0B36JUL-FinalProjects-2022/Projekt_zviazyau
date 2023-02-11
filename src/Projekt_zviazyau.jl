module Projekt_zviazyau

export SimpleNet,train

include("utils.jl")


using LinearAlgebra
using Colors, Plots
using Statistics


# ReLU activation function
function ReLU(Z)
    return max.(0,Z)
end

# Derivative of ReLU activation function
function d_ReLU(Z)
    return Z .> 0
end

# id activation function

function id(Z)
    return Z
end



# Simple Neural Network with 1 hidden dense layer
struct SimpleNet{T<:Real}
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
end

# Constructor for SimpleNet function
SimpleNet(n1, n2, n3) = SimpleNet(randn(n2, n1), randn(n2), randn(n3, n2), randn(n3))

# SimpleNet prediction functor
function (m::SimpleNet)(x)
    z1 = m.W1*x .+ m.b1
    a1 = ReLU(z1);
    z2 = m.W2*a1 .+ m.b2
    a2 = id(z2)
    return a2
end

function simplenet_grad(m::SimpleNet, x::AbstractVector, y; ϵ=1e-10)
    z1 = m.W1*x .+ m.b1
    a1 = ReLU(z1)
    z2 = m.W2*a1 .+ m.b2
    a2 = id(z2)

    # Place for loss function
    # l = -sum(y .* log.(a2 .+ ϵ))

    # e_z2 = exp.(z2)
    # l_part = (- e_z2 * e_z2' + Diagonal(e_z2 .* sum(e_z2))) / sum(e_z2)^2

    d_z2 = a2 - y
    d_w2 = d_z2 * a1'
    d_b2 = d_z2
    d_z1 = m.W2' * d_z2 .* d_ReLU(z1)
    d_w1 = d_z1 * x'
    d_b1 = d_z1


    # l_a2 = - y ./ (a2 .+ ϵ)
    # l_z2 = l_a2 # l_part * l_a2
    # l_a1 = m.W2' * l_z2
    # l_z1 = l_a1 .* (a1 .> 0)
    # l_x = m.W1' * l_z1

    # l_W2 = l_z2 * a1'
    # l_b2 = l_z2
    # l_W1 = l_z1 * x'
    # l_b1 = l_z1

    # return l, l_W1, l_b1, l_W2, l_b2
    return d_w1, d_b1, d_w2, d_b2
end

function mean_tuple(d::AbstractArray{<:Tuple})
    Tuple([mean([d[k][i] for k in 1:length(d)]) for i in 1:length(d[1])])
end


function train(m,X_train,y_train)
    alpha = 1e-2
    max_iter = 2000
    # L = zeros(max_iter)
    for iter in 1:max_iter
        grad_all = [simplenet_grad(m, X_train[:,k], y_train[:,k]) for k in 1:size(X_train,2)]
        grad_mean = mean_tuple(grad_all)

        # L[iter] = grad_mean[1]

        m.W1 .-= alpha*grad_mean[1]
        m.b1 .-= alpha*grad_mean[2]
        m.W2 .-= alpha*grad_mean[3]
        m.b2 .-= alpha*grad_mean[4]
    end
end

end # module Projekt_zviazyau

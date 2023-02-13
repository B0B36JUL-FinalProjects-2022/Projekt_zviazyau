module Projekt_zviazyau

export SimpleNet,train #, simplenet_grad, ReLU, d_ReLU, id, d_id, MSE, d_MSE

include("utils.jl")

using LinearAlgebra
using Colors, Plots
using Statistics

# ReLU activation function
ReLU(Z) = max.(0,Z)

# Derivative of ReLU activation function
d_ReLU(Z) =  Z .> 0

# id activation function
id(Z) = Z

# Derivative of id function
d_id(Z) = ones(size(Z))

# Mean Squared Error loss function
MSE(a2,y) = sum((a2 .- y ).^2,dims=2) / size(y,2)

# Derivative of MSE function
d_MSE(a2,y) = (2 *(a2 .- y))

# Simple Neural Network with 1 hidden dense layer
struct SimpleNet{T<:Real}
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
end

# Constructor for SimpleNet function with normal distribution initialization
# SimpleNet(n1, n2, n3) = SimpleNet(randn(n2, n1), randn(n2), randn(n3, n2), randn(n3))

# Constructor for SimpleNet function with Xavier initialization
using Distributions
d = Truncated(Normal(0, 1), -sqrt(6/(9216+30)), sqrt(6/(9216+30)))  #Construct the distribution type
SimpleNet(n1, n2, n3) = SimpleNet(rand(d,n2, n1), rand(d,n2), rand(d,n3, n2), rand(d,n3))

# SimpleNet prediction functor
function (m::SimpleNet)(x)
    z1 = m.W1*x .+ m.b1
    a1 = ReLU(z1)
    z2 = m.W2*a1 .+ m.b2
    a2 = id(z2)
    return a2
end

function simplenet_grad(m::SimpleNet, x, y; ϵ=1e-10)
    samples_amount = size(y,2) # SA

    z1 = m.W1*x .+ m.b1
    #100xSA =  100x9216 * 9216xSA .+ 100xSA

    a1 = ReLU(z1)
    # 100xSA = 100xSA

    z2 = m.W2*a1 .+ m.b2
    #30xSA = 30x100 * 100xSA + 30xSA

    a2 = id(z2)
    #30xSA = 30xSA

    # Place for loss function
    loss = MSE(a2,y)

    d_a2 = d_MSE(a2,y)
    # 30xSA = 30xSA

    d_z2 = d_a2 .* d_id(a2)
    # 30xSA = 30xSA .* 30xSA

    d_a1 = m.W2' * d_z2
    # 100xSA = 100x30 * 30xSA

    d_z1 = d_a1 .* d_ReLU(z1)
    # 100xSA = 100xSA .* 100xSA

    d_w2 = (d_z2 * a1') / samples_amount
    # 30x100 = 30xSA * SAx100
    d_b2 = sum(d_z2,dims=2) / samples_amount
    # 30x1 = 30x1
    d_w1 = (d_z1 * x') / samples_amount
    # 100x9216 = 100xSA * SAx9216
    d_b1 = sum(d_z1,dims=2) / samples_amount
    # 100x1 = 100x1

    return d_w1, d_b1, d_w2, d_b2, loss
end

function train(m,X_train,y_train;epoch = 1000)
    alpha = 1e-5
    L = zeros(epoch)

    println("MSE Start: ", sum(MSE(m(X_train),y_train)))
    for cur_epoch in 1:epoch
        grad = simplenet_grad(m,X_train,y_train)
        L[cur_epoch] = sum(grad[5])

        m.W1 .-= alpha*grad[1]
        m.b1 .-= alpha*grad[2]
        m.W2 .-= alpha*grad[3]
        m.b2 .-= alpha*grad[4]

        if cur_epoch % 50 == 0 || cur_epoch == 1
            println("MSE: ", L[cur_epoch])
        end
    end
end

end # module Projekt_zviazyau

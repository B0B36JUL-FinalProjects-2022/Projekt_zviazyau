export MediumNet, train

using LinearAlgebra
using Colors, Plots

""" Simple Neural Network with 1 hidden dense layer """
struct MediumNet{T<:Real}
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
    W3::Matrix{T}
    b3::Vector{T}
end

""" Constructor for MediumNet function with Xavier initialization. """
function MediumNet(n1, n2, n3, n4)
    d = xavier_init(n1,n4)
    net = MediumNet(rand(d,n2, n1), rand(d,n2), rand(d,n3, n2), rand(d,n3),rand(d,n4, n3),rand(d,n4))
    return net
end

""" MediumNet prediction functor. """
function (m::MediumNet)(x)
    z1 = m.W1*x .+ m.b1
    a1 = ReLU(z1)
    z2 = m.W2*a1 .+ m.b2
    a2 = softplus(z2)
    z3 = m.W3*a2 .+ m.b3
    a3 = id(z3)
    return a3
end

"""Backward propagation, searching for gradient for Gradient Descent method."""
function gradient(m::MediumNet, x, y; ϵ=1e-10)
    samples_amount = size(y,2)

    z1 = m.W1*x .+ m.b1
    a1 = ReLU(z1) # 1st activation function is ReLU
    z2 = m.W2*a1 .+ m.b2
    a2 = softplus(z2) # 2nd activation function is softplus
    z3 = m.W3*a2 .+ m.b3
    a3 = id(z3) # 3rd activation function is identity

    # Loss function is MSE
    loss = MSE(a3,y)

    d_a3 = d_MSE(a3,y)
    d_z3 = d_a3 .* d_id(a3)
    d_a2 = m.W3' * d_z3
    d_z2 = d_a2 .* d_softplus(a2)
    d_a1 = m.W2' * d_z2
    d_z1 = d_a1 .* d_ReLU(z1)
    d_w3 = (d_z3 * a2') / samples_amount
    d_b3 = sum(d_z3,dims=2) / samples_amount
    d_w2 = (d_z2 * a1') / samples_amount
    d_b2 = sum(d_z2,dims=2) / samples_amount
    d_w1 = (d_z1 * x') / samples_amount
    d_b1 = sum(d_z1,dims=2) / samples_amount

    return loss, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3
end

function updata_parameters(m::MediumNet,grad,alpha)
    # Update parameters
    m.W1 .-= alpha*grad[2]
    m.b1 .-= alpha*grad[3]
    m.W2 .-= alpha*grad[4]
    m.b2 .-= alpha*grad[5]
    m.W3 .-= alpha*grad[6]
    m.b3 .-= alpha*grad[7]
end
export SimpleNet

using LinearAlgebra
using Colors, Plots

""" Simple Neural Network with 1 hidden dense layer """
struct SimpleNet{T<:Real}
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
end

""" Constructor for SimpleNet function with Xavier initialization. """
function SimpleNet(n1, n2, n3)
    d = xavier_init(n1,n3)
    net = SimpleNet(rand(d,n2, n1), rand(d,n2), rand(d,n3, n2), rand(d,n3))
    return net
end

""" SimpleNet prediction functor. """
function (m::SimpleNet)(x)
    z1 = m.W1*x .+ m.b1
    a1 = ReLU(z1)
    z2 = m.W2*a1 .+ m.b2
    a2 = id(z2)
    return a2
end

"""Backward propagation, searching for gradient for Gradient Descent method."""
function gradient(m::SimpleNet, x, y; ϵ=1e-10)
    samples_amount = size(y,2)

    z1 = m.W1*x .+ m.b1
    a1 = ReLU(z1) # 1st activation function is ReLU
    z2 = m.W2*a1 .+ m.b2
    a2 = id(z2) # 2nd activation function is id

    # Loss function is MSE
    loss = MSE(a2,y)

    d_a2 = d_MSE(a2,y)
    d_z2 = d_a2 .* d_id(a2)
    d_a1 = m.W2' * d_z2
    d_z1 = d_a1 .* d_ReLU(z1)

    d_w2 = (d_z2 * a1')
    d_b2 = sum(d_z2,dims=2)
    d_w1 = (d_z1 * x')
    d_b1 = sum(d_z1,dims=2)

    return loss,d_w1, d_b1, d_w2, d_b2
end

function updata_parameters(m::SimpleNet,grad,alpha)
    m.W1 .-= alpha*grad[2]
    m.b1 .-= alpha*grad[3]
    m.W2 .-= alpha*grad[4]
    m.b2 .-= alpha*grad[5]
end
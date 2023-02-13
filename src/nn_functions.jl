export xavier_init, ReLU, d_ReLU, id, d_id, softplus, d_softplus, MSE, d_MSE

using Distributions

""" Normal distribution for Xavier initialization. """
xavier_init(n_in,n_out; mu = 0, sigma = 1) = Truncated(Normal(mu, sigma), -sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out)))

""" ReLU activation function. """
ReLU(Z) = max.(0,Z)

""" Derivative of ReLU activation function. """
d_ReLU(Z) =  Z .> 0

""" Identity activation function. """
id(Z) = Z

""" Derivative of identity function. """
d_id(Z) = ones(size(Z))

""" Softplus activation function. """
softplus(Z) = log.(1 .+ exp.(Z))

""" Derivative of softplus activation function. """
d_softplus(Z) = 1 ./ (1 .+ exp.(-Z))


""" Mean Squared Error loss function. """
MSE(a2,y) = sum((a2 .- y ).^2,dims=2) / size(y,2)

""" Derivative of MSE function. """
d_MSE(a2,y) = (2 *(a2 .- y))

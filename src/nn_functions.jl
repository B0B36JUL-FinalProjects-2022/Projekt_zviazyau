export xavier_init, ReLU, d_ReLU, id, d_id, MSE, d_MSE

using Distributions

""" Normal distribution for Xavier initialization. """
xavier_init(n_in,n_out; mu = 0, sigma = 1) = Truncated(Normal(mu, sigma), -sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out)))

""" ReLU activation function. """
ReLU(Z) = max.(0,Z)

""" Derivative of ReLU activation function. """
d_ReLU(Z) =  Z .> 0

""" id activation function. """
id(Z) = Z

""" Derivative of id function. """
d_id(Z) = ones(size(Z))

""" Mean Squared Error loss function. """
MSE(a2,y) = sum((a2 .- y ).^2,dims=2) / size(y,2)

""" Derivative of MSE function. """
d_MSE(a2,y) = (2 *(a2 .- y))

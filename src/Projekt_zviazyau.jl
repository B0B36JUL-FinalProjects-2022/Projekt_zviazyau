module Projekt_zviazyau

include("utils.jl")
include("nn_functions.jl")
include("simplenet.jl")
include("mediumnet.jl")

export train

""" Training neural network via Gradient Descent. """ 
function train(m,X_train,y_train;epoch = 1000, alpha = 1e-3)
    L = zeros(epoch)

    for cur_epoch in 1:epoch
        grad = gradient(m,X_train,y_train)
        L[cur_epoch] = sum(grad[1])

        updata_parameters(m,grad,alpha)

        # Print simple training statistics
        if cur_epoch % 50 == 0 || cur_epoch == 1
            println("MSE: ", L[cur_epoch])
        end
    end
end

end

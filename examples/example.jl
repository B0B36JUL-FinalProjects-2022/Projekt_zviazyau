# Basic script
using Revise
using Projekt_zviazyau

using FileIO,JLD2

train_path = "/home/ezvezdov/Programming/Projekt_zviazyau/kaggle_data/training.csv"
test_path = "/home/ezvezdov/Programming/Projekt_zviazyau/kaggle_data/test.csv"

X_train,y_train = transform_train_data(train_path)
# preview_image(X_train,y_train,2)

# simplenet = load("example.jld2","simplenet")
simplenet = SimpleNet(size(X_train,1), 100, size(y_train,1))
train(simplenet,X_train,y_train;epoch = 400)

simplenet

new_y = simplenet(X_train)
preview_image(X_train,new_y,5)

jldsave("example3.jld2"; simplenet)


X_test = transform_test_data(test_path)
y_test = nothing
for img_num in 1:size(X_test,2)
    global  y_test
    if y_test == nothing
        y_test = simplenet(X_test[:,img_num])
    else
        y_test = hcat(y_test,simplenet(X_test[:,img_num]))
    end
end

image_num = 4
preview_image(X_test,y_test,image_num)


# # initialize weights and biases
# W1 = randn(9216, 100)
# b1 = zeros(1, 100)
# W2 = randn(100, 30)
# b2 = zeros(1, 30)

# # set learning rate and number of iterations
# learning_rate = 0.01
# num_iterations = 1000
# X = X_train'
# Y = y_train'

# # loop over the specified number of iterations
# for i = 1:num_iterations
#     # forward propagation
#     Z1 = X * W1 .+ b1
#     A1 = max.(0, Z1)
#     Z2 = A1 * W2 .+ b2
#     A2 = Z2
    
#     # compute cost
#     cost = sum((A2 - Y).^2)
    
#     # backward propagation
#     dZ2 = 2 * (A2 - Y)
#     dW2 = A1' * dZ2
#     db2 = sum(dZ2, dims=1)
#     dA1 = dZ2 * W2'
#     dZ1 = dA1 .* (Z1 .> 0)
#     dW1 = X' * dZ1
#     db1 = sum(dZ1, dims=1)
    
#     # update parameters
#     W1 = W1 .- learning_rate * dW1
#     b1 = b1 .- learning_rate * db1
#     W2 = W2 .- learning_rate * dW2
#     b2 = b2 .- learning_rate * db2
    
#     # print cost every 100 iterations
#     if i % 100 == 0
#         println("Cost after iteration $i: $cost")
#     end
# end

# using Colors
# inessa = FileIO.load("/home/ezvezdov/Programming/Projekt_zviazyau/kaggle_data/inessa.png")
# styopa = FileIO.load("/home/ezvezdov/Programming/Projekt_zviazyau/kaggle_data/styopa.png")
# zhenya = FileIO.load("/home/ezvezdov/Programming/Projekt_zviazyau/kaggle_data/zhenya.png")
# img = styopa
# X = Float32.(Gray.(img))
# X2 = reshape(X,9216)
# y_test = simplenet(X2)
# preview_image(X2,y_test,1)

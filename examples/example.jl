# Basic script
using Revise
using Projekt_zviazyau

using FileIO,JLD2

train_path = "/home/ezvezdov/Programming/Projekt_zviazyau/kaggle_data/training.csv"
test_path = "/home/ezvezdov/Programming/Projekt_zviazyau/kaggle_data/test.csv"

X_train,y_train = transform_train_data(train_path)
# preview_image(X_train,y_train,2)

simplenet = load("example.jld2","simplenet")
# simplenet = SimpleNet(size(X_train,1), 100, size(y_train,1))
# train(simplenet,X_train,y_train)


X_test = transform_test_data(test_path)
y_test = simplenet(X_test[:,1])
y_test = nothing
for img_num in 1:size(X_test,2)
    if typeof(y_test) == nothing
        y_test = simplenet(X_test[:,img_num])
        continue
    end

    y_test = hcat(y_test,simplenet(X_test[:,img_num]))
end

image_num = 2
preview_image(X_test,y_test,image_num)

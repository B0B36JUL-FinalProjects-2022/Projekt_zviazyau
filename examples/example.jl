using Revise
using Projekt_zviazyau

using FileIO,JLD2

# Download data from and extract to /kaggle_data
# https://www.kaggle.com/competitions/facial-keypoints-detection/data


# Set 
train_path = "kaggle_data/training.csv"
test_path = "kaggle_data/test.csv"
IdLookupTable_path = "kaggle_data/IdLookupTable.csv"

# Read data from CSV
# X11_train,y11_train, keys11, X4_train,y4_train, keys4 = transform_train_data(train_path)
# X_test = transform_test_data(test_path)

# Read data from saved jld (kaggle_data folder)
X11_train ,y11_train, keys11, X4_train,y4_train, keys4 = load("kaggle_data/train_data.jld2","X11_train","y11_train", "keys11", "X4_train","y4_train", "keys4")
X_test = load("kaggle_data/test_data.jld2","X_test")

# Save data to jld to kaggle_data folder
jldsave("kaggle_data/test_data.jld2"; X_test)
jldsave("kaggle_data/train_data.jld2"; X11_train,y11_train,keys11,X4_train,y4_train,keys4)

# SIMPLENET
#########################################################################
# Simplenet for 11 keypoinst
simplenet11 = load("saved_networks/simplenet11.jld2","simplenet11")
simplenet11 = SimpleNet(size(X11_train,1), 100, size(y11_train,1))
println("Training simplenet11")
train(simplenet11,X11_train,y11_train;epoch = 400,alpha = 1e-3)
jldsave("saved_networks/simplenet11.jld2"; simplenet11)

# Simplenet for 4 keypoinst
simplenet4 = load("saved_networks/simplenet4.jld2","simplenet4")
simplenet4 = SimpleNet(size(X4_train,1), 100, size(y4_train,1))
println("Training simplenet4")
train(simplenet4,X4_train,y4_train;epoch = 400,alpha = 1e-3)
jldsave("saved_networks/simplenet4.jld2"; simplenet4)

# Make prediction 
y4_test = simplenet4(X_test)
y11_test = simplenet11(X_test)
#########################################################################

# MEDIUMNET
#########################################################################
# Mediumnet for 11 keypoinst
mediumnet11 = load("saved_networks/mediumnet11.jld2","mediumnet11")
mediumnet11 = MediumNet(size(X11_train,1), 200, 100, size(y11_train,1))
println("Training mediumnet11")
train(mediumnet11,X11_train,y11_train;epoch = 400,alpha = 1e-3)
jldsave("saved_networks/mediumnet11.jld2"; mediumnet11)

# Mediumnet for 4 keypoinst
mediumnet4 = load("saved_networks/mediumnet4.jld2","mediumnet4")
mediumnet4 = SimpleNet(size(X4_train,1), 100, size(y4_train,1))
println("Training mediumnet4")
train(mediumnet4,X4_train,y4_train;epoch = 400,alpha = 1e-3)
jldsave("saved_networks/mediumnet4.jld2"; mediumnet4)

# Make prediction 
y4_test = mediumnet4(X_test)
y11_test = mediumnet11(X_test)
#########################################################################



# Merge data
y_test, keys_all = merge_data(y4_test,keys4, y11_test,keys11)

# Make submission
make_submission(IdLookupTable_path,y_test,keys_all)

# Preview 10th image from testing set with predicted keypoints
image_num = 10
preview_image(X_test,y_test,image_num)

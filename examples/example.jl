using Revise
using Projekt_zviazyau
using FileIO,JLD2

# Download data from 
# https://www.kaggle.com/competitions/facial-keypoints-detection/data
# and extract to /kaggle_data.
# test.zip and training.zip should be extracted to kaggle_data/ too.


# Set file pathes
train_path = "kaggle_data/training.csv"
test_path = "kaggle_data/test.csv"
IdLookupTable_path = "kaggle_data/IdLookupTable.csv"

# Read train/test data from CSV
X11_train,y11_train, keys11, X4_train,y4_train, keys4 = transform_train_data(train_path)
X_test = transform_test_data(test_path)

# Load train/test data from saved jld (kaggle_data folder)
# X11_train ,y11_train, keys11, X4_train,y4_train, keys4 = load("kaggle_data/train_data.jld2","X11_train","y11_train", "keys11", "X4_train","y4_train", "keys4")
# X_test = load_object("kaggle_data/test_data.jld2")

# Save training/testing data to kaggle_data folder
# save_object("kaggle_data/test_data.jld2", X_test)
# jldsave("kaggle_data/train_data.jld2"; X11_train,y11_train,keys11,X4_train,y4_train,keys4)


# SIMPLENET
#########################################################################
# Simplenet for 11 keypoinst

simplenet11 = SimpleNet(size(X11_train,1), 100, size(y11_train,1))
# simplenet11 = load_object("saved_networks/simplenet11.jld2")
train(simplenet11,X11_train,y11_train;epoch = 400,alpha = 1e-3)
# save_object("saved_networks/simplenet12.jld2",simplenet11)

# Simplenet for 4 keypoinst

simplenet4 = SimpleNet(size(X4_train,1), 100, size(y4_train,1))
# simplenet4 = load_object("saved_networks/simplenet4.jld2")
train(simplenet4,X4_train,y4_train;epoch = 400,alpha = 1e-3)
# save_object("saved_networks/simplenet4.jld2",simplenet4)

# Make prediction 
y4_test = simplenet4(X_test)
y11_test = simplenet11(X_test)
#########################################################################


# MEDIUMNET
#########################################################################
# Mediumnet for 11 keypoinst

mediumnet11 = MediumNet(size(X11_train,1), 200, 100, size(y11_train,1))
# mediumnet11 = load_object("saved_networks/mediumnet11.jld2")
train(mediumnet11,X11_train,y11_train;epoch = 400,alpha = 1e-3)
# save_object("saved_networks/mediumnet11.jld2",mediumnet11)

# Mediumnet for 4 keypoinst

mediumnet4 = MediumNet(size(X4_train,1), 200,100, size(y4_train,1))
# mediumnet4 = load_object("saved_networks/mediumnet4.jld2")
train(mediumnet4,X4_train,y4_train;epoch = 400,alpha = 1e-3)
# save_object("saved_networks/mediumnet4.jld2", mediumnet4)

# Make prediction 
y4_test = mediumnet4(X_test)
y11_test = mediumnet11(X_test)
#########################################################################



# Merge data
y_test, keys_all = merge_data(y4_test,keys4, y11_test,keys11)

# Make submission
make_submission(IdLookupTable_path,y_test,keys_all)

# Preview images from testing set with predicted keypoints
preview_image(X_test,y_test,1)
preview_image(X_test,y_test,315)
preview_image(X_test,y_test,786)
preview_image(X_test,y_test,666)

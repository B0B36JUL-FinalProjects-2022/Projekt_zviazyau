# /examples/example.jl
using Revise
using Projekt_zviazyau

train_path = "/home/ezvezdov/Programming/Projekt_zviazyau/kaggle_data/training.csv"
test_path = "/home/ezvezdov/Programming/Projekt_zviazyau/kaggle_data/test.csv"

X,Y = transform_data(train_path)
preview_image(X,Y,2)
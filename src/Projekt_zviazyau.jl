module Projekt_zviazyau

export transform_data,preview_image

using CSV
using DataFrames
using LinearAlgebra
using Colors, Plots
using Statistics

IMAGE_SIZE = 96
COLOR_MAX = 255

function transform_data(path::String)
    df = DataFrame(CSV.File(path))

    # TODO: REMOVE
    ######################
    df = first(df, 10)
    ######################

    X = parse.(Float64, mapreduce(permutedims, vcat, split.(df.Image," ")))
    X /= COLOR_MAX

    select!(df, Not(:Image))
    
    Y = Matrix(df)
    return X, Y
end

function preview_image(X,Y,image_num; flip = true)
    
    # Reshaping an flipping image
    current_image = reshape(X[image_num,:],96,96);
    current_image = flip ? PermutedDimsArray(current_image, (2, 1)) : current_image
    
    # Reshaping keypoints
    current_keypoints = Y[image_num,:]
    current_keypoints = reshape(current_keypoints,2,15)

    # Plot data
    plot(Gray.(reshape(current_image,IMAGE_SIZE,IMAGE_SIZE)))
    plot!(current_keypoints[1,:], current_keypoints[2,:], seriestype=:scatter, label="Face keypoints")
    
end

end # module Projekt_zviazyau

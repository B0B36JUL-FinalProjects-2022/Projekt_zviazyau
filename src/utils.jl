export transform_train_data,transform_test_data,preview_image

using CSV
using DataFrames

IMAGE_SIZE = 96
IMAGE_HALFSIZE = 48
COLOR_MAX = 255

function transform_image(df)
    X = parse.(Float64, mapreduce(permutedims, vcat, split.(df.Image," ")))
    X /= COLOR_MAX # scale to [0,1]
    X = X'
    return X
end


function transform_train_data(path::String)
    df = DataFrame(CSV.File(path))

    df = dropmissing(df)
    # TODO: REMOVE
    ######################
    df = first(df, 400)
    ######################
    
    X = transform_image(df);

    select!(df, Not(:Image))
    
    y = Matrix(df)
    y = (y .- IMAGE_HALFSIZE) ./ IMAGE_HALFSIZE # scale to [-1,1]
    y = y'

    return X, y
end

function transform_test_data(path::String)
    df = DataFrame(CSV.File(path))
    # TODO: REMOVE
    ######################
    df = first(df, 10)
    ######################
    X = transform_image(df);
    return X
end

function preview_image(X,y,image_num; flip = true)
    
    # Reshaping an flipping image
    current_image = reshape(X[:,image_num],IMAGE_SIZE,IMAGE_SIZE);
    current_image = flip ? PermutedDimsArray(current_image, (2, 1)) : current_image
    
    # Reshaping keypoints
    current_keypoints = y[:,image_num]
    current_keypoints = reshape(current_keypoints,2,15)
    current_keypoints = (current_keypoints .* IMAGE_HALFSIZE) .+ IMAGE_HALFSIZE

    # Plot data
    plot(Gray.(reshape(current_image,IMAGE_SIZE,IMAGE_SIZE)))
    plot!(current_keypoints[1,:], current_keypoints[2,:], seriestype=:scatter, label="Face keypoints")
    
end

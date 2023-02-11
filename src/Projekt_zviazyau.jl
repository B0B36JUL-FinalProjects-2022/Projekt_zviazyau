module Projekt_zviazyau

export transform_data,preview_image, SimpleNet

using CSV
using DataFrames
using LinearAlgebra
using Colors, Plots
using Statistics

IMAGE_SIZE = 96
IMAGE_HALFSIZE = 48
COLOR_MAX = 255

function transform_data(path::String)
    df = DataFrame(CSV.File(path))

    # TODO: REMOVE
    ######################
    df = first(df, 10)
    ######################

    X = parse.(Float64, mapreduce(permutedims, vcat, split.(df.Image," ")))
    X /= COLOR_MAX # scale to [0,1]

    select!(df, Not(:Image))
    
    y = Matrix(df)
    y = (y .- IMAGE_HALFSIZE) ./ IMAGE_HALFSIZE # scale to [-1,1]
    return X', y'
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

struct SimpleNet{T<:Real}
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
end

SimpleNet(n1, n2, n3) = SimpleNet(randn(n2, n1), randn(n2), randn(n3, n2), randn(n3))

function ReLU(Z)
    return max.(0,Z)
end



end # module Projekt_zviazyau

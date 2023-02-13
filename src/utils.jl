export transform_train_data,transform_test_data,make_submission, preview_image

using CSV
using DataFrames

IMAGE_SIZE = 96
IMAGE_HALFSIZE = 48
COLOR_MAX = 255

Keypoints_type = Dict(
    "left_eye_center_x" => 1,"left_eye_center_y" => 2,
    "right_eye_center_x" => 3,"right_eye_center_y" => 4,
    "left_eye_inner_corner_x" => 5,"left_eye_inner_corner_y" => 6,
    "left_eye_outer_corner_x" => 7,"left_eye_outer_corner_y" => 8,
    "right_eye_inner_corner_x" => 9,"right_eye_inner_corner_y" => 10,
    "right_eye_outer_corner_x" => 11,"right_eye_outer_corner_y" => 12,
    "left_eyebrow_inner_end_x" => 13,"left_eyebrow_inner_end_y" => 14,
    "left_eyebrow_outer_end_x" => 15,"left_eyebrow_outer_end_y" => 16,
    "right_eyebrow_inner_end_x" => 17,"right_eyebrow_inner_end_y" => 18,
    "right_eyebrow_outer_end_x" => 19,"right_eyebrow_outer_end_y" => 20,
    "nose_tip_x" => 21,"nose_tip_y" => 22,
    "mouth_left_corner_x" => 23,"mouth_left_corner_y" => 24,
    "mouth_right_corner_x" => 25,"mouth_right_corner_y" => 26,
    "mouth_center_top_lip_x" => 27,"mouth_center_top_lip_y" => 28,
    "mouth_center_bottom_lip_x" => 29,"mouth_center_bottom_lip_y" => 30
)

""" Parse image date to float and scale image colors from [0,255] to [0,1] """
function transform_image(df)
    X = parse.(Float64, mapreduce(permutedims, vcat, split.(df.Image," ")))
    X /= COLOR_MAX # scale to [0,1]
    X = X'
    return X
end

""" Prepare train data. """
function transform_train_data(path::String)
    df = DataFrame(CSV.File(path))

    df = dropmissing(df)
    # TODO: REMOVE
    ######################
    # df = first(df, 400)
    ######################
    
    X = transform_image(df);

    select!(df, Not(:Image))
    
    y = Matrix(df)
    y = (y .- IMAGE_HALFSIZE) ./ IMAGE_HALFSIZE # scale to [-1,1]
    y = y'

    return X, y
end

""" Prepare test data. """
function transform_test_data(path::String)
    df = DataFrame(CSV.File(path))
    # TODO: REMOVE
    ######################
    # df = first(df, 10)
    ######################
    X = transform_image(df);
    return X
end

""" Make submission for kaggle Facial Keypoints Detection competition. """
function make_submission(IdLookupTable_path::String,y)
    df = DataFrame(CSV.File(IdLookupTable_path))
    select!(df, Not(:Location)) # Remove Location column (because of missing)
    df[!,:Location] = zeros(size(df,1)) # Add new Location column with zeros

    # Scale back y to [0,96]
    y = (y .* IMAGE_HALFSIZE) .+ IMAGE_HALFSIZE

    for i in 1:size(df,1)
        df[i,:].Location = y[:,df[i,:].ImageId][Keypoints_type[df[i,:].FeatureName]]
    end

    # Remove unnecessary columns
    select!(df, Not(:ImageId))
    select!(df, Not(:FeatureName))

    # Save data to "kaggle_data" folder
    CSV.write("kaggle_data/submission.csv", df)
end

""" Show image with corresponding facial keypoints. """
function preview_image(X,y,image_num = 1; flip = true)
    
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

export transform_train_data,transform_test_data,merge_data, make_submission, preview_image

using CSV
using DataFrames

IMAGE_SIZE = 96
IMAGE_HALFSIZE = 48
COLOR_MAX = 255

keypoints_types = [
    "left_eye_center_x","left_eye_center_y",
    "right_eye_center_x","right_eye_center_y",
    "left_eye_inner_corner_x","left_eye_inner_corner_y",
    "left_eye_outer_corner_x","left_eye_outer_corner_y",
    "right_eye_inner_corner_x","right_eye_inner_corner_y",
    "right_eye_outer_corner_x","right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x","left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x","left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x","right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x","right_eyebrow_outer_end_y",
    "nose_tip_x","nose_tip_y",
    "mouth_left_corner_x","mouth_left_corner_y",
    "mouth_right_corner_x","mouth_right_corner_y",
    "mouth_center_top_lip_x","mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x","mouth_center_bottom_lip_y"
]

""" Parse image date to float and scale image colors from [0,255] to [0,1] """
function transform_image(df)
    X = parse.(Float64, mapreduce(permutedims, vcat, split.(df.Image," ")))
    X /= COLOR_MAX # scale to [0,1]
    X = X'
    return X
end

""" Transformate keypoinst data, scale data from [0,96] to [-1,1]. """
function trainsform_keypoints(df)
    y = Matrix(df)
    y = (y .- IMAGE_HALFSIZE) ./ IMAGE_HALFSIZE # scale to [-1,1]
    y = y'
    return y
end

""" Prepare train data. """
function transform_train_data(path::String)
    df = DataFrame(CSV.File(path))
    samples_count = size(df,1)

    df4 = deepcopy(df)
    df11 = deepcopy(df)

    keys4 = String[]
    keys11 = String[]
    for key in keypoints_types
        # Remove column if more than 1% is missing
        if sum(ismissing.(df[:,key]))/samples_count > 1e-2
            select!(df4, Not(key))
            push!(keys11, key)

        else
            select!(df11, Not(key))
            push!(keys4, key)
        end
        
    end

    df4 = dropmissing(df4)
    X4 = transform_image(df4)
    select!(df4, Not(:Image))
    y4 = trainsform_keypoints(df4)

    df11 = dropmissing(df11)
    X11 = transform_image(df11);
    select!(df11, Not(:Image))
    y11 = trainsform_keypoints(df11)

    return X11, y11, keys11, X4, y4, keys4
end

""" Prepare test data. """
function transform_test_data(path::String)
    df = DataFrame(CSV.File(path))
    X = transform_image(df);
    return X
end

""" Merge prediction splits. """
function merge_data(y1,keys1,y2,keys2)
    keys_all = vcat(keys1,keys2)
    y = vcat(y1,y2)
    return y, keys_all
end


""" Make submission for kaggle Facial Keypoints Detection competition. """
function make_submission(IdLookupTable_path::String,y, keys_all)
    df = DataFrame(CSV.File(IdLookupTable_path))
    select!(df, Not(:Location)) # Remove Location column (because of missing)
    df[!,:Location] = zeros(size(df,1)) # Add new Location column with zeros

    # Scale back y to [0,96] and round to integer value
    y = (y .* IMAGE_HALFSIZE) .+ IMAGE_HALFSIZE
    y = round.(y)

    # Transform data into DataFrame
    df_pred = DataFrame(y', keys_all)

    # Fill .Location column
    for row in eachrow(df)
        image_id = row.ImageId
        feature_name = row.FeatureName
        row.Location = df_pred[image_id,feature_name]
    end

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
    if length(current_keypoints) == 30
        current_keypoints = reshape(current_keypoints,2,15)
    else
        current_keypoints = reshape(current_keypoints,2,4)
    end
    current_keypoints = (current_keypoints .* IMAGE_HALFSIZE) .+ IMAGE_HALFSIZE

    # Plot data
    plot(Gray.(reshape(current_image,IMAGE_SIZE,IMAGE_SIZE)))
    plot!(current_keypoints[1,:], current_keypoints[2,:], seriestype=:scatter, label="Face keypoints")
    
end

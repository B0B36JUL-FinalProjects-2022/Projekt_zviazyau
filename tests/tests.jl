using Projekt_zviazyau
using Test
using Random
using FileIO,JLD2
using CSV,DataFrames

@testset "Projekt_zviazyau" begin
    @testset "nn_functions.jl" begin
        """ Find derivative using finite difference method. """
        finite_difference(f,x;epsilon=1e-8) = (f.(x .+ epsilon) - f.(x)) / epsilon
    
        rng = MersenneTwister(1234)
    
        @testset "activation functions derivatives" begin
            x = vcat([-125,125,0,1,-1,-35.35,56.34,1e-5,-1e-5],rand(rng,10))
    
            @test d_ReLU(x) ≈ finite_difference(ReLU,x) atol=1e-3
            @test d_id(x) ≈ finite_difference(id,x) atol=1e-3
            @test d_softplus(x) ≈ finite_difference(softplus,x) atol=1e-3
        end
    
        @testset "Loss functions derivatives" begin
            y_true = [1, 2, 3.05, 4.6, -4, 6, 0, 18, 1e-3, -1e-3, 1e3, -1e3]
            y_pred = [2.5, 3, 4.10, 5.3, -4, 1, 3.4, 9, 1e-3, -1e-3, 2e3, -1e3]
            expected_grad = [-3, -2, -2.1, -1.4, 0, 10, -6.8, 18, 0, 0, -2e3, 0]
    
            @test d_MSE(y_true,y_pred) ≈ expected_grad atol=1e-3
        end
    end
    
    @testset "utils.jl" begin
        train_path = "kaggle_data/training.csv"
        test_path = "kaggle_data/test.csv"
        
        img_pixels = 96*96
        keypoinst_amount = 30 
        test_img_amount = 1783
        keys_all = ["left_eye_center_x","left_eye_center_y","right_eye_center_x","right_eye_center_y",
                    "left_eye_inner_corner_x","left_eye_inner_corner_y","left_eye_outer_corner_x","left_eye_outer_corner_y",
                    "right_eye_inner_corner_x","right_eye_inner_corner_y","right_eye_outer_corner_x","right_eye_outer_corner_y",
                    "left_eyebrow_inner_end_x","left_eyebrow_inner_end_y","left_eyebrow_outer_end_x","left_eyebrow_outer_end_y",
                    "right_eyebrow_inner_end_x","right_eyebrow_inner_end_y","right_eyebrow_outer_end_x","right_eyebrow_outer_end_y",
                    "nose_tip_x","nose_tip_y","mouth_left_corner_x","mouth_left_corner_y","mouth_right_corner_x","mouth_right_corner_y",
                    "mouth_center_top_lip_x","mouth_center_top_lip_y","mouth_center_bottom_lip_x","mouth_center_bottom_lip_y"]

        # Read data from CSV
        X11_train,y11_train, keys11, X4_train,y4_train, keys4 = transform_train_data(train_path)
        X_test = transform_test_data(test_path)

        # Load data from jld
        # X11_train ,y11_train, keys11, X4_train,y4_train, keys4 = load("kaggle_data/train_data.jld2","X11_train","y11_train", "keys11", "X4_train","y4_train", "keys4")
        # X_test = load_object("kaggle_data/test_data.jld2")
    
        @testset "transform_train_data()" begin
            # Test dimensions
            @test size(X11_train,1) == img_pixels
            @test size(X4_train,1) == img_pixels
            @test size(y4_train,1) + size(y11_train,1)== keypoinst_amount

            # Test data range
            @test maximum(X11_train) <= 1 && minimum(X11_train) >= 0
            @test maximum(X4_train) <= 1 && minimum(X4_train) >= 0
            @test maximum(y4_train) <= 1 && minimum(y4_train) >= -1
            @test maximum(y11_train) <= 1 && minimum(y11_train) >= -1
        end
        @testset "transform_test_data()" begin
            # Test dimensions
            @test size(X_test,1) == img_pixels
            @test size(X_test,2) == test_img_amount
            @test length(keys4) + length(keys11) == keypoinst_amount

            # Test data range
            @test maximum(X_test) <= 1 && minimum(X_test) >= 0
        end
        @testset "merge_data()" begin
            y11 = zeros(22,test_img_amount)
            y4 = zeros(8,test_img_amount)
            expected = zeros(keypoinst_amount,test_img_amount)
            y, keys_all = merge_data(y11,keys11, y4,keys4)

            @test length(keys_all) == keypoinst_amount
            @test y == expected
        end
        @testset "make_submission()" begin
            sample_submission_path = "kaggle_data/SampleSubmission.csv"
            test_submission_path = "kaggle_data/test_submission.csv"
            IdLookupTable_path = "kaggle_data/IdLookupTable.csv"
            
            y = zeros(keypoinst_amount,test_img_amount) .-1
            make_submission(IdLookupTable_path,y, keys_all; output_path = test_submission_path)
            test_submission = DataFrame(CSV.File(test_submission_path))
            sample_submission = DataFrame(CSV.File(sample_submission_path))
            
            # Compare with sample submission
            @test test_submission == sample_submission
        end
    end      
end


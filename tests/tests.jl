using Projekt_zviazyau
using Test
using Random

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
        X11_train,y11_train, keys11, X4_train,y4_train, keys4 = transform_train_data(train_path)
    
        @testset "transform_train_data()" begin
            @test size(X11_train,1) == img_pixels
            @test size(X4_train,1) == img_pixels
            @test size(y4_train,1) + size(y11_train,1)== keypoinst_amount
        end
        @testset "transform_test_data()" begin
            test_path = "kaggle_data/test.csv"
            X_test = transform_test_data(test_path)
            @test size(X_test,1) == img_pixels
            @test size(X_test,2) == test_img_amount
            @test length(keys4) + length(keys11) == keypoinst_amount
        end
        @testset "merge_data()" begin
            y_test, keys_all = merge_data(y4_test,keys4, y11_test,keys11)
            @test length(keys_all) == keypoinst_amount
            @test size(y_test,1) == keypoinst_amount
        end
        @testset "make_submission()" begin
            sample_submission_path = "kaggle_data/SampleSubmission.csv"
            test_submission_path = "kaggle_data/test_submission.csv"
            IdLookupTable_path = "kaggle_data/IdLookupTable.csv"
            
            y = zeros(keypoinst_amount,test_img_amount) .-1
            make_submission(IdLookupTable_path,y, keys_all; output_path = test_submission_path)

            test_submission = DataFrame(CSV.File(test_submission_path))
            sample_submission = DataFrame(CSV.File(sample_submission_path))
            
            @test test_submission == sample_submission
        end
    end      
end


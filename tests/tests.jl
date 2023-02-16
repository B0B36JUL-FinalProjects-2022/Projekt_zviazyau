using Projekt_zviazyau
using Test
using Random

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
    test_path = "kaggle_data/test.csv"
    train_path = "kaggle_data/training.csv"
    img_pixels = 96*96
    keypoinst_amount = 30 

    
    @testset "transform_train_data" begin
        X11_train,y11_train, keys11, X4_train,y4_train, keys4 = transform_train_data(train_path)
        @test size(X11_train,1) == img_pixels
        @test size(X4_train,1) == img_pixels
        @test size(y4_train,1) + size(y11_train,1)== keypoinst_amount
    end
    @testset "transform_test_data" begin
        X_test = transform_test_data(test_path)
        @test size(X_test,1) == img_pixels
        @test length(keys4) + length(keys11) == keypoinst_amount
    end
    
end


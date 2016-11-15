function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.

C_vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
C_size = size(C_vals,1);
sigma_size = size(sigma_vals,1);

error_matrix = zeros(C_size,sigma_size);

for i = 1:C_size
    for j = 1:sigma_size
        model= svmTrain(X, y, C_vals(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vals(j)));
        predictions = svmPredict(model, Xval);
        new_error = mean(double(predictions ~= yval));
        error_matrix(i,j) = new_error;
    end
end

%Get the index of minimum error from error_matrix
[value,idx] = min(error_matrix(:));

[c_idx, sigma_idx] = ind2sub(size(error_matrix),idx);


C = C_vals(c_idx);
sigma = sigma_vals(sigma_idx);

% find index of minimum error
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end

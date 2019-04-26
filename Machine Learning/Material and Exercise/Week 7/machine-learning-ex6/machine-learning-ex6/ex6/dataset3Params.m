function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%Testing SVM for all of the values mentioned on page 7
C_values = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_values = [0.01 0.03 0.1 0.3 1 3 10 30];

for i = 1:length(C_values)
    for j = 1:length(sigma_values)
        % Set value for C
        C = C_values(i);
        % Set value for Sigma
        sigma = sigma_values(j);
        %Training Svm model using pre-built modules
        svm = svmTrain(X,y,C, @(x1,x2) gaussianKernel(x1,x2,sigma));
        % Generate Prediction based upon models
        predictions = svmPredict(svm, Xval);
        % Compoute error in current svm model
        svm_error(i,j) = mean(double(predictions ~= yval));
    end
end;

% Find Svm model with least error and save it in vector [ i j ]
[i j] = find(svm_error == min(min(svm_error)));

% Assign lowest svm error value model's C and Sigma value to real values.
C = C_values(i);
sigma = sigma_values(j);


% =========================================================================

end

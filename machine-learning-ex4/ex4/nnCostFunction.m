function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Add ones to the X data matrix
X = [ones(m, 1) X];

% =================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% 
% Calculation using loop
%total = 0;
% 
% for i = 1 : m
%   Y_vect = zeros(num_labels,1);
%   A_2 = sigmoid(Theta1 * X(i,:)');  
%   A_2 = [1;A_2];
%   H_theta = sigmoid(Theta2 * A_2);
%   Y_vect(y(i)) = 1;
%   total = total + sum(-Y_vect.*log(H_theta)- (1-Y_vect).* log(1-H_theta));
% end
% J = total/m;

% Using vectorization
eye_mat = eye(num_labels);
yk = eye_mat(y,:);
 
a1 = X;             % a1 -> 5000 X 401
z2 = a1 * Theta1';  % z2 -> 5000  X 25
a2 = sigmoid(z2);   % a2 -> 5000 X 25
z3 = ([ones(size(a2,1),1) a2]) * Theta2'; % z3-> 5000 X 26
a3 = sigmoid(z3); % a3 -> 5000 X 10

J = sum(sum(-yk.*log(a3) - (1 - yk).*log(1-a3)))/m;

% Regularized cost function

J = J + (lambda/(2 * m)) * (sum(sum((Theta1(:,2:end).^2))) + sum(sum((Theta2(:,2:end).^2))));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
 
d3 = a3 - yk;  % d3 - 5000 X 10
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2); % d2 -> 5000 X 25

Delta_1 = d2' * a1; % Delta1 -> 25 X 401
Delta_2 = d3' * ([ones(size(a2,1),1) a2]); % Delta2 -> 10 X 26


Theta1_grad = Delta_1/m;
Theta2_grad = Delta_2/m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% ========================================================================= 

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad = Theta1_grad + ((lambda/m) * Theta1);
Theta2_grad = Theta2_grad + ((lambda/m) * Theta2);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

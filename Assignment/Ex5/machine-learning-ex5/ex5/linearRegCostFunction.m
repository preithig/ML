function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

h = X * theta
firstpart = (h - y) .^ 2

theta = theta(2:size(theta,1),1);

secondpart1 = sum (theta .^ 2)
secondpart = ( lambda / (2* m) ) * secondpart1

J = 1/ (2*m) * sum(firstpart)  + secondpart

grad(1,1) = 1/m * ( X(:,1)' * (h - y) )

X = X(:,2:size(X,2));

gradfirstpart = 1/m * (X' * (h-y))
gradsecondpart = lambda/m * theta

grad(2:end,1) = gradfirstpart + gradsecondpart

grad = grad(:);

end

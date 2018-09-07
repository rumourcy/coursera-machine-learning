function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    yhat = X * theta;
    theta = theta - alpha * (X' * (yhat - y)) / m;

    J_history(iter) = computeCost(X, y, theta);
    
end

end

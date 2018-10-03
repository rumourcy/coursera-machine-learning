function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);

h = sigmoid(X * theta);
loss = -y .* log(h) - (1 - y) .* log(1 - h);
reg = (lambda / (2 * m)) * sum(theta(2:end) .^ 2);


theta(1) = 0;

J = (1 / m) * sum(loss) + reg;
grad = (1 / m) * (X' * (h -y)) + (lambda / m) * theta;

end

function [J, grad] = costFunction(theta, X, y)

m = length(y);

h = sigmoid(X * theta);
loss = -y .* log(h) - (1 - y) .* log(1 - h);
J = (1 / m) * sum(loss);
grad = (1 / m) * (X' * (h -y));

end

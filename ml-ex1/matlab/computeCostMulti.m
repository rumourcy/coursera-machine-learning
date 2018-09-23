function J = computeCostMulti(X, y, theta)

m = length(y);

yhat = X * theta;
loss = yhat - y;

J = (loss' * loss) / (2 * m);

end

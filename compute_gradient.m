function grad = compute_gradient(w, X, y)
    N = size(X, 1);
    predictions = X * w;
    grad = (2/N) * X' * (predictions - y);
end
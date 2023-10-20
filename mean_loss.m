function loss = mean_loss(w, X, y)
    N = length(y);
    predictions = X * w;
    loss = sum((predictions - y).^2) / N;
end
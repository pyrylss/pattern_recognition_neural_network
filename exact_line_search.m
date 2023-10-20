function alpha = exact_line_search(x, d, f)
    % Function for exact line search
    % x: Current point
    % d: Search direction
    % f: Objective function handle
    
    % Define the 1D function for line search
    phi = @(alpha) f(x + alpha * d);
    
    % Minimize the 1D function using MATLAB's built-in minimizer
    alpha = fminbnd(phi, 0, 10); % Assuming a reasonable range [0,10] for alpha
end
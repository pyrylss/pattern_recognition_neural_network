

function [x_star, k] = GM(x0, f, g, epsG, c1, c2, maxiter, epsal, ils)
    % x0: Initial point
    % f: Objective function handle
    % g: Gradient function handle (like compute_gradient)
    % epsG: Tolerance for the gradient norm
    % c1, c2: Constants for the Wolfe conditions
    
    % Initialize
    xk = x0;
    k = 0; % Iteration counter
    f_prev = f(xk); % Initialize previous function value
    

    while true
        
        % Compute gradient at current point
        grad = g(xk);
        f_current = f(xk);
        
        % Check stopping criterion
        if norm(grad) <= epsG
            break;
        end
        
        % Compute search direction
        dk = -grad;
        
        % Compute alpham
        alpham = 2 * (f_current - f_prev) / (grad * dk);

        % Perform line search to find step size (alpha)
        if(ils==2)
            almin= 10^-6; rho=0.5; iW=1;
            [alpha, iout] = uo_BLS(xk,dk,f_current, g, alpham, almin, rho, c1, c2, iW);
        end
        if(ils==3)
            [alpha, iout] = uo_BLSNW32(f, g, xk, dk, alpham, c1, c2, maxiter, epsal);
        end
        
        
        % Update the current point
        xk = xk + alpha * dk;

        % Store values for next iteration
        f_prev = f_current;
        
        % Update iteration counter
        k = k + 1;
    end
    
    % Assign the optimal solution
    x_star = xk;
end
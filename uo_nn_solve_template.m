

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F.-Javier Heredia https://gnom.upc.edu/heredia
% Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)

% Input parameters:
%
% num_target : set of digits to be identified.
%    tr_freq : frequency of the digits target in the data set.
%    tr_seed : seed for the training set random generation.
%       tr_p : size of the training set.ti
%    te_seed : seed for the test set random generation.
%       te_q : size of the test set.
%         la : coefficient lambda of the decay factor.
%       epsG : optimality tolerance.
%       kmax : maximum number of iterations.
%        ils : line search (1 if exact, 2 if uo_BLS, 3 if uo_BLSNW32)
%     ialmax :  formula for the maximum step lenght (1 or 2).
%    kmaxBLS : maximum number of iterations of the uo_BLSNW32.
%      epsal : minimum progress in alpha, algorithm up_BLSNW32
%      c1,c2 : (WC) parameters.
%        isd : optimization algorithm.
%     sg_al0 : \alpha^{SG}_0.
%      sg_be : \beta^{SG}.
%      sg_ga : \gamma^{SG}.
%    sg_emax : e^{SGÇ_{max}.
%   sg_ebest : e^{SG}_{best}.
%    sg_seed : seed for the first random permutation of the SG.
%        icg : if 1 : CGM-FR; if 2, CGM-PR+      (useless in this project).
%        irc : re-starting condition for the CGM (useless in this project).
%         nu : parameter of the RC2 for the CGM  (useless in this project).
%
% Output parameters:
%
%    Xtr : X^{TR}.
%    ytr : y^{TR}.
%     wo : w^*.c
%     fo : {\tilde L}^*.
% tr_acc : Accuracy^{TR}.
%    Xte : X^{TE}.
%    yte : y^{TE}.
% te_acc : Accuracy^{TE}.
%  niter : total number of iterations.
%    tex : total running time (see "tic" "toc" Matlab commands).
%

Xtr= 0; ytr= 0; wo= 0; fo= 0; tr_acc= 0; Xte= 0; yte= 0; te_acc= 0; niter= 0; tex= 0;

fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n')
fprintf('[uo_nn_solve] Pattern recognition with neural networks.\n')
fprintf('[uo_nn_solve] %s\n',datetime)
fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n')

%
% Generate training data set
%
fprintf('[uo_nn_solve] Training data set generation.\n')
[Xtr, ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);


%
% Generate test data set
%
fprintf('[uo_nn_solve] Test data set generation.\n');
te_freq = 0.0;
[Xte, yte] = uo_nn_dataset(te_seed, te_q, num_target, te_freq);

%
% Optimization
%
fprintf('[uo_nn_solve] Optimization\n'); 
tic;  % Start the timer

sig = @(Xtr) 1./(1+exp(-Xtr));

y = @(Xtr,w) sig(w'*sig(Xtr));

% Objective function
L = @(w,Xtr,ytr) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2)+ (la*norm(w)^2)/2;

% Gradient function
gL = @(w,Xtr,ytr) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+la*w;

%initialise w
w = zeros(size(Xtr,1),1);  

function value = f(w)
    value = L(w, Xtr, ytr);
end

function value = g0(w)
    value = gL(w, Xtr, ytr);
end

global_conv = [];
local_conv = [];

grad_prev = 1;
dk_prev = 1;
f_prev = 0;
niter = 1;

if isd == 1  % Gradient Method (GM)
    
    while niter <= kmax

        % objective function
        f_current = L(w, Xtr, ytr);
         % Store the change in objective function value for local convergence
        local_diff = abs(f_current - f_prev);
        local_conv = [local_conv, local_diff];
        
        
        % Compute gradient
        grad = gL(w, Xtr, ytr);
        global_conv = [global_conv, norm(grad)];
       
        
        % Check stopping criterion
        if norm(grad) <= epsG
            break;
        end
        
        % Compute search direction
        dk = -grad;
        
        % Compute alpham
        %alpham = 2 * (f_current - f_prev) / (grad' * dk);
        alpham = (grad_prev' * dk_prev) / (grad' * dk);
        
        % Perform line search to find step size (alpha)
        if(ils==3)
            [alpha, iout] = uo_BLSNW32(@f, @g0, w, dk, alpham, c1, c2, kmaxBLS, epsal);
        end
        
        % Update w
        w = w + alpha * dk;
        
        % Store values for next iteration
        grad_prev = grad;
        dk_prev = dk;
        f_prev = f_current;
        
        % Update iteration counter
        niter = niter + 1;
    end
    wo = w;
    fo = L(w, Xtr, ytr);
%
%
% Quasi-Newton Method (QNM)
%
%
elseif isd == 2  % Quasi-Newton Method (QNM)
    H = eye(size(Xtr, 1));
    
    while niter <= kmax

        % Compute gradient
        grad = gL(w, Xtr, ytr);
    
        % Store the norm of the gradient for global convergence
        global_conv = [global_conv, norm(grad)];
    
        % Compute the objective function value
        f_current = L(w, Xtr, ytr);
    
        % Store the change in objective function value for local convergence
        local_diff = abs(f_current - f_prev);
        local_conv = [local_conv, local_diff];
    
        % Check stopping criterion
        if norm(grad) <= epsG
            break;
        end
    
        % Compute search direction
        dk = -H * grad;
        
        % Compute alpham
        alpham = (grad_prev' * dk_prev) / (grad' * dk);
    
        % Perform line search to find step size (alpha)
        if(ils==3)
            [alpha, iout] = uo_BLSNW32(@f, @g0, w, dk, alpham, c1, c2, kmaxBLS, epsal);
        end
    
        % Update w and compute the new gradient
        w_new = w + alpha * dk;
        grad_new = gL(w_new, Xtr, ytr);
        
        % BFGS Update
        s = w_new - w;
        yg = grad_new - grad;
        rho = 1 / (yg' * s);
        
        H = (eye(size(H)) - rho * s * yg') * H * (eye(size(H)) - rho * yg * s') + rho * (s * s');
        
        % Update values for next iteration
        w = w_new;
        grad = grad_new;
        f_prev = f_current;
        grad_prev = grad;
        dk_prev = dk;
    
        % Update iteration counter
        niter = niter + 1;
    
    end
    wo = w;
    fo = L(w, Xtr, ytr);
%
%
% Stochastic Gradient
%
%
elseif isd == 7  % Stochastic Gradiend
    rng(sg_seed);
    p = size(Xtr, 2);
    m = floor(sg_ga*p); % size of the minibatch
    sg_k_e = ceil(p/m);
    sg_k_max = sg_emax*sg_k_e;
    e = 0;  % Counter for number of epochs
    s = 0;
    best_loss = inf;
    k = 0;
    sg_k = ceil(sg_be * sg_k_max);
    sg_al = 0.01 * sg_al0;
    while e <= sg_emax && s < sg_ebest
        P = randperm(p);
        
        for i = 0:ceil(p/m - 1)

            % Calculate start and end indices for the current mini-batch
            start_idx = i * m + 1;
            end_idx = min((i + 1) * m, p);
            
            % Get the indices for the current mini-batch
            S = P(start_idx:end_idx);
            
            % Index the training data with the mini-batch indices
            xi = Xtr(:, S);  
            yi = ytr(:, S);

            

            grad = gL(w, xi, yi);  % Compute gradient for the current instance

            dk = -grad;
            
            if k <= sg_k
                alpha = (1 - k / sg_k) * sg_al0 + (k / sg_k) * sg_al0;
            else
                alpha = sg_al;
            end
            
            w = w + alpha*dk;  % Update weights
            k = k + 1;   % Increment the iteration counter
                 
        end
        
        e = e + 1;  % Increment epoch counter
        loss = L(w,Xte,yte);
        if best_loss > loss
            best_loss = loss;
        else
            s = s+1;
        end
        
    end
    
    wo = w;
    fo = L(w, Xtr, ytr);
else
    fprintf('[uo_nn_solve] Invalid option for optimization algorithm.\n');
end
tex = toc;  % Get the elapsed time 

%uo_nn_Xyplot(Xtr, ytr, wo)

fprintf('[uo_nn_solve] Local convergence.\n');
%disp(local_conv);


%
% Training accuracy
%
fprintf('[uo_nn_solve] Training Accuracy.\n');
% Calculate predictions for training data

y_pred_train = y(Xtr, wo);
y_pred_train = (y_pred_train >= 0.5); % Convert probabilities to class labels

% Calculate accuracy for training data
tr_acc = sum(y_pred_train == ytr) / length(ytr);
fprintf('[uo_nn_solve] Training Accuracy: %.2f%%\n', tr_acc * 100);

%
% Test accuracy
%
fprintf('[uo_nn_solve] Test Accuracy.\n');
% Calculate predictions for test data
y_pred_test = y(Xte, wo);
y_pred_test = (y_pred_test >= 0.5); % Convert probabilities to class labels

% Calculate accuracy for test data
te_acc = sum(y_pred_test == yte) / length(yte);
fprintf('[uo_nn_solve] Test Accuracy: %.2f%%\n', te_acc * 100);


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
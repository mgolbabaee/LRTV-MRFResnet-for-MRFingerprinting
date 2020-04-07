function [X] = solve_LRTV(data, ELR, TV, param)
% function for reconstruction
%
% Input 
%       data: k-space measurements
%       ELR: the forward/adjoint nufft operators combined with low-rank subspace dimensionality reduction (see Asslander et al. 2018)
%       TV: 2D or 3D Total Variation shrinkage/norm operators (see TV.operator.m)
%       param:
%           maxit: maximum fista iterations (default 10)
%           tol: objective tolerence (default 10e-3)
%           backtrack: true for backtracking adaptive stepsize, false no backtracking (default true) 
%           step: step size, or the initial stepsize when backtracking is active (default 10)
%           lambda: TV regularisation weights (default 0.1)
%           
% Output
%       X: reconstructed (dimension-reduced) time-series of magnetisation images (TSMI) 
%
% (c) Mohammad Golbabaee (m.golbabaee@bath.ac.uk) 2019
%% Input parsing
disp('performing LRTV reconstruction...');

if ~isfield(param,'maxit'); param.maxit = 10; end 
if ~isfield(param,'tol'); param.tol = 1e-3; end 
if ~isfield(param,'backtrack'); param.backtrack = true; end  
if ~isfield(param,'step'); param.step = 10; end 
if ~isfield(param,'lambda'); param.lambda = .1; end 

% initializations
tol = param.tol;
maxit= param.maxit;
step = param.step;
lambda = param.lambda;
backtrack= param.backtrack;
t = 1;
x2_prev = 0;
obj_prev = 0;

%===========MAIN FISTA LOOP
for i = 1:maxit
    
    if i==1
        err=0; val=0; X=0;
    else        
        val = TV.norm(X);
        err = ELR*X;  
    end
    
    err = err - data;   
    grad =   ELR'*err;   % compute the adjoint
    cvxobj = 1/2*norm(err(:))^2 ; % compute fidelity error norm


    %---------backtracking line search
    done = 0;
    while ~done
        x2    =   X - grad * step;
        if lambda>0
            x2 = TV.prox(x2,step*lambda); %prox
        end
        
        if ~ backtrack
            done = 1;
        else            
            err = ELR*x2;  
            tmp = 1/2* norm( reshape(err - data,[],1))^2;
            
            if (tmp  > cvxobj +  real( grad(:)'*(x2(:)-X(:)))  + 1/(2*step) * norm(x2(:)-X(:))^2 ) 
                disp('reducing stepsize...');
                step = step/2;
            else
                done = 1;
            end
        end
    end
    %----------------------
    
    t_prev = t;
    t = 0.5*(1+sqrt(1+4*t^2));
    
    X     =   x2 + (t_prev-1)/(t)* (x2-x2_prev); % FISTA acceleration 

    x2_prev = x2; 
    obj = cvxobj + lambda*val;
    fprintf('=== Iter=%i, Obj_FISTA: |y-Ax|^2 + la.J(x) =%e\n', i,obj);
    
    % stoppage criterium
    if abs(obj-obj_prev)/obj <tol
        break;
    end
    obj_prev = obj;
end


end




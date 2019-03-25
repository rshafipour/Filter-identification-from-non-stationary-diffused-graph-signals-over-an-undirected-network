% Contact Information: Rasoul Shafipour (rshafipo@ur.rochester.edu)
% 
% Algorithm 1, Graph filter identification using Projected Gradient Descent (PGD) in the following paper:
% 
% @article{RSSSAMGM_TSP18,
% 	title={Identifying the Topology of Undirected Networks from Diffused Non-stationary Graph Signals},
% 	author={R. Shafipour and S. Segarra and A. G. Marques and G. Mateos},
% 	journal={IEEE Trans. Signal Processing},
% 	year={2019},
% 	note={(submitted; see also arXiv:1801.03862 [eess.SP])}
% }

% Inputs:
% C_x: input covariance tensor of size N*N*M (N=#nodes and M=#covariance pairs)
% C_y: estimated output covariance tensor of size N*N*M (N=#nodes and M=#covariance pairs)
% H: true underlying filter 
% step_size: learning rate of PGD -- can be modified using line search
% num_iter: maximum number of PGD iterations
% stop_err: stopping criterion

% Outputs:
% H_hat_gd: Estimated graph filter
% filter_id_error_gd: estimated relative error in estimating the filter


function [H_hat_gd,filter_id_error_gd] = filter_est_pgd(C_x,C_y,H,step_size,num_iter,stop_err)

N = size(C_x,1); % number of nodes
M = size(C_x,3); % number of covariance pairs
t = 0; % initializing iterations
H_hat_prev = rand(N,N);
H_hat_gd = rand(N,N);
while t < num_iter && norm(H_hat_gd - H_hat_prev,'fro')/norm(H_hat_gd,'fro') > stop_err
    H_hat_prev = H_hat_gd;
    grad = zeros(N,N);
    for m = 1:M % computes gradient
         grad = grad - 4 * C_y(:,:,m) * H_hat_gd * C_x(:,:,m) + ...
         4 * H_hat_gd * C_x(:,:,m) * H_hat_gd' * H_hat_gd * C_x(:,:,m);
    end
        H_hat_gd = H_hat_gd - step_size * grad;
        H_hat_gd = (H_hat_gd + H_hat_gd') / 2; % projection
        t = t+1;
end

filter_id_error_gd =  min(norm(H_hat_gd - H,'fro')/norm(H,'fro'),norm(-H_hat_gd - H,'fro')/norm(H,'fro'));

end
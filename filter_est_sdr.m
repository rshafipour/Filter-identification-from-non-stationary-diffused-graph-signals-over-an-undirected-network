% Contact Information: Rasoul Shafipour (rshafipo@ur.rochester.edu)
% 
% Algorithm 2, Graph filter identification using Semidefinite Relaxation (SDR) in the following paper:
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
% num_rand: number of randomizations
% H: true underlying filter 

% Outputs:
% H_hat_sdr: Estimated graph filter
% filter_id_error_sdr: estimated relative error in estimating the filter

% This code further requires a khatri-rao product function for kr(A,B)
% which returns the Khatri-Rao product of two matrices A and B. The result is
% formed by the columnwise Kronecker products, where the k-th column of the
% Khatri-Rao product is calculated as kron(A(:,k),B(:,k)).

function [H_hat_sdr , filter_id_error_sdr] = filter_est_sdr(C_x,C_y,num_rand,H)

N = size(C_x,1); % number of nodes
M = size(C_x,3); % number of covariance pairs
C_xyx = zeros(N,N,M);
U_xyx = zeros(N,N,M);
F = zeros(N,N,M);
I = zeros(N,N,M);
J = zeros(N^2,N,M);
for m=1:M
 C_x_m = C_x(:,:,m);
 C_y_m = C_y(:,:,m);
 Cxm_sqrt = sqrtm(C_x_m);
 C_xyx(:,:,m) = Cxm_sqrt * C_y_m * Cxm_sqrt;
 [U_xyx(:,:,m),~] = eig(C_xyx(:,:,m));
 F(:,:,m) = inv(Cxm_sqrt) * sqrtm(C_xyx(:,:,m)) * U_xyx(:,:,m);
 I(:,:,m) = U_xyx(:,:,m)' * inv(Cxm_sqrt);
 J(:,:,m) = kr(I(:,:,m)' , F(:,:,m)); % calls kr function
 % for Khatri-rao product.
end


K = [];
for i = 1:M-1
    KJ = zeros((M-i)*N^2, M*N);
    for j=1:M-i
        KJ( (j-1)*(N^2) + 1 : j*(N^2) , (j-1)*N + 1 : (j*N) )= J(:,:,j);
        KJ( (j-1)*(N^2) + 1 : j*(N^2) , (j+i-1)*N + 1:(j+i)*N )= -J(:,:,j+i);
    end
    K=[K ; KJ];
end


%% CVX
error = zeros(num_rand,1);
obj_approx = zeros(num_rand,1);
b_hat = zeros(N*M,num_rand);

cvx_begin quiet
    variable B(N*M,N*M) semidefinite
     minimize( trace(K' * K * B) )
     subject to
     B >= 0
     diag(B) == ones(N*M,1)
cvx_end

mu = zeros(1,N*M);
sigma = B;
for i=1:num_rand
z = mvnrnd(mu,sigma);
b_hat(:,i) = sign(z);
obj_approx(i) = trace(K' * K * b_hat(:,i) * b_hat(:,i)');
end

[~,index] = min(obj_approx);
b_hat = b_hat(:,index);
H_hat_sdr = F(:,:,1) * diag(b_hat(1:N)) * I(:,:,1); % Estimated filter
filter_id_error_sdr = min(norm(H_hat_sdr - H,'fro')/norm(H,'fro'),norm(-H_hat_sdr - H,'fro')/norm(H,'fro'));
end
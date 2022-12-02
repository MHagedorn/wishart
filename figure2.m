% supplementary material on "Expected Value of Matrix Quadratic Forms 
% with Wishart distributed Random Matrices" (2022)

% simple example of how two algorithms can be compared using Theorem 1
% based on Code of Jarre and Hagedorn
clear all
clc

% initialization
n     = 10;
kmax  = 1e7;
kappa = 5;     % condition of A
tol   = 1e-10; % tolerance for feasibility
beta  = 0;     % weight of the iterates

rng(131022)    % seed


% construction of a symmetric, positive semidefinite matrix Sigma
Sigma = randn(n);        % random matrix
Sigma = Sigma*Sigma';    % to make Sigma symmetric and pos. semidefinite
Sigma = 0.5*(Sigma + Sigma.');
if norm(Sigma-Sigma') > tol
    disp('Matrix Sigma is not symmetric.')
end
if min(eig(Sigma)) <= 0
    disp('Matrix Sigma is not positive definite.')
end


% a symmetric positive definite matrix A is needed with norm 1 
% and predefined condition kappa
A = randn(n);     % standard normal distributed entries
A = A+A.';  % to make the matrix symmetric so that all eigenvalues are real
[u,d] = eig(A);   % eigenvalue decomposition
d = abs(diag(d)); % to make the matrix positive definite
dmin = min(d);    % minimal eigenvalue
dmax = max(d);    % maximal eigenvalue
d = (d-dmin)/(dmax-dmin)*(1-1/sqrt(kappa))+1/sqrt(kappa);
A = u.*(ones(n,1)*d.');
A = A*A';
A = 0.5*(A+A.'); 
if abs(norm(A)-1) > tol
    disp('Norm of A is not one.')
end
if abs(cond(A)-kappa) > tol
    disp('Condition of A is not kappa.')
end
if norm(A-A') > tol
    disp('Matrix A is not symmetric.')
end
if min(eig(A)) < 0
    disp('Matrix A is not positive semidefinite.')
end


gamma = 1e-3; % step length determined by trial and error

tgrad = A*Sigma*A'; % true gradient is rgrad*x
x = randn(n,1);     % initial value
x = x/norm(x);      % standardized
    
% for analysis
ngs    = zeros(1,kmax+1);   % norms of gradients of SGD
ngs2   = zeros(1,kmax+1);   % norms of gradients of ASGD
xs     = zeros(n,kmax+1);   % iterates of SGD
xbars  = zeros(n,kmax+1);   % iterates of ASGD
xis    = zeros(n,kmax);     % noises of SGD
xis2   = zeros(n,kmax);     % noises of ASGD
covs  = zeros(1, kmax);
covs2 = zeros(1, kmax);
[U,D] = eig(Sigma);         % eigenvalue decomposition of Sigma 
d     = diag(D);            % the vector with the diagonal entries of D

ngs(:,1)   = norm(tgrad*x);
xbars(:,1) = x;
xs(:,1)    = x;
    
xbar  = x; % sum of weihted iterates
sigma = 0; % sum of weights
   
for l = 1:kmax
        
    % r^l and b^l from the multivariate normal distribution:
    rb = mvnrnd(zeros(1,n),Sigma,2); % first row r, second row b
        
    a  = A*rb(1,:)';          % a^l = A*r^l
    b  = rb(2,:)'/sqrt(n);    % standardized b^l
        
    sgrad = (a'*x)*a + b;     % stochastic gradient
    xi    = sgrad - tgrad*x;  % noise xi^l
        
    x = x - gamma * sgrad;    % update
        
    sigma = sigma + l^beta;    
    xbar  = xbar + l^beta * x;
    
    ngs(l+1)  = sqrt(sum((tgrad*(x)).^2));
    ngs2(l+1) = sqrt(sum((tgrad*(xbar/sigma)).^2));
    xs(:,l+1) = x;
    xbars(:,l+1) = xbar/sigma;
    xis(:,l)  = xi;
    xis2(:,l) = (a'*(xbar/sigma))*a + b - tgrad*(xbar/sigma);
    B = A'*(x*x')*A;
    E = U*(2*(d*d').*(U'*B*U) + trace(U'*B*U*D)*D)*U';
    Covtheo = A*E*A' + Sigma - A*Sigma*B*Sigma*A';
    % at the optimal solution is Cov(xi) = Sigma
    covs(l) = norm(Sigma - Covtheo);
    B = A'*((xbar/sigma)*(xbar/sigma)')*A;
    E = U*(2*(d*d').*(U'*B*U) + trace(U'*B*U*D)*D)*U';
    Covtheo = A*E*A' + Sigma - A*Sigma*B*Sigma*A';
    covs2(l) = norm(Sigma - Covtheo);
end
    
xfinal = xbar/sigma; % final iterate
disp(['Norm of the final ASGD gradient: ',num2str(norm(tgrad*xfinal))])
  
set(gcf,'Position',  [100, 100, 1000, 800])
subplot(2,2,1)
semilogy(1:kmax+1, ngs, 'DisplayName','SGD')
hold on
semilogy(1:kmax+1, ngs2, '--', 'DisplayName','ASGD')
xlabel('Number of iterations')
ylabel('$\| \nabla f(x^k)\|$','Interpreter','latex')
xlim([0  kmax+1])
legend

subplot(2,2,3)
semilogy(1:kmax+1, sqrt(sum(xs.^2,1)), 'DisplayName','SGD')
hold on
semilogy(1:kmax+1, sqrt(sum(xbars.^2,1)), '--', 'DisplayName','ASGD')
xlabel('Number of iterations')
ylabel('$\| x^k - x^{opt}\|$','Interpreter','latex')
xlim([0  kmax+1])
legend
    
subplot(2,2,2)
nxis  = cumsum(xis,2)./(1:kmax);        % expected values
nxis  = sqrt(sum(nxis.^2,1));           % norm of expected values
nxis2 = cumsum(xis2,2)./(1:kmax);       % expected values
nxis2 = sqrt(sum(nxis2.^2,1));          % norm of expected values
loglog(1:kmax, nxis, 'DisplayName','SGD')
hold on
loglog(1:kmax, nxis2, '--', 'DisplayName','ASGD')
xlabel('Number of samples to estimate E(\xi^k)')
% E_{theo} = 0_n
ylabel('$\| E_{exact}-E_{empiric}\|$','Interpreter','latex')
xlim([0  kmax])
legend
   
subplot(2,2,4)
semilogy(1:kmax, covs, 'DisplayName', 'SGD')
hold on
semilogy(1:kmax, covs2, '--', 'DisplayName', 'ASGD')
xlabel('Number of iterations')
ylabel('$ \| Cov(\xi^k) - \Sigma\| $',...
    'Interpreter','latex')
legend



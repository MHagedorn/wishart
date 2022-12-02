% supplementary material on "Expected Value of Matrix Quadratic Forms 
% with Wishart distributed Random Matrices" (2022)

% Numerical example for the validity of Theorem 1
clear all
clc

rng(121022)                % seed for reproducibility
n    = 10;                 % number of dimensions
k    = 3;                  % number of degrees of freedom
tol  = 1e-8;               % tolerance for feasibility tests
runs = 10;                 % number of independent runs
mmax = 1e6;                % number of generated Q to approx E(QBQ)
peepholes = 0:log10(mmax); % after 1, 10, 100, .. iterations

Ediff = zeros(runs, length(peepholes));
for run = 1:runs
    disp([num2str(run),'/',num2str(runs)])

% construction of a symmetric matrix B
B = rand(n,n);             % random matrix
B = B + B';                % to make B symmetric
if norm(B-B') > tol
    disp('Matrix B is not symmetric.')
end

% construction of a symmetric, positive semidefinite matrix Sigma
Sigma = randn(n);          % random matrix
Sigma = Sigma*Sigma';      % to make Sigma symmetric and pos. semidefinite
if norm(Sigma-Sigma') > tol
    disp('Matrix Sigma is not symmetric.')
end
if min(eig(Sigma)) < 0
    disp('Matrix Sigma is not positive semidefinite.')
end


[U,D] = eig(Sigma);        % eigenvalue decomposition of Sigma 
if norm(Sigma-U*D*U') > tol
    disp('Eigendecomposition of Sigma incorrect.')
end



% expected value matrix E(QBQ) due to theorem 1
d = diag(D);               % the vector with the diagonal entries of D
E = k*U*(2*(d*d').*(U'*B*U) + trace(U'*B*U*D)*D)*U'+(k^2-k)*Sigma*B*Sigma;


% approximating E(QBQ) by averaging 
Eapprox = zeros(n,n);

for i=1:mmax
    Q = wishrnd(Sigma,k);  % random Wishart distributed Q
    Eapprox = Eapprox + Q*B*Q; 
    if sum(log10(i) == peepholes) == 1 
        Ediff(run,log10(i)==peepholes) = norm(E - Eapprox/i)/norm(E);
    end
end

end


err  = sqrt(var(Ediff));          % standard deviation
arm = mean(Ediff);                % arithmetic mean  
errorbar(10.^peepholes, arm, err) % plot relative error in dependence of m
set(gcf,'Position',  [100, 100, 700, 500])
set(gca, 'XScale','log', 'YScale','log')
grid
axis([0.7  1.3*mmax   0.2*arm(end)  arm(1)+3*err(1)])
xlabel('Number of samples to estimate E($QBQ$)',...
    'Interpreter','latex') 
ylabel('$\| E_{empiric} - E_{exact}\| / \|E_{exact}\|$',...
    'Interpreter','latex')


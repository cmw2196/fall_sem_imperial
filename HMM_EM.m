% Data generating function

N1  = 10;         % number of sequences
T1  = 100;        % length of the sequence
pi1= [0.5; 0.5]; % inital probability pi_1 = 0.5 and pi_2 =0.5

%%two states hence A is a 2X2 matrix 
A1  = [0.4 0.6 ; 0.4 0.6 ];         %p(y_t|y_{t-1})

%%alphabet of 6 letters (e.g., a die with 6 sides) E(i,j) is the
E1 = [1/6 1/6 1/6 1/6 1/6 1/6;      %p(x_t|y_{t}) 
    1/10 1/10 1/10 1/10 1/10 1/2];

[ Y, S ] = HmmGenerateData(N1, T1, pi1, A1, E1 ); 

%%Y is the set of generated observations 
%%S is the set of ground truth sequence of latent vectors 

Amat  = [0.5 0.5 ; 0.5 0.5];
E1 = [1/6 1/6 1/6 1/6 1/6 1/6;      %p(x_t|y_{t}) 
    1/6 1/6 1/6 1/6 1/6 1/6];
pi2 = [.5, .5];
iters = 4;
[A_new, pi_new, theta_new] = EM(Y, E1, Amat, pi2, iters); % run E and M steps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Perform the EM algorithm
% First I need to make functions to calculate the forward and backward
% probabilities

    
% forward probabilities alpha(t) - these are normalized
function [alpha_zt, cn] = calculate_alpha(n, T, obs_data, A, emission_prob, pi)
cn = zeros(T, 1);

alpha_zt(1,:) = [emission_prob(1, obs_data(n,1)) * pi(1), emission_prob(2, obs_data(n,1)) * pi(2)];    %alpha(z_1)
cn(1) = sum(alpha_zt(1,:));
alpha_zt(1,:) = alpha_zt(1,:) ./ cn(1);

for t = 2:T
    roll = obs_data(n,t);
    p_xz = [emission_prob(1,roll) emission_prob(2,roll)]; % emission probs for the t'th roll
    cn(t) = sum(alpha_zt(t-1,:));
    alpha_zt(t,:) = (1/cn(t)) .* (p_xz .* (alpha_zt(t-1,:) * A));  % calculate alphaHAT(t)
end
end

% backward probabilities beta(t)
function beta_zt = calculate_beta(n, T, obs_data, A, emission_prob, pi)
beta_size = size(obs_data,2); 
beta_zt = zeros(size(obs_data,2), size(emission_prob, 1));

beta_zt(beta_size,:) = 1/size(emission_prob,1);  % base case
[alpha, cn] = calculate_alpha(n, T, obs_data, A, emission_prob, pi);

for t = beta_size-1:-1:1
    roll = obs_data(n, t);
    p_xz = [emission_prob(1, roll), emission_prob(2, roll)];  % emission probs for t+1'th roll
    beta_zt(t,:) =(1/cn(t+1)).* ((beta_zt(t+1,:) .* p_xz) * transpose(A)) ;  % calculate betaHAT_t by counting backwards from T, T-1, T-2, ...t
end
end
       



function [A_new, pi_new, theta_new] = EM(obs, emission_prob, A, pi, iternum)
% These stand for, respectively, noStrings, noTimepts, noZstates, noXstates
N = size(obs, 1);
T = size(obs, 2);
Z = size(emission_prob, 1);
X = size(emission_prob, 2);

E_ztk = zeros(size(obs,1), T, Z);
E_ztj_ztk = zeros(size(obs, 1), T, Z, Z);  %dim: (string num, observations, Z x Z)


% repeat EM algorithm several times

for s = 1:iternum
if s > 1
    A = A_new
    pi = pi_new
    emission_prob = theta_new
end
% E Step
for n = 1:N
    % for each String n, calculate the arrays a(t), beta(t) for all t =
    % 1...T
    [aT, cn] = calculate_alpha(n, T, obs, A, emission_prob, pi);  %an array of a_1...a_T, and the corresponding c1...cn
    bT = calculate_beta(n, T, obs, A, emission_prob, pi); % an array of b_1...b_T
    jp_X = sum(aT(T,:));  % a_T1 + a_T2 = p(x_1...x_T)      
    %cn = prediction(n, T, obs, A, emission_prob, jp_X, aT);  %calculate p(xt|xt-1) for all t

    for t = 1:T 
        for j = 1:Z %over the two prior states Z
            E_ztk(n,t,j) = aT(t,j)*bT(t,j);  % find E[z_tj] with normalized alphas and betas
            if t ~= 1
                aMat = repmat(transpose(aT(t-1,:)), 1, 2); % Z x Z matrix of alphas
                b = [emission_prob(1, obs(n,t)), emission_prob(2, obs(n,t))];
                bMat = repmat(bT(t,:) .* b, 2 ,1); % Z x Z matrixs of beta * eprob
                E_ztj_ztk(n,t,:,:) = cn(t) .*aMat .* A .* bMat; %find E[z_t-1, z_t]
            end
        end
    end
end 
 

    % M Step
    % initialize final parameter arrays
    theta_new = zeros(Z, X);  % Z x noStatesX
    theta_denom = zeros(Z ,1); % Z x noStates 1
    pi_new = zeros(1, Z); % 1 x Z
    A_new = zeros(Z); % Z x Z
    A_denom = zeros(Z);  % Z % Z - will calculate final A with elementwise division
    
    for n = 1:N  % sum over all sequences
        for t = 1:T  % over all time points
            for k = 1:Z % over all states Z
                pi_new(1,k) = pi_new(1,k) + E_ztk(n,1,k);
                if t ~= 1
                    for r = 1:Z  % couldnt get this to work rowwise so another loop...
                        A_new(k,r) = A_new(k,r) + E_ztj_ztk(n,t,k,r);
                        A_denom(k,r) = A_denom(k,r) + sum(E_ztj_ztk(n,t,k,:));
                    end
                end  
                theta_denom(k) = theta_denom(k) + E_ztk(n,t,k);                
                for j= 1:X % over all states at X|Z
                    if obs(n,t) == j    %replacement for kronecker delta
                        theta_new(k,j) = theta_new(k,j) + E_ztk(n,t,k); % 1 is the selector
                    end
                end
            end
        end
    end
    theta_denom = repmat(theta_denom,1, 6);
    theta_new = theta_new ./ theta_denom;
    pi_new = pi_new ./ sum(pi_new);
    A_new = A_new ./ A_denom;
end
end

    
    
    
    



    


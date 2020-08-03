%%%2 tasks: (1) pricing options; (2) decomposing bond excess returns%%
addpath(genpath(cd)) 
%% Pricing options

% Simulation price path 2 assets
S1 = 100; %underlying price stock 1
S2 = 100; %underlying price stock 2
rf = .05;
sigma1 = 0.2;
sigma2 = 0.35;
T = 1;
p = 0.5; %this is just the assumption for correlation. The range of correlation coefficients follow when estimating portfolio values.
d = 0; 
No_Sim = 50000;
%S_T = S*exp((rf-0.5*sigma^2)*T+sigma*sqrt(T)*randn); %This is for one
%stock

%simulation of asset paths
No_step = 250;%number of time steps in the interval (like 250 days in a year)
Delta_T = T/No_step;%time increment
%Generating random numbers (shocks) for 2 variables 
shocks1  = randn(No_Sim, No_step);
shocks2 = shocks1.*p + sqrt(1-p^2).*randn(No_Sim, No_step); % this already indicates correlation
%Generating storage for the paths of 2 stocks
S_T1 = NaN(No_Sim, No_step+1);
S_T1(:,1) = S1;
S_T2 = NaN(No_Sim, No_step+1);
S_T2(:,1) = S2;
%Paths for both stocks 
for i = 2:No_step+1
    % Path for stock 1
    S_T1(:,i) = S_T1(:,i-1).*exp((rf-0.5*sigma1^2)*Delta_T+sigma1*sqrt(Delta_T).*shocks1(:,i-1));
    % Path for stock 2
    S_T2(:,i) = S_T2(:,i-1).*exp((rf-0.5*sigma2^2)*Delta_T+sigma2*sqrt(Delta_T).*shocks2(:,i-1));
end
Mean_S_T1 = mean(S_T1);
Mean_S_T2 = mean(S_T2);

Portfolio = S_T1.*0.5 + S_T2.*0.5;

%Plotting asset paths and portfolio value distribution at in the end
%Firstly, plotting the average price path for both securities
subplot(2,2,1)
plot(Mean_S_T1) %I plot one of the 50k generated paths to see how the paths relate.
title('Average Price Path of Two Securities')
xlabel('Day')
ylabel('Price (S_T)')
hold on
plot(Mean_S_T2)
xlabel('Day')
ylabel('Price (S_T)')
hold off
legend('S1','S2','Location','northwest')
% Plotting single price path for both securities
subplot(2,2,2)
plot(S_T1(25000,:)) %I plot one of the 50k generated paths to see how the paths relate.
title('Single Price Path of Two Securities')
xlabel('Day')
ylabel('Price (S_T)')
hold on
plot(S_T2(25000,:))
xlabel('Day')
ylabel('Price (S_T)')
hold off
legend('S1','S2','Location','northwest')
% Plotting the portfolio distribution at day = 250 
subplot(2,2,3)
histogram(Portfolio(:,end))
title('Distribution of Portfolio Prices at t=250')
xlabel('Portfolio value')
ylabel('Count')

p = linspace(-1,1,21); %the range of correlation coefficients. 
% to calculate the percentile prctile(Portfolio(:,51)',0.05)
%Estimating portfolio value and simulating it for each correlation
%coefficient.

No_Sim = 10000;
No_draws = 10000;
for y = 1:length(p)
    for i = 1:No_Sim
        shocks1 = randn(No_draws,1);
        shocks2 = shocks1.*p(y) + sqrt(1-p(y)^2).*randn(No_draws,1);
        S_T1 = S1*exp((rf-0.5*sigma1^2)*T+sigma1*sqrt(T)*shocks1);
        S_T2 = S2*exp((rf-0.5*sigma2^2)*T+sigma2*sqrt(T)*shocks2);
        Portfolio = S_T1.*0.5 + S_T2.*0.5;
        percentile_P = prctile(Portfolio, 0.05); % this is for the 5th percentile 
        Val_P(i,y) = mean(Portfolio);
        fifth_percentile(i,y) = mean(percentile_P);
        %also need to do the AVERAGE PORTFOLIO to plot the distributio of
        %it at various correlations.
    end
end

%distribution of 5th portfolio percentile as a function of the correlation coeff. 
figure
plot(p,mean(fifth_percentile))
xlabel('Correlation')
ylabel('Portfolio value - 5th percentile')
%distribution of portfolio value at different correlation coefficients
figure
subplot(5,1,1)
histogram(Val_P(:,1))
title('Distribution of Portfolio Value at P = -1') %P = -1
xlabel('Portfolio value')
ylabel('Count')
subplot(5,1,2)
histogram(Val_P(:,6))
title('Distribution of Portfolio Value at P = -0.5') %P = -0.5
xlabel('Portfolio value')
ylabel('Count')
subplot(5,1,3)
histogram(Val_P(:,11))
title('Distribution of Portfolio Value at P = 0') %P = 0
xlabel('Portfolio value')
ylabel('Count')
subplot(5,1,4)
histogram(Val_P(:,16))
title('Distribution of Portfolio Value at P = 0.5') %P = 0,5
xlabel('Portfolio value')
ylabel('Count')
subplot(5,1,5)
histogram(Val_P(:,21))
title('Distribution of Portfolio Value at P = 1') %P = 1
xlabel('Portfolio value')
ylabel('Count')


%Pricing an exchange option
for y = 1:length(p)
    for i = 1:No_Sim
        shocks1 = randn(No_draws,1);
        shocks2 = shocks1.*p(y) + sqrt(1-p(y)^2).*randn(No_draws,1);
        S_T1 = S1*exp((rf-0.5*sigma1^2)*T+sigma1*sqrt(T)*shocks1);
        S_T2 = S2*exp((rf-0.5*sigma2^2)*T+sigma2*sqrt(T)*shocks2);
        Dummy_exchange = S_T1 > S_T2;
        Payoff = Dummy_exchange.*(max(0,S_T1-S_T2)); 
        Ave_Payoff = mean(Payoff);
        Val_Opt(i,y) = Ave_Payoff * exp(-rf*T);
    end
end

Option_value_corrs = mean(Val_Opt); %Average option price at each correlaton coeff
Val_Opt_negativecorr = Option_value_corrs(:,1); %Option price at p=-1
Val_Opt_poscorr = Option_value_corrs(:,end); %Option price at p=1
mean(Option_value_corrs); %Option price averaged across correlations

%Graph of option value against correlations. 
figure
plot(p,Option_value_corrs)
title('The Distribution of Option Price Across Different Correlations')
xlabel('Correlation')
ylabel('Option value')


%% Decomposing bond excess returns
% load data
load('lecture6.mat')

dates = lbusdate(dates(1,:),dates(2,:));
T = length(dates); % 675 monthly observations
J = length(mats); %40 different maturities
rx_u = rx./12; %unannualizing bond excess returns.

% 1. Nelson - Siegel factor loadings and estimation of Phi_P0 and Phi_P1
tau   = mats; % this is just time period
term1 = ones(J,1);
term2 = (1-exp(-lambda*tau))./(lambda*tau) ;
term3 = term2 -exp(-lambda*tau);
L = [term1, term2, term3]; %Nelson-Siegel factors

%Extracting these factors
Z = nan(3,T); %Z includes time-series of betas on these 3 dynamic Nelson-Siegel factors
for i = 1:T
    Z(:,i) = regress(yld(:,i), L); % Z is betas when regressing yield against these factors
end

%Estimating factor dynamics
t = 1:T-1; % from 1 to 674
Y = Z(:,t+1); % Zt+1
A = [ones(1,T-1);Z(:,t)]; % Zt; the vector of 1's is for the estimation of Phi_p0 
Phi_p = mrdivide(Y,A);
Phi_p0 = Phi_p(:,1);
Phi_p1 = Phi_p(:,2:end);

%Estimating capital sigma-hat 
%solving for w
W = Y-Phi_p*A;
Sigma_hat = zeros(3);
for i = 1:t
    Sigma_hat = Sigma_hat + (W(:,i)*W(:,i)');
end
Sigma_hat = Sigma_hat./674;

% 2. Estimation of Phi_Q0 and Phi_Q1
Re = rx_u; % this is to keep track with the notation
D = Re(:,t+1); %removing NaN's from the first column of Re

% we can set up the equation as Re = [ R0, B, R1][i1T,Z, Z(-)]^T where
% [ R0, B, R1] is X and [i1T,Z, Z(-)]^T is F and Re is D; hence we solve D = xF
%then I use mrdivide(D, C)
F = [ones(T-1,1)';Z(:,t+1);Z(:,t)];
X = mrdivide(D,F);
R0 = (X(:,1));
B = (X(:,2:4));
R1 = (X(:,5:7)); 

%Estimating error term E:
E = D - X*F; 

% solving for Phi_q1: R1 = -B*Phi_q1 with mldivide function
Phi_q1 = mldivide(-B,R1);

% solving for Phi_q0
for i = 1:8
    C0(i) = 0.5.*(B(i,:)*Sigma_hat*B(i,:)');
end
C0 = C0';

Phi_q0 = mldivide(-B,R0+C0);

% Calculating lambdas
Lambda0 = Phi_p0 - Phi_q0;
Lambda1 = Phi_p1 - Phi_q1;

% 3. Simulations synthetic samples of lambdas 
%Matrix of residuals U:
U = [W;E];
Nr_Sim = 8000;
    
Lambda0_star = zeros(3,1,Nr_Sim);   
Lambda1_star = zeros(3,3,Nr_Sim);   
for i = 1:Nr_Sim
    %randomizing the colums of matrix U
    U_star = datasample(U,size(U,2),2);
    %Estimating Z using phi_p and random error component
    Z_star = nan(3,T);
    Z_star(:,1) = Z(:,1);
    for y = 2:675
        Z_star(:,y) = Phi_p0 + Phi_p1*Z_star(:,y-1) + U_star(1:3,y-1);
    end
    %Again estimating phi_p which is calles phi_p_star
    Y_star = Z_star(:,t+1); % Zt+1
    A_star = [ones(1,T-1);Z_star(:,t)]; % Zt; the vector of 1's is for the estimation of Phi_p0 
    Phi_p_star = mrdivide(Y_star,A_star);
    Phi_p0_star = Phi_p_star(:,1);
    Phi_p1_star = Phi_p_star(:,2:end);
    % Estimating sigma_hat_star
    W_star = Y_star-Phi_p_star*A_star;
    Sigma_hat_star = zeros(3);
    for i1 = 1:t
        Sigma_hat_star = Sigma_hat_star + (W_star(:,i1)*W_star(:,i1)');
    end
    Sigma_hat_star = Sigma_hat_star./674;
    %Estimating phi_q as phi_q_star
    F_star = [ones(T-1,1)';Z_star(:,t+1);Z_star(:,t)];
    rx_star = X*F_star + U_star(4:11,:);
    F = [ones(T-1,1)';Z(:,t+1);Z(:,t)];
    X_star = mrdivide(rx_star,F_star);
    R0_star = (X_star(:,1));
    B_star = (X_star(:,2:4));
    R1_star = (X_star(:,5:7)); 
    Phi_q1_star = mldivide(-B_star,R1_star);
    for i2 = 1:8
        C0_star(i2) = 0.5.*(B_star(i2,:)*Sigma_hat_star*B_star(i2,:)');
    end
    %C0_star = C0_star';
    Phi_q0_star = mldivide(-B_star,R0_star+C0_star');
    % Calculating lambdas
    Lambda0_star(:,:,i) = Phi_p0_star - Phi_q0_star;
    Lambda1_star(:,:,i) = Phi_p1_star - Phi_q1_star;
end

%Standard error of Lambda_star estimates
stdev_lambda0 = std(Lambda0_star,0,3);
stdev_lambda1 = std(Lambda1_star,0,3);

% 4. Beta representation and decomposition of z for 10yr bond. 
%bn should be 3X1 vector lambda term should be 3x1 vector as well
b10 = B(end,:);

%10yr bond excess returns. 
Exp_r_10 = b10*(Lambda0+Lambda1*Z(:,1:end-1)).*12.*100; %- 0.5.*b10*Sigma_hat*b10'; %MIGHT NEED TO SUBTRACT THE VARIANCE COMPONENT

%Obtaining a 3xT-1 vector for Lambda1*Z, and decomposing it for Z1 Z2 Z3. 
Lambda1Zt = Lambda1*Z(:,1:end-1);
Lambda1z1 = Lambda1Zt(1,:);
Lambda1z2 = Lambda1Zt(2,:);
Lambda1z3 = Lambda1Zt(3,:);

%Calculating exp. 10yr bond excess returns associated with each z. 
Exp_r_10z1 = b10(:,1)*(Lambda0(1,1)+Lambda1z1).*12.*100; %- (0.5.*(b10(:,1)*diag(Sigma_hat(1,1))*b10(:,1 -> only 1 row from this. 
Exp_r_10z2 = b10(:,2)*(Lambda0(2,1)+Lambda1z2).*12.*100; %- (0.5.*(b10(:,2)*Sigma_hat(2,2)*b10(:,2)'));
Exp_r_10z3 = b10(:,3)*(Lambda0(3,1)+Lambda1z3).*12.*100; %- (0.5.*(b10(:,3)*Sigma_hat(3,3)*b10(:,3)'));
Exp_r_10 - Exp_r_10z1 - Exp_r_10z2 - Exp_r_10z3; %to check if all 10yr ex. return is accounted

%Illustrating exp. excess returns in area plot
figure
area(datetime(dates(1,2:675),'ConvertFrom','datenum','Format','yyyy'),[Exp_r_10z1;Exp_r_10z2;Exp_r_10z3]')
title('Decomposition of 10-Year Bond Expected Excess Returns')
ylabel('%')
legend('Z1','Z2','Z3')

clearvars -except dates tau L Z Phi_p Phi_p0 Phi_p1 Sigma_hat Re D R0 B R1 E Phi_q1 C0 Phi_q0 Lambda0 Lambda1 U Lambda0_star Lambda1_star stdev_lambda0 stdev_lambda1 b10 Exp_r_10 Exp_r_10z1 Exp_r_10z2 Exp_r_10z3 b10 Mean_S_T1 Mean_S_T2 S_T1 S_T2 Portfolio p fifth_percentile(i,y) Val_P(i,y)  Val_Opt(i,y) Option_value_corrs Val_Opt_negativecorr Val_Opt_poscorr mean(Option_value_corrs) shocks1 shocks2 sigma1 sigma2
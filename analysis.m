%% 2-dim Simulation Experiment - V3
% Each customer choose 7 times from 7 different choice sets
% Gibbs sampling with Metropolis step
% problabel: denominator changed to maxpdf
% MLE added
%% Initialization
clear; close all; clc;

%% Simulation - generate X,Y
% choice set: X(m,:) = [id,1/price,rate]
M = 1000;
X = zeros(M,3);
% id
X(:,1) = 1:M;
% 1/price, rate
X(:,2:3) = mvnrnd([0.5,0.5],[0.0225,-0.01575;-0.01575,0.0225],M);

% users: Y(n,:) = [id,age,income]
N = 200;
Y = zeros(N,3);
% id
Y(:,1) = 1:N;
% age
Y(:,2) = normrnd(0.5,0.15,N,1);
% Y(1:100,2) = normrnd(0.2,0.04,100,1);
% Y(101:200,2) = normrnd(0.8,0.04,100,1);
% income
Y(:,3) = normrnd(0.5,0.15,N,1);
% Y(1:100,3) = normrnd(0.8,0.04,100,1);
% Y(101:200,3) = normrnd(0.2,0.04,100,1);
% Y(1:N/2,2:3) = mvnrnd([0.4,0.4],[0.02,0.01;0.01,0.02],N/2);
% Y(N/2+1:N,2:3) = mvnrnd([0.6,0.6],[0.02,0.01;0.01,0.02],N/2);

%% Simulation - set parameters - get choice
% number of labels
K = 2;
T = 2*2^K-1;
% theta
thetaL = repmat(struct('muAge',0,'sdAge',0,'muInc',0,'sdInc',0),K,1);
thetaL(1).muAge = 0.3;
thetaL(1).sdAge = 0.07;
thetaL(1).muInc = 0.3;
thetaL(1).sdInc = 0.07;
thetaL(2).muAge = 0.7;
thetaL(2).sdAge = 0.07;
thetaL(2).muInc = 0.7;
thetaL(2).sdInc = 0.07;

% betaL(k,:) = [1/price,rate] dim:K*2
betaL = repmat(struct('p',0,'r',0),K,1);
betaL(1).p = 0.9;
betaL(1).r = 0.1;
betaL(2).p = 0.1;
betaL(2).r = 0.9;

% set choice set, each user has T choice sets
C = zeros(N,T,16);
for n = 1:N
    for i = 1:T
        C(n,i,1:15) = randsample(M,15);
    end
end

% get user labels, preferences, and top 3 choices according to the model
[Z,BL,TOP] = model(N,K,T,X,C,Y,thetaL,betaL);
% get the best choice
C(:,:,16) = TOP(:,:,1);

%% Define Sampling Parameters
% hyper parameter, number of labels
K = 2;
% repetition times
R = 1000;
G = 3;
% probability of user n having label k, latent variable
PZ = zeros(R,N,K);
% samples of user characteristics represented by labels
theta = repmat(struct('muAge',0,'sdAge',0,'muInc',0,'sdInc',0),R,K);
% samples of user preferences represented by labels
beta = repmat(struct('p',0,'r',zeros(1,3)),R,K);
% accuracy of sampled model paramters
acc = zeros(100,1);

%% Initialize Sampling Parameters, r = 1
% binary vector matrix with K-1, K dims
A = binaryVecMatrix(K-1);
AA = binaryVecMatrix(K);
% AM: all combinations of labels, AM(k,:,:): all combinations of labels
% with label k
AM = zeros(K,2^(K-1),K);
for k = 1:K
    AM(k,:,:) = [A(:,1:k-1),ones(2^(K-1),1),A(:,k:end)];
end
% user characteristics represented by labels
theta0 = repmat(struct('muAge',0,'sdAge',0,'muInc',0,'sdInc',0),K,1);
% user preferences represented by labels
beta0 = repmat(struct('p',0,'r',0),K,1);
% for k = 1:K
%     % initialize user characteristics represented by labels
%     theta0(k) = struct('muAge',rand,'sdAge',thetaL(k).sdAge,'muInc',rand,'sdInc',thetaL(k).sdInc);
%     % initialize user preferences represented by labels
%     beta0(k) = struct('p',rand,'r',rand); 
% end

theta0(1).muAge = rand;
theta0(1).sdAge = 0.07;
theta0(1).muInc = rand;
theta0(1).sdInc = 0.07;
theta0(2).muAge = rand;
theta0(2).sdAge = 0.07;
theta0(2).muInc = rand;
theta0(2).sdInc = 0.07;
beta0(1).p = 0.95;
beta0(1).r = 0;
beta0(2).p = 0.2;
beta0(2).r = 0.8;

% probability of user n having label k, latent variable
PZ0 = ones(N,K);
for k = 1:K
    PZ0(:,k) = problabel(Y,theta0(k));
end

% probability of user n making the specific 7 choices under all label
% combinations
PC0 = zeros(N,2^K);
% compute linear combined beta under AA(2:end,:) indicated conditons, except
% AA(1,:) which means "having none of the labels"
betatemp0 = ComputeBetatemp(AA(2:end,:),K,beta0);

% betatemp: matrix of 2^K * dim of beta
for n = 1:N
    b0 = rand(1,2);
    betatemp = cat(1,b0,betatemp0);
    PC0(n,:) = probchoice(T,K,n,betatemp,X,C);
end

%% MLE - theta
% fun = @(x)my_mle(theta0,x,K,N,Y,AA,PC0);
% lb = [0,0,0,0];
% ub = [1,1,1,1];
% x = zeros(10,4);
% for i = 1:10
%     x0 = rand(1,4);
%     x(i,:) = fmincon(fun,x0,[],[],[],[],lb,ub);
% end
% plot(x(:,1),x(:,2),'.');
% xlabel('muAge');
% ylabel('muInc');
% title('MLE of theta1,2')
% axis([0,1,0,1]);
% hold on;
% plot(x(:,3),x(:,4),'ro');
% legend('theta1','theta2');
% hold off;
% %% surface - one - beta
% [x,y]=meshgrid(0.1:0.01:1,0.1:0.01:1);
% z = zeros(91);
% for i = 1:91
%     for j = 1:91
%         z(i,j) = my_mle_beta(beta0,[x(i,j),y(i,j)],K,T,N,X,C,AA,PZ0);
%     end
% end
% surfc(x,y,z);
% xlabel('beta1.p');
% ylabel('beta1.r');
% zlabel('-log(likelihood)');
% %% MLE - one - beta
% % global search
% options = optimoptions('fmincon','Display','iter');
% fun = @(x)my_mle_beta(beta0,x,K,T,N,X,C,AA,PZ0);
% problem = createOptimProblem('fmincon','objective',fun,'lb',[0.1,0.1],...
%     'ub',[1,1],'x0',rand(1,2),'options',options);
% gs = GlobalSearch;
% [x,fval] = run(gs,problem);
% % simple fmincon
% % options = optimoptions('fmincon','Display','iter','DiffMinChange',1e-3,'TolX',1e-16);
% % Aueq = [];
% % bueq = [];
% % Aeq = [];
% % beq = [];
% % lb = [0.1,0.1];
% % ub = [1,1];
% % nonlcon = [];
% % x = zeros(10,2);
% % for i = 1:10
% %     x0 = rand(1,2);
% %     [x(i,:),fval,exitflag,output] = fmincon(fun,x0,Aueq,bueq,Aeq,beq,lb,ub,nonlcon,options);
% % end
% 
% %% MLE - beta
% % global search
% x = zeros(10,4);
% for i = 1:10
%     options = optimoptions('fmincon','Display','iter');
%     fun = @(x)my_mle_beta(beta0,x,K,T,N,X,C,AA,PZ0);
%     problem = createOptimProblem('fmincon','objective',fun,'lb',[0.1,0.1,0.1,0.1],...
%     'ub',[1,1,1,1],'x0',rand(1,4),'options',options);   
%     gs = GlobalSearch;
%     [x(i,:),fval] = run(gs,problem);
% end
% plot(x(:,1),x(:,2),'o');
% hold on;
% plot(x(:,3),x(:,4),'*');
% title('MLE of beta - using GlobalSearch, fmincon');
% xlabel('price sensitive');
% ylabel('rate sensitive');
% axis([0,1,0,1]);
% legend('beta1','beta2');
% hold off;
% % options = optimoptions('fmincon','Display','iter','DiffMinChange',1e-3,'TolX',1e-16,'algorithm','sqp');
% % fun = @(x)my_mle_beta(beta0,x,K,T,N,X,C,AA,PZ0);
% % Aueq = [];
% % bueq = [];
% % Aeq = [];
% % beq = [];
% % lb = [0.1,0.1,0.1,0.1];
% % ub = [1,1,1,1];
% % nonlcon = [];
% % % x0 = rand(1,4);
% % % [x,fval,exitflag,output] = fmincon(fun,x0,Aueq,bueq,Aeq,beq,lb,ub,nonlcon,options);
% % x = zeros(10,4);
% % for i = 1:10
% %     x0 = rand(1,4);
% %     [x(i,:),fval,exitflag,output] = fmincon(fun,x0,Aueq,bueq,Aeq,beq,lb,ub,nonlcon,options);
% % end
% 
%% MLE - all
% fun = @(x)my_mle_all(theta0,beta0,x,K,T,N,X,Y,C,AA);
% lb = [0,0,0,0,0.01,0.01,0.01,0.01];
% ub = [1,1,1,1,1,1,1,1];
% x = zeros(10,8);
% for i = 1:10
%     x0 = rand(1,8);
%     x(i,:) = fmincon(fun,x0,[],[],[],[],lb,ub);
% end
% global search
options = optimoptions('fmincon','Display','iter');
fun = @(x)my_mle_all(theta0,beta0,x,K,T,N,X,Y,C,AA);
problem = createOptimProblem('fmincon','objective',fun,'lb',[0,0,0,0,0.01,0.01,0.01,0.01],...
    'ub',[1,1,1,1,1,1,1,1],'x0',[0.3,0.3,0.7,0.7,0.9,0.1,0.1,0.9],'options',options);
gs = GlobalSearch;
[x,fval] = run(gs,problem);
%% Gibbs sampling
% coefficient to be multiplied to xdis
coef = 1;
for r = 1:R
    disp(r);
    for gap = 1:G
        % theta0, beta0
        for k = 1:K
            % all combinations of labels with label k
            Aa = squeeze(AM(k,:,:));
            % metropolis steps
            nsamples = 5;
            % theta0
            % muAge
            flag = 1;
            logpdf = @(x)loglikelihood(flag,K,T,k,x,X,Y,C,PZ0,PC0,theta0,beta0,AA,Aa); % log pdf of target distribution
            proprnd = @(x)randsample(0:0.01:1,1,true,arrayfun(@(xnew) normpdf(xnew,x,0.1),0:0.01:1)); % proposal distribution random number generator, x is old value, xnew is new value
            init = theta0(k).muAge;
            temp = mhsample(init,nsamples,'logpdf',logpdf,'proprnd',proprnd,'symmetric',1);
            theta0(k).muAge = temp(end);
%             % sdAge
%             flag = 2;
%             logpdf = @(x)loglikelihood(flag,K,T,k,x,X,Y,C,PZ0,PC0,theta0,beta0,AA,Aa); % log pdf of target distribution
%             proprnd = @(x)randsample(0.01:0.01:1,1,true,arrayfun(@(xnew) normpdf(xnew,x,0.1),0.01:0.01:1)); % proposal distribution random number generator, x is old value, xnew is new value
%             init = theta0(k).sdAge;
%             temp = mhsample(init,nsamples,'logpdf',logpdf,'proprnd',proprnd,'symmetric',1);
%             theta0(k).sdAge = temp(end);
            % muInc
            flag = 3;
            logpdf = @(x)loglikelihood(flag,K,T,k,x,X,Y,C,PZ0,PC0,theta0,beta0,AA,Aa); % log pdf of target distribution
            proprnd = @(x)randsample(0:0.01:1,1,true,arrayfun(@(xnew) normpdf(xnew,x,0.1),0:0.01:1)); % proposal distribution random number generator, x is old value, xnew is new value
            init = theta0(k).muInc;
            temp = mhsample(init,nsamples,'logpdf',logpdf,'proprnd',proprnd,'symmetric',1);
            theta0(k).muInc = temp(end);
%             % sdInc
%             flag = 4;
%             logpdf = @(x)loglikelihood(flag,K,T,k,x,X,Y,C,PZ0,PC0,theta0,beta0,AA,Aa); % log pdf of target distribution
%             proprnd = @(x)randsample(0.01:0.01:1,1,true,arrayfun(@(xnew) normpdf(xnew,x,0.1),0.01:0.01:1)); % proposal distribution random number generator, x is old value, xnew is new value
%             init = theta0(k).sdInc;
%             temp = mhsample(init,nsamples,'logpdf',logpdf,'proprnd',proprnd,'symmetric',1);
%             theta0(k).sdInc = temp(end);
            % update P(Znk|Yn,theta_k) for all n
            PZ0(:,k) = problabel(Y,theta0(k));
            % beta
            % 1/price
            flag = 5;
            logpdf = @(x)loglikelihood(flag,K,T,k,x,X,Y,C,PZ0,PC0,theta0,beta0,AA,Aa); % log pdf of target distribution
            proprnd = @(x)randsample(0:0.01:1,1,true,arrayfun(@(xnew) normpdf(xnew,x,0.1),0:0.01:1)); % proposal distribution random number generator, x is old value, xnew is new value
            init = beta0(k).p;
            temp = mhsample(init,nsamples,'logpdf',logpdf,'proprnd',proprnd,'symmetric',1);
            beta0(k).p = temp(end);
            % rate
            flag = 6;
            logpdf = @(x)loglikelihood(flag,K,T,k,x,X,Y,C,PZ0,PC0,theta0,beta0,AA,Aa); % log pdf of target distribution
            proprnd = @(x)randsample(0:0.01:1,1,true,arrayfun(@(xnew) normpdf(xnew,x,0.1),0:0.01:1)); % proposal distribution random number generator, x is old value, xnew is new value
            init = beta0(k).r;
            temp = mhsample(init,nsamples,'logpdf',logpdf,'proprnd',proprnd,'symmetric',1);
            beta0(k).r = temp(end);
            % update P(Cn|beta,Zn)) for all Zn where Znk = 1, for all n
            % compute linear combined beta under Aa(label combinations) indicated conditions 
            betatemp = ComputeBetatemp(Aa,K,beta0);
            % update P(Cn|beta,Zn)) under Aa indicated conditions
            for n = 1:N
                PC0(n,ComputeIndex(Aa)) = probchoice(T,K,n,betatemp,X,C);
            end
        end
    end
    % record sampling parameters
    PZ(r,:,:) = PZ0;
    theta(r,:) = theta0;
    beta(r,:) = beta0;
    % calculate model accuracy
    if r > 900
        [~,~,TOPS] = model(N,K,T,X,C,Y,theta0,beta0);
        acctemp = 0;
        for n = 1:N
            for t = 1:T
                acctemp = acctemp + sum(ismember(C(n,t,16),TOPS(n,t,:)));
            end
        end
        acc(r - 900) = acctemp / (T * N);
    end
%     % test convergence
%     len = 5000;
%     if r == 2 * len
%         % theta mean
%         muAgem = mean(reshape([theta(r - len + 1:r,:).muAge],[len,K])); % 1*K
%         sdAgem = mean(reshape([theta(r - len + 1:r,:).sdAge],[len,K]));
%         muIncm = mean(reshape([theta(r - len + 1:r,:).muInc],[len,K]));
%         sdIncm = mean(reshape([theta(r - len + 1:r,:).sdInc],[len,K]));
%         pGenm  = squeeze(mean(reshape([theta(r - len + 1:r,:).pGen],[2,len,K]),2)); % 2*K
%         pOccm = squeeze(mean(reshape([theta(r - len + 1:r,:).pOcc],[3,len,K]),2)); % 3*K
%         % beta mean
%         muRatem = mean(reshape([beta(r - len + 1:r,:).muRate],[len,K])); % 1*K
%         sdRatem = mean(reshape([beta(r - len + 1:r,:).sdRate],[len,K]));
%         muTimem = mean(reshape([beta(r - len + 1:r,:).muTime],[len,K]));
%         sdTimem = mean(reshape([beta(r - len + 1:r,:).sdTime],[len,K]));
%         muYearm = mean(reshape([beta(r - len + 1:r,:).muYear],[len,K]));
%         sdYearm = mean(reshape([beta(r - len + 1:r,:).sdYear],[len,K]));
%         muGenrm = squeeze(mean(reshape([beta(r - len + 1:r,:).muGenr],[3,len,K]),2)); % 3*K
%         covGenrm = squeeze(mean(reshape([beta(r - len + 1:r,:).covGenr],[3,3,len,K]),3)); % 3*3*K
%      else
%         if r > 2 * len
%             % update theta mean
%             muAgem = muAgem + ([theta(r,:).muAge] - [theta(r - len,:).muAge]) ./ len;
%             sdAgem = sdAgem + ([theta(r,:).sdAge] - [theta(r - len,:).sdAge]) ./ len;
%             muIncm = muIncm + ([theta(r,:).muInc] - [theta(r - len,:).muInc]) ./ len;
%             sdIncm = sdIncm + ([theta(r,:).sdInc] - [theta(r - len,:).sdInc]) ./ len;
%             pGenm = pGenm + (reshape([theta(r,:).pGen],[2,K]) - reshape([theta(r - len,:).pGen],[2,K])) ./ len;
%             pOccm = pOccm + (reshape([theta(r,:).pOcc],[3,K]) - reshape([theta(r - len,:).pOcc],[3,K])) ./ len;
%             % update beta mean
%             muRatem = muRatem + ([beta(r,:).muRate] - [beta(r - len,:).muRate]) ./ len;
%             sdRatem = sdRatem + ([beta(r,:).sdRate] - [beta(r - len,:).sdRate]) ./ len;
%             muTimem = muTimem + ([beta(r,:).muTime] - [beta(r - len,:).muTime]) ./ len;
%             sdTimem = sdTimem + ([beta(r,:).sdTime] - [beta(r - len,:).sdTime]) ./ len;
%             muYearm = muYearm + ([beta(r,:).muYear] - [beta(r - len,:).muYear]) ./ len;
%             sdYearm = sdYearm + ([beta(r,:).sdYear] - [beta(r - len,:).sdYear]) ./ len;
%             muGenrm = muGenrm + (reshape([beta(r,:).muGenr],[3,K]) - reshape([beta(r - len,:).muGenr],[3,K])) ./ len;
%             covGenrm = covGenrm + (reshape([beta(r,:).covGenr],[3,3,K]) - reshape([beta(r - len,:).covGenr],[3,3,K])) ./ len;
%             % judge if converge theta
%             if abs([theta(r,:).muAge] - muAgem) < 0.01 .* muAgem
%                 disp('muAge converge');
%             end
%             if abs([theta(r,:).sdAge] - sdAgem) < 0.01 .* sdAgem
%                 disp('sdAge converge');
%             end
%             if abs([theta(r,:).muInc] - muIncm) < 0.01 .* muIncm
%                 disp('muInc converge');
%             end
%             if abs([theta(r,:).sdInc] - sdIncm) < 0.01 .* sdIncm
%                 disp('muAge converge');
%             end
%             if abs(reshape([theta(r,:).pGen],[2,K]) - pGenm) < 0.01 .* pGenm
%                 disp('pGen converge');
%             end
%             if abs(reshape([theta(r,:).pOcc],[3,K]) - pOccm) < 0.01 .* pOccm
%                 disp('pOcc converge');
%             end
%             % judge if converge beta
%         end
%     end
        
end

%% Plot Accuracy
plot(1:100,acc);
axis([0 100 0 1]);
xlabel('iteration');
ylabel('accuracy');
title('Model Accuracy');
%% Plot
% X
plot(X(:,2),X(:,3),'.');
axis([0 1 0 1]);
xlabel('1/price');
ylabel('rate');
title('Choice Set');
% Y
plot(Y(find(Z(:,1)>0),2),Y(find(Z(:,1)>0),3),'*');
xlabel('age');
ylabel('income');
title('Users');
hold on;
plot(Y(find(Z(:,2)>0),2),Y(find(Z(:,2)>0),3),'o');
plot(Y(find(sum(Z,2)==0),2),Y(find(sum(Z,2)==0),3),'.');
legend('Label 1','Label 2','No labels');
plot(thetaL(1).muAge,thetaL(1).muInc,'r.','MarkerSize',20);
plot(thetaL(2).muAge,thetaL(2).muInc,'r.','MarkerSize',20);
hold off;
%% Plot
% betaL
plot(betaL(1,1),betaL(1,2),'k*','MarkerSize',10);
hold on;
plot(betaL(2,1),betaL(2,2),'ko','MarkerSize',10);
axis([0,1,0,1]);
% plot(0.5,0.5,'r.','MarkerSize',20);
xlabel('price sensitive');
ylabel('rate sensitive');
title('Preference');
legend('Label 1','Label 2','user n');
hold off;
%% Plot
% theta beta
figure
% theta label 1
subplot(2,2,1);
plot([theta(:,1).muAge],[theta(:,1).muInc],'.');
axis([0,1,0,1]);
title('label 1');
xlabel('muAge');
ylabel('muInc');
% theta label 2
subplot(2,2,2);
plot([theta(:,2).muAge],[theta(:,2).muInc],'.');
axis([0,1,0,1]);
title('label 2');
xlabel('muAge');
ylabel('muInc');
% beta label 1
subplot(2,2,3);
plot([beta(:,1).p],[beta(:,1).r],'.');
axis([0,1,0,1]);
title('label 1');
xlabel('price sensitive');
ylabel('rate sensitive');
% beta label 2
subplot(2,2,4);
plot([beta(:,2).p],[beta(:,2).r],'.');
axis([0,1,0,1]);
title('label 2');
xlabel('price sensitive');
ylabel('rate sensitive');
%% Plot
% theta beta partly
set = 900:1000;
figure
% theta label 1
subplot(2,2,1);
plot([theta(set,1).muAge],[theta(set,1).muInc],'.');
axis([0,1,0,1]);
title('label 1');
xlabel('muAge');
ylabel('muInc');
% theta label 2
subplot(2,2,2);
plot([theta(set,2).muAge],[theta(set,2).muInc],'.');
axis([0,1,0,1]);
title('label 2');
xlabel('muAge');
ylabel('muInc');
% beta label 1
subplot(2,2,3);
plot([beta(set,1).p],[beta(set,1).r],'.');
axis([0,1,0,1]);
title('label 1');
xlabel('price sensitive');
ylabel('rate sensitive');
% beta label 2
subplot(2,2,4);
plot([beta(set,2).p],[beta(set,2).r],'.');
axis([0,1,0,1]);
title('label 2');
xlabel('price sensitive');
ylabel('rate sensitive');
%% Plot
% theta 1 2
figure
filename = 'theta-1,2.gif';
for r = 1:100
    plot(theta(r,1).muAge,theta(r,1).muInc,'*');
    xlabel('muAge');
    ylabel('muInc');
    axis([0.1 0.8 0.1 0.8])
    hold on;
    plot(theta(r,2).muAge,theta(r,2).muInc,'o');
    hold off;
    drawnow
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if r == 1;
      imwrite(imind,cm,filename,'gif', 'Loopcount',1,'DelayTime',1);
    else
      imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',1);
    end
end

% beta 1 2
figure
filename = 'beta-1,2(1-100).gif';
for r = 1:100
    plot(beta(r,1).p,beta(r,1).r,'*');
    xlabel('1/price');
    ylabel('rate');
    axis([0 1 0 1])
    hold on;
    plot(beta(r,2).p,beta(r,2).r,'o');
    hold off;
    drawnow
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if r == 1;
      imwrite(imind,cm,filename,'gif', 'Loopcount',1,'DelayTime',1);
    else
      imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',1);
    end
end
% theta beta 1 2
figure
filename = 'theta-beta-1,2(4901-5000).gif';
for r = 1:100
    plot(theta(4900+r,1).muAge,theta(4900+r,1).muInc,'b*');
    xlabel('muAge   1/price (theta:b  beta:r)');
    ylabel('muInc   rate');
    axis([0 1 0 1])
    hold on;
    plot(theta(4900+r,2).muAge,theta(4900+r,2).muInc,'bo');
    plot(beta(r,1).p,beta(r,1).r,'r*');
    plot(beta(r,2).p,beta(r,2).r,'ro');
    hold off;
    drawnow
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if r == 1;
      imwrite(imind,cm,filename,'gif', 'Loopcount',1,'DelayTime',1);
    else
      imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',1);
    end
end
%% Plot focus on one user
usid = 1;
% Y
plot(Y(find(Z(:,1)>0),2),Y(find(Z(:,1)>0),3),'*');
xlabel('age');
ylabel('income');
title('Users');
hold on;
plot(Y(find(Z(:,2)>0),2),Y(find(Z(:,2)>0),3),'o');
plot(Y(find(sum(Z,2)==0),2),Y(find(sum(Z,2)==0),3),'.');
plot(Y(usid,2),Y(usid,3),'r.','MarkerSize',20);
legend('Label 1','Label 2','No labels','User 1');
hold off;
% beta
plot(BL(usid,1),BL(usid,2),'r.','MarkerSize',20);
axis([0 1 0 1]);
xlabel('price sensitive');
ylabel('rate sensitive');
title('Preference of user 1');
% choice
plot(X(C(usid,2:16),2),X(C(usid,2:16),3),'.');
axis([0 1 0 1]);
xlabel('1/price');
ylabel('rate');
title('Choice of user 1');
hold on;
plot(X(C(usid,17),2),X(C(usid,17),3),'r.','MarkerSize',20);
legend('Choice set','Choice');
hold off;



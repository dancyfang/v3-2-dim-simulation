function [Z,BL,TOP] = model(N,K,T,X,C,Y,thetaL,betaL)
% convert struct to matrix
beta = zeros(K,2);
for k = 1:K
    beta(k,:) = [betaL(k).p,betaL(k).r];
end

% get user labels
% label indicators, Z(n,k) = 0 or 1
Z = zeros(N,K);
for k = 1:K
    Z(:,k) = problabel(Y,thetaL(k)) > 0.5;
end

% get user preferences and top 3 choices
BL = zeros(N,2);
TOP = zeros(N,T,3);
for n = 1:N
    if sum(Z(n,:)) > 0
        BL(n,:) = Z(n,:) ./ sum(Z(n,:),2) * beta;
    else
        BL(n,:) = rand(1,2);
    end
    % return top 3 choices
    for i = 1:T
        TOP(n,i,:) = makechoice(BL(n,:),X,squeeze(C(n,i,1:15)));
    end
end
end
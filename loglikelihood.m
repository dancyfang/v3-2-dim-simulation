function p = loglikelihood(flag,K,T,k,x,X,Y,C,PZ0,PC0,theta0,beta0,AA,Aa)
N = size(Y,1);
% update parameter
switch flag
    case 1
        theta0(k).muAge = x;
    case 2
        theta0(k).sdAge = x;
    case 3
        theta0(k).muInc = x;
    case 4
        theta0(k).sdInc = x;
    case 5
        beta0(k).p = x;
    case 6
        beta0(k).r = x;
end
switch flag
    case {1,2,3,4}
        % update P(Znk|Yn,theta_k) for all n
        PZ0(:,k) = problabel(Y,theta0(k));
    case {5,6}
        % update P(Cn|beta,Zn)) for all Zn where Znk = 1, for all n
        % compute linear combined beta under Aa(label combinations) indicated conditions 
        betatemp = ComputeBetatemp(Aa,K,beta0);
        % update P(Cn|beta,Zn)) under Aa indicated conditions
        for n = 1:N
                PC0(n,ComputeIndex(Aa)) = probchoice(T,K,n,betatemp,X,C);
        end
end

% p = 1;
% for n = 1:N
%     % P(Z_n|Yn,theta0) for all combination of labels for user n
%     temp = prod(AA .* repmat(PZ0(n,:),2^K,1) + (1-AA) .* repmat((1-PZ0(n,:)),2^K,1), 2);
%     p = p * (PC0(n,:) * temp);
% end

p = 0;
for n = 1:N
    % P(Z_n|Yn,theta0) for all combination of labels for user n
    temp = prod(AA .* repmat(PZ0(n,:),2^K,1) + (1-AA) .* repmat((1-PZ0(n,:)),2^K,1), 2);
    p = p + log(PC0(n,:) * temp);
end

end
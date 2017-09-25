function p = my_mle_beta(beta0,x,K,T,N,X,C,AA,PZ0)
% update parameter
beta0(1).p = x(1);
beta0(1).r = x(2);
% beta0(2).p = x(3);
% beta0(2).r = x(4);

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

p = 0;
for n = 1:N
    % P(Z_n|Yn,theta0) for all combination of labels for user n
    temp = prod(AA .* repmat(PZ0(n,:),2^K,1) + (1-AA) .* repmat((1-PZ0(n,:)),2^K,1), 2);
    p = p + log(PC0(n,:) * temp);
end
p = -p;

end
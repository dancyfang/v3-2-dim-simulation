function p = my_mle_all(theta0,beta0,x,K,T,N,X,Y,C,AA)
% update parameter
theta0(1).muAge = x(1);
theta0(1).muInc = x(2);
theta0(2).muAge = x(3);
theta0(2).muInc = x(4);
beta0(1).p = x(5);
beta0(1).r = x(6);
beta0(2).p = x(7);
beta0(2).r = x(8);

PZ0 = zeros(N,K);
for k = 1:K
    PZ0(:,k) = problabel(Y,theta0(k));
end

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
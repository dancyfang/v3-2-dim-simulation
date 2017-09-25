function p = my_mle(theta0,x,K,N,Y,AA,PC0)
% update parameter
theta0(1).muAge = x(1);
theta0(1).muInc = x(2);
theta0(2).muAge = x(3);
theta0(2).muInc = x(4);
PZ0 = zeros(N,K);
for k = 1:K
    PZ0(:,k) = problabel(Y,theta0(k));
end
p = 0;
for n = 1:N
    % P(Z_n|Yn,theta0) for all combination of labels for user n
    temp = prod(AA .* repmat(PZ0(n,:),2^K,1) + (1-AA) .* repmat((1-PZ0(n,:)),2^K,1), 2);
    p = p + log(PC0(n,:) * temp);
end
p = -p;
end
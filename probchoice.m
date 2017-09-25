function pc = probchoice(T,K,n,betatemp,X,C)
pc = ones(1,size(betatemp,1));
for i = 1:T
    Xt = X(C(n,i,1:16),2:end)'; % dim:2*16
    p1 = exp(betatemp * Xt(:,16)); % dim:size(betatemp,1)*1
    p2 = sum(exp(betatemp * Xt(:,1:15)),2); 
    pc = pc .* (p1 ./ p2)';
end
% 
%     
% % sample each dimension of Bn according to P(Bn|beta,Zn)
% % Bn: (beta_rate,(beta_genr1,beta_genr2,beta_genr3))
% % sample number = M
% M = 50;
% % beta for: rate,(g1,g2,g3)
% size1 = 4;
% % beta for (g1,g2,g3)
% size2 = 3;
% Bn = zeros(M,size1);
% % sample beta_rate from normal mixtures
% mu = Wn * vertcat(beta.muRate); % Wn is a row vector, vertcat(beta.muRate) is [beta_k1.muRate;beta_k2.muRate;beta_k3.muRate]
% sd = Wn .^ 2 * vertcat(beta.sdRate);
% Bn(:,1) = normrnd(mu,sd,M,1);
% % sample (beta_genr1,beta_genr2,beta_genr3) from multivariant-normal mixtures
% mu = Wn * vertcat(beta.muGenr);
% cov = zeros(size2);
% for k = 1:K
%     cov = cov + Wn(k) ^2 .* beta(k).covGenr;
% end
% Bn(:,2:4) = mvnrnd(mu,cov,M);
% % use E(P(C_n|B_n,S_n)) to calculate integration, Monte Carlo integration
% Xt = X(Cn(2:17),2:5)';
% Xcn = Xt(:,16);
% Xsn = Xt(:,1:15);
% P1 = exp(Bn * Xcn);
% P2 = sum(exp(Bn * Xsn),2);
% p = mean(P1 ./ P2);
% 
% end

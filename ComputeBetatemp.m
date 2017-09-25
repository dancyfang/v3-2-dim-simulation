% Compute the linear combined beta under all combinations of labels with Zk = 1 
function betatemp = ComputeBetatemp(Aa,K,beta0)
% combine beta0 into a matrix
betaMt = [beta0(1).p,beta0(1).r];
for i = 2:K
    betaMt = cat(1,betaMt,[beta0(i).p,beta0(i).r]);
end
% all combinations of preferences
betatemp = Aa ./ (sum(Aa,2) * ones(1,size(Aa,2))) * betaMt; % nrow = 2^(K-1)
end
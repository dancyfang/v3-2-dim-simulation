function A = binaryVecMatrix(N)
A = zeros(2^N,N);
for k = 1:N
    r = N - k;
    v = [zeros(2^r,1);ones(2^r,1)];
    A(:,k) = repmat(v,2^(k-1),1);
end
end

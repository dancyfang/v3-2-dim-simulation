% A is a binary matrix, each row is a binary vector which represents a binary number
% index returns the decimal value of each binary number
% eg. A = [0 1 0;0 1 1;1 0 0;1 0 1]
% I = [2;3;4;5]
function I = ComputeIndex(A)
I = A * (2.^(size(A,2)-1:-1:0))';
end



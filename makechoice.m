function c = makechoice(BLn,X,Cn)
[~,ind] = sort(X(Cn,2:end) * BLn','descend');
c = Cn(ind(1:3));
end

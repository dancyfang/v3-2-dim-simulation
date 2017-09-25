function p = proposalpdf(x,y,low,up,sig)
if (x < low || x > up)
    p = 0;
else
    p = normpdf(x,y,sig);
end;
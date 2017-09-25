% users' probabilities of having label k
function p = problabel(Y,theta_k)
% age
p1 = pdf('Normal',Y(:,2),theta_k.muAge,theta_k.sdAge);
% income
p2 = pdf('Normal',Y(:,3),theta_k.muInc,theta_k.sdInc);
p = p1 .* p2;
p0 = pdf('Normal',theta_k.muAge,theta_k.muAge,theta_k.sdAge) * ...
    pdf('Normal',theta_k.muInc,theta_k.muInc,theta_k.sdInc);
% a: set according to (f(3sigma)/f(0))^2a = 0.5
a = 0.07;
p = (p ./ p0).^a;
end
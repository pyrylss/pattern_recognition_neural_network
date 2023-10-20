% [start] Alg. BLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [al,iWout] = uo_BLS(x,d,f,g,almax,almin,rho,c1,c2,iW)
% iWout = 0: al does not satisfy any WC
% iWout = 1: al satisfies (WC1)
% iWout = 2: al satisfies WC (WC1+WC2)
% iWout = 3: al satisfies SWC (WC1+SWC2)
al    = almax;
iWout = 0;
WC1  = @(al) f(x+al*d) <= f(x)+c1*al*g(x)'*d;
WC2  = @(al) g(x+al*d)'*d >= c2*g(x)'*d;
SWC2 = @(al) abs(g(x+al*d)'*d) <= c2*abs(g(x)'*d);

while al > almin
    if WC1(al)
        iWout = 1;
        if SWC2(al)
            iWout = 3;
            break
        elseif WC2(al)
            iWout = 2;
            break
            
        end
    end
            

    al = rho*al;
end

if al <= almin
    iWout = 0;
    al = 0;
end


end
% [end] Alg. BLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = linearize(M, x0, method)
%LINEARIZE  Jacobian of the full ODE at x0 (complex-step default, FD check).
if nargin < 3, method = 'cs'; end
n = M.nx;
A = zeros(n);
switch lower(method)
    case 'cs'
        h = 1e-200;
        for k = 1:n
            xp = x0; xp(k) = xp(k) + 1i*h;
            A(:, k) = imag(full.rhs(xp, M))/h;
        end
    case 'fd'
        for k = 1:n
            hk = 1e-7*max(1, abs(x0(k)));
            xp = x0; xp(k) = xp(k) + hk;
            xm = x0; xm(k) = xm(k) - hk;
            A(:, k) = (full.rhs(xp, M) - full.rhs(xm, M))/(2*hk);
        end
end
L.Asys = A;
L.method = method;
end

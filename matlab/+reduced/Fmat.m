function F = Fmat(s, w0)
%FMAT  Frequency-shaping matrix F(s).
%   F = [[beta, alpha], [-alpha, beta]], alpha = w0^2/(s^2+w0^2),
%   beta = s*w0/(s^2+w0^2). Complex equivalent w0/(s+j*w0); F(0) = -J2;
%   F = beta*I2 - alpha*J2 (commutes with rotations).
den = s.^2 + w0^2;
% guard the structural singularity at s = +-j*w0 (rotating-frame image of
% the stationary-frame fundamental). Callers must never evaluate there. A
% clear error is better than a silent Inf/NaN inside a root finder.
assert(abs(den) > 1e-9*w0^2, 'reduced:Fmat:singular', ...
    'F(s) evaluated at the singular point s = +-j*w0 (s = %g%+gj)', real(s), imag(s));
a = w0^2 ./ den;
b = s*w0 ./ den;
F = [b, a; -a, b];
end

function [Yd, parts] = Ydelta(s, Leff, v0, i0, c, wb)
%YDELTA  GFM synchronization-loop admittance, closed form.
%
%   Y_delta(s) = -(G/(1 + G c)) m w^T          (rank one)
%   m = J2 i0 - Ydq(s) J2 v0,  w = i0 + Ydq(s)^T v0,  c = v0^T m,
%   G(s) = F_P(s)/(J_v s^2 + D_v s),  F_P = kP/(1 + TP s),
%   Ydq(s) = -(1/(w0 Leff)) F(s)     (injection convention; with this
%   sign c(0) = Q0 + V0^2/(w0 Leff) > 0).
%
%   v0, i0 : local-frame operating point (theta_j0 = 0 alignment)
%   c      : converter params (kP, TP, Hv, Dv on the converter base)
%   wb     : w0 [rad/s]
%
%   All converter-base pu. F commutes with rotations, so Ydq is frame
%   invariant; m, w carry the frame through v0, i0.

J2 = [0, -1; 1, 0];
Ydq = -reduced.Fmat(s, wb)/(wb*Leff);

Jv = 2*c.Hv/wb;                 % J_v = 2Hv/w0
Dv = c.Dv/wb;                   % D_v = device Dv / w0 ((1) of the paper divided by w0)

m = J2*i0 - Ydq*(J2*v0);
w = i0 + Ydq.'*v0;
cs = v0.'*m;

if s == 0
    % exact s -> 0 limit: G ~ kP/(Dv s) -> inf, so G/(1+Gc) -> 1/c and
    % Y_delta(0) = -m w^T / c with the (real) s = 0 constituents
    % (Ydq(0) = J2/(w0 Leff)). The exact static-root family of the analytic
    % threshold uses it. It agrees with the evaluation at s = 1e-9.
    G = Inf;
    Yd = -(m*w.')/cs;
else
    FP = c.kP/(1 + c.TP*s);
    G  = FP/(Jv*s^2 + Dv*s);
    Yd = -(G/(1 + G*cs)) * (m*w.');
end
if nargout > 1
    parts = struct('m', m, 'w', w, 'c', cs, 'G', G, 'Ydq', Ydq);
end
end

function G = gfm_red_ss(prt, Leff, wb)
%GFM_RED_SS  State-space of the GFM SYNCHRONIZATION admittance Y_delta ONLY.
%   This is the closed form of Ydelta.m. The inner Y_dq = -F/(w0 Leff) is
%   left out on purpose: it sits on the network side as B_in (the split of
%   the reduced model). Realized exactly as
%   Y_delta = (inner + sync) - (pure inner): 7 states.
%
%   Block 1 (5 states): inner + sync, local frame
%     dxi = -w0 J2 xi + (dV_loc - J2 v0 dth)/Leff,  I1_loc = -xi + J2 i0 dth
%     dPf = (kP dPe - Pf)/TP, dPe = i0' dV_loc + v0' dI1_loc
%     Jv dom = -Pf - Dv_man om,  dth = om
%   Block 2 (2 states): pure inner, SUBTRACTED
%     dxi2 = -w0 J2 xi2 + dV_loc/Leff,              I2_loc = -xi2
%   Output: dI_common = Rr (I1_loc - I2_loc).  Converter-base pu.
J2 = [0 -1; 1 0];
c = prt.cvt;
v0 = prt.v0;  i0 = prt.i0;
Jv = 2*c.Hv/wb;  Dv = c.Dv/wb;
Rr = reduced.rot2(prt.th0);

A = zeros(7); B = zeros(7, 2);
ixi = 1:2; iPf = 3; iom = 4; ith = 5; ix2 = 6:7;
A(ixi, ixi) = -wb*J2;
B(ixi, :)   = (1/Leff)*Rr.';
A(ixi, ith) = -(J2*v0)/Leff;
A(iPf, ixi) = -(c.kP/c.TP)*(v0.');
B(iPf, :)   = (c.kP/c.TP)*(i0.'*Rr.');
A(iPf, ith) = (c.kP/c.TP)*(v0.'*(J2*i0));
A(iPf, iPf) = -1/c.TP;
A(iom, iPf) = -1/Jv;
A(iom, iom) = -Dv/Jv;
A(ith, iom) = 1;
A(ix2, ix2) = -wb*J2;
B(ix2, :)   = (1/Leff)*Rr.';

C = zeros(2, 7); D = zeros(2, 2);
C(:, ixi) = -Rr;
C(:, ith) =  Rr*(J2*i0);
C(:, ix2) = +Rr;                      % subtract the pure-inner injection

G.A = A; G.B = B; G.C = C; G.D = D;
end

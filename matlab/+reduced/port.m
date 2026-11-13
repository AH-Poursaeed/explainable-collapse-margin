function prt = port(dtype, sp, Vpcc, Sinj_conv, freeze_sync)
%PORT  2x2 terminal-admittance state space of one standalone converter.
%
%   Linearizes the device RHS at its operating point. The input is the PCC
%   voltage and the output is the injected current, both in the COMMON
%   frame (the device's internal rotation by theta(t) absorbs the frame
%   change by itself). CONVERTER-base pu.
%
%   dtype        : 'GFL' | 'VSM'
%   sp           : device param struct (cvt with .Ssys == .Sn for 1:1 base)
%   Vpcc         : PCC voltage phasor (pu)
%   Sinj_conv    : injected complex power, converter-base pu
%   freeze_sync  : (VSM only) true -> delete theta and w states, giving the
%                  inner-loop admittance Y_dq (finite at j*w0)
%
%   Returns prt with A,B,C,D and Y = @(s) 2x2 admittance (injection
%   convention: dI_inj = Y dV), plus the local-frame operating point
%   (v0, i0, P0, Q0, th0) consumed by the Y_delta closed form.

if nargin < 5, freeze_sync = false; end
assert(abs(sp.cvt.Ssys - sp.cvt.Sn) < 1e-9, 'reduced:port:base', ...
    'port extraction requires converter-base params (set c.Ssys = c.Sn)');

[xd, ref, s] = dev.init_device(dtype, sp, Vpcc, Sinj_conv);
sp.ref = ref;

switch upper(dtype)
    case 'GFL', rhsf = @(x, vd, vq) dev.gfl_rhs(x, vd, vq, sp);
    case 'VSM', rhsf = @(x, vd, vq) dev.gfm_vsm_rhs(x, vd, vq, sp);
end
V0d = real(Vpcc); V0q = imag(Vpcc);

% residual gate on the standalone device
[f0, ID0, IQ0] = rhsf(xd, V0d, V0q);
prt.residual = norm(f0, inf);

n = numel(xd); h = 1e-200;
A = zeros(n); B = zeros(n, 2); C = zeros(2, n); D = zeros(2, 2);
for k = 1:n
    xp = xd; xp(k) = xp(k) + 1i*h;
    [fk, IDk, IQk] = rhsf(xp, V0d, V0q);
    A(:, k) = imag(fk)/h;
    C(:, k) = imag([IDk; IQk])/h;
end
[f1, ID1, IQ1] = rhsf(xd, V0d + 1i*h, V0q);
B(:, 1) = imag(f1)/h;  D(:, 1) = imag([ID1; IQ1])/h;
[f2, ID2, IQ2] = rhsf(xd, V0d, V0q + 1i*h);
B(:, 2) = imag(f2)/h;  D(:, 2) = imag([ID2; IQ2])/h;

if freeze_sync
    k = sp.k;
    drop = [k.theta, k.w];
    keep = setdiff(1:n, drop);
    A = A(keep, keep);  B = B(keep, :);  C = C(:, keep);
end

prt.A = A; prt.B = B; prt.C = C; prt.D = D;
prt.Y = @(sv) C*((sv*eye(size(A,1)) - A)\B) + D;

% local-frame operating point for the closed form: terminal voltage and injected
% current in the device's local frame (theta_j0 = 0 alignment)
prt.th0 = s.th;
prt.v0  = [s.vpd; s.vpq];
prt.i0  = [s.igd; s.igq];
prt.P0  = s.Ppcc;  prt.Q0 = s.Qpcc;
prt.ref = ref;  prt.xd = xd;  prt.s = s;
end

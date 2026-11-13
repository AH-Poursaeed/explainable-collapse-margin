function out = smib(type, Xline, opt)
%SMIB  Single converter behind an RL line to an infinite bus (18 states).
%   A test bed for one device in the limit case. It uses the same dev.* RHS
%   as the system model, a PCC shunt capacitor, an RL line and an infinite bus
%   with a FIXED phasor. It also serves as a cross-check of the analytic
%   threshold.
%
%   type  : 'GFL' | 'VSM'
%   Xline : line reactance, CONVERTER-base pu (SCR = 1/Xline at V = 1)
%   opt   : .P0 (default 0.9), .Q0 (default 0.1), .Rline (default Xline/10),
%           .Bc (PCC shunt, default 0.02), .Sn (default 500 MVA), .tune,
%           .Vinf (default 1.0)
%
%   States: [dev(14); iline(2); vpcc(2)]  (line: PCC -> infinite bus)
%   Returns eigenvalues, sigma_dom/zeta_min (sub-synchronous band), residual.

if nargin < 3, opt = struct(); end
P0 = getdef(opt,'P0',0.9); Q0 = getdef(opt,'Q0',0.1);
Rl = getdef(opt,'Rline',Xline/10);
Bc = getdef(opt,'Bc',0.02);
Sn = getdef(opt,'Sn',500);
Vinf = getdef(opt,'Vinf',1.0);
tune = getdef(opt,'tune',struct());

P = params.sys_const();
c = params.converter(type, Sn, P, tune);
c.Ssys = Sn;                              % harness works 1:1 on the converter base
sp.cvt = c; sp.k = dev.state_map(type); sp.nx = c.nx; sp.Ssys = Sn;
wb = P.wb;

% ---- operating point: converter injects P0 and HOLDS |Vpcc| = Vhold --------
% This is the PV-bus convention, as for the plants in the 39-bus system.
% With it, equilibria exist down to very low SCR, where a fixed-Q injection
% reaches the loadability nose. Unknowns: angle(Vpcc) and Q0. Newton, 2x2.
Vhold = getdef(opt, 'Vhold', 1.0);
Z = Rl + 1j*Xline;
th_v = asin(min(0.95, P0*Xline/(Vhold*Vinf)));   % seed
Q0 = getdef(opt, 'Q0seed', Q0);
u = [th_v; Q0];
gfun = @(u) resid2(u, P0, Vhold, Vinf, Z, Bc);
for it = 1:80
    g0 = gfun(u);
    if norm(g0) < 1e-14, break; end
    h = 1e-7;
    J = [gfun(u+[h;0])-g0, gfun(u+[0;h])-g0]/h;
    du = - J \ g0;
    % damped step
    a = 1;
    while a > 1e-4 && norm(gfun(u + a*du)) > norm(g0)
        a = a/2;
    end
    u = u + a*du;
end
assert(norm(gfun(u)) < 1e-10, 'checks:smib:op', ...
    'operating-point solve failed (|g|=%.1e)', norm(gfun(u)));
th_v = u(1);  Q0 = u(2);
Vp = Vhold*exp(1j*th_v);
Iline0 = conj((P0 + 1j*Q0)/Vp) - 1j*Bc*Vp;
out.Q0 = Q0;

% ---- device init ------------------------------------------------------------
[xd, ref] = dev.init_device(type, sp, Vp, P0 + 1j*Q0);
sp.ref = ref;

x0 = [xd; real(Iline0); imag(Iline0); real(Vp); imag(Vp)];
prm = struct('sp',sp,'type',type,'Rl',Rl,'Xl',Xline,'Bc',Bc,'wb',wb,'Vinf',Vinf);

f0 = checks.smib_rhs(x0, prm);
out.residual = norm(f0, inf);

% ---- linearize (complex step) -----------------------------------------------
n = numel(x0); A = zeros(n); h = 1e-200;
for k = 1:n
    xp = x0; xp(k) = xp(k) + 1i*h;
    A(:,k) = imag(checks.smib_rhs(xp, prm))/h;
end
out.A = A;
lam = eig(A);
out.lam = lam;
% no zero mode here (the infinite bus anchors the angle)
inband = abs(imag(lam)) <= 2*pi*55;
lamB = lam(inband);
[out.sigma_dom, ii] = max(real(lamB));
out.lam_dom = lamB(ii);
zeta = -real(lamB)./max(abs(lamB),eps);
out.zeta_min = min(zeta);
out.f_dom = abs(imag(out.lam_dom))/(2*pi);
out.x0 = x0; out.prm = prm; out.Vpcc = Vp;
end

function v = getdef(s, f, v)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); end
end

function g = resid2(u, P0, Vhold, Vinf, Z, Bc)
% KCL residual with |V| held: V = Vhold e^{j th}, injection S = P0 + jQ.
V = Vhold*exp(1j*u(1));  S = P0 + 1j*u(2);
gz = V - Vinf - Z*(conj(S)/conj(V) - 1j*Bc*V);
g = [real(gz); imag(gz)];
end

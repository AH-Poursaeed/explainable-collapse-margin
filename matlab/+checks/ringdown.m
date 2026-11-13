function rd = ringdown(out, which, amp, Tsim)
%RINGDOWN  Nonlinear time-domain cross-check of an eigenvalue prediction.
%
%   Perturbs the equilibrium along the real part of the target mode's right
%   eigenvector. Integrates the FULL NONLINEAR model (ode15s, analytic
%   Jacobian = Asys near the equilibrium). Projects the trajectory onto the
%   mode's left eigenvector, and fits sigma and omega of the modal
%   coordinate z(t) ~ z0 e^{lambda t}. The check passes when |Dsigma|/|sigma|
%   and |Domega|/omega are within a few percent: the linear and the
%   nonlinear models then agree.
%
%   out   : run_scenario output (with Asys, x0, M, S)
%   which : 'zeta' (the zeta_min mode, default) or 'dom' (sigma_dom mode)
%   amp   : perturbation amplitude (default 1e-4)
%   Tsim  : simulation window [s] (default 6)

if nargin < 2, which = 'zeta'; end
if nargin < 3, amp = 1e-4; end
if nargin < 4, Tsim = 6; end

A = out.Asys;  x0 = out.x0;  M = out.M;  S = out.S;
switch which
    case 'zeta', lam_t = S.lam_zeta;
    case 'dom',  lam_t = S.lam_dom;
end

[V, D] = eig(A);
lam = diag(D);
[~, im] = min(abs(lam - lam_t));
phi = V(:, im);
W = inv(V);
w_left = W(im, :).';

dx0 = amp * real(phi)/norm(real(phi));
odef = @(t, x) full.rhs(x, M);
opts = odeset('RelTol', 1e-9, 'AbsTol', 1e-11, 'Jacobian', @(t,x) A);
[tt, xx] = ode15s(odef, [0 Tsim], x0 + dx0, opts);

% modal coordinate
z = (xx - x0.') * conj(w_left);
% fit on a window after the fast transients (skip the first 5% of Tsim,
% stop when |z| decays below 1e-3 of its start to stay above noise)
i0 = find(tt > 0.05*Tsim, 1);
zs = abs(z)/abs(z(i0));
i1 = find(zs(i0:end) < 1e-3, 1);
if isempty(i1), i1 = numel(tt); else, i1 = i0 + i1 - 1; end
seg = i0:i1;
% sigma from log-magnitude slope; omega from unwrapped phase slope
pmag = polyfit(tt(seg), log(abs(z(seg))), 1);
sig_fit = pmag(1);
ph = unwrap(angle(z(seg)));
pph = polyfit(tt(seg), ph, 1);
om_fit = abs(pph(1));

rd.lam_pred = lam(im);
rd.sig_fit = sig_fit;
rd.om_fit = om_fit;
rd.err_sig = abs(sig_fit - real(lam(im)))/max(abs(real(lam(im))), 1e-6);
rd.err_om  = abs(om_fit - abs(imag(lam(im))))/max(abs(imag(lam(im))), 1e-6);
rd.t = tt; rd.z = z; rd.seg = seg;
end

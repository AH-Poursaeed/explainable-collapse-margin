function q3_beyond_fold
%Q3_BEYOND_FOLD  The operating equilibrium through and beyond the fold.
%   It follows the operating equilibrium through k* = 1.9730. It also tests
%   whether unstable-mode seeding explains the straddle onset times: k = 1.98
%   diverges at 2.20 s and k = 2.00 at 4.86 s, although the unstable
%   eigenvalue of k = 2.00 is 4x larger.
%
%   Part 1  continuation: k = 1.90 : 0.005 : 2.05 on the design ray.
%     Records PF success, polished-equilibrium residual, state-branch
%     continuity |x0(k) - x0(k_prev)|_inf, sigma_dom, n unstable modes,
%     and the dominant-eigenvector continuity (|<v_k, v_prev>|).
%   Part 2  projection, k in {1.98, 2.00}: integrate the reference pulse to
%     its end (RK4, 5 us) and expand x(te) - x0 in the eigenbasis. Record
%     the unstable-mode coefficient (the seeding), the first-order predicted
%     amplitude of the terminal-|V| deviation, and the predicted onset time
%     to the 0.5 pu divergence threshold. Also dG = 0.15/0.30 at k = 1.98
%     for the ln-4 delay rule.
%
%   Outputs: results/final/tables/q3_continuation.csv, q3_projection.csv
%

maxNumCompThreads(2);
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'final', 'tables');
P = params.sys_const();                                        %#ok<NASGU>
dv = repmat({'GFL'}, 1, 10);  dv{9} = 'VSM';  dv{10} = 'VSM';
PULSE = struct('bus', 31, 'dG', 0.60, 't0', 0.05, 'dur', 0.020);

% ---- Part 1: continuation through the fold ---------------------------------
f1 = fullfile(outdir, 'q3_continuation.csv');
if ~isfile(f1)
    ks = 1.90:0.005:2.05;
    rows = {};
    xprev = [];  vprev = [];
    for i = 1:numel(ks)
        cfg = struct('devtype', {dv}, 'kL', 1.0, 'k', ks(i));
        o = run_scenario(cfg);
        r = struct('k', ks(i), 'pf_ok', o.pf_ok, 'resid', NaN, ...
            'resid_pu', NaN, 'sigma_dom', NaN, 'f_dom', NaN, ...
            'n_unstable', NaN, 'dx_prev', NaN, 'vdot_prev', NaN, ...
            'Vmin_bus', NaN, 'fail', string(''));
        if isfield(o, 'fail') && ~isempty(o.fail)
            r.fail = string(o.fail);
            if isfield(o, 'residual'), r.resid = o.residual; end
        else
            r.resid = o.residual;  r.resid_pu = o.residual_pu;
            r.sigma_dom = o.S.sigma_dom;  r.f_dom = o.S.f_dom;
            lam = o.S.lam;
            r.n_unstable = nnz(real(lam) > 1e-9 & abs(lam) > 1e-6);
            r.Vmin_bus = min(o.res.bus(:, 8));
            if ~isempty(xprev)
                r.dx_prev = max(abs(o.x0 - xprev));
            end
            lr = real(lam);  lr(abs(lam) < 1e-6) = -inf;   % skip the zero mode
            [~, iu] = max(lr);
            vd = o.S.V(:, iu);
            if ~isempty(vprev)
                r.vdot_prev = abs(vprev'*vd)/(norm(vprev)*norm(vd));
            end
            xprev = o.x0;  vprev = vd;
        end
        rows{end+1} = r;                                       %#ok<AGROW>
        fprintf('[cont k=%.3f] pf=%d fail=%s sigma=%+.4f resid=%.2e dx=%.3g\n', ...
            ks(i), r.pf_ok, r.fail, r.sigma_dom, r.resid, r.dx_prev);
    end
    writetable(struct2table([rows{:}]), f1);
    fprintf('saved %s\n', f1);
end

% ---- Part 2: eigenbasis projection of the pulse-end state ------------------
f2 = fullfile(outdir, 'q3_projection.csv');
if ~isfile(f2)
    cases = {2.00, 0.60; 1.98, 0.60; 1.98, 0.30; 1.98, 0.15; 1.97, 0.60};
    rows = {};
    for c = 1:size(cases, 1)
        kk = cases{c, 1};  dG = cases{c, 2};
        cfg = struct('devtype', {dv}, 'kL', 1.0, 'k', kk);
        o = run_scenario(cfg);
        assert(isempty(o.fail), o.fail);
        A = full(o.Asys);
        [V, D] = eig(A);
        lam = diag(D);
        W = inv(V);                       % rows = left eigvecs (dual basis)
        lr = real(lam);  lr(abs(lam) < 1e-6) = -inf;       % skip the zero mode
        [sig_u, iu] = max(lr);
        % pulse to end (RK4 at 5 us, to stay accurate through the forcing window)
        PU = PULSE;  PU.dG = dG;
        cs = mkcase(o, PU, dG);
        te = PU.t0 + PU.dur + 0.01;       % 10 ms after pulse end
        [~, XX] = rk4(cs, 5e-6, te);
        dx = XX(:, end) - o.x0;
        coef = W*dx;                      % eigenbasis expansion
        cu = coef(iu);
        % first-order terminal-|V| amplitude of the unstable mode component
        vamp = vterm_lin(V(:, iu), o);    % per-plant |dV| for unit coefficient
        a0 = abs(cu)*max(vamp);
        t_pred = te + log(0.5/max(a0, realmin))/max(sig_u, realmin);
        % share of the perturbation in the unstable mode (energy fraction)
        share = abs(cu)*norm(V(:, iu))/max(norm(dx), realmin);
        rows{end+1} = struct('k', kk, 'dG', dG, 'sigma_u', sig_u, ...
            'n_unstable', nnz(real(lam) > 1e-9 & abs(lam) > 1e-6), ...
            'cu_abs', abs(cu), 'a0_vterm', a0, 'share', share, ...
            'te', te, 't_pred_dev05', t_pred, 'norm_dx', norm(dx, inf)); %#ok<AGROW>
        fprintf('[proj k=%.2f dG=%.2f] sig_u=%+.4f |cu|=%.3e a0=%.3e share=%.3e t_pred=%.2f s\n', ...
            kk, dG, sig_u, abs(cu), a0, share, t_pred);
    end
    writetable(struct2table([rows{:}]), f2);
    fprintf('saved %s\n', f2);
end
fprintf('q3_beyond_fold complete.\n');
end

% ------------------------------------------------------------------------
function cs = mkcase(o, PULSE, dG)
cs.M = o.M;  cs.x0 = o.x0;  cs.A = full(o.Asys);
b = PULSE.bus;
cs.ivb = [o.M.ix_bus(2*b-1); o.M.ix_bus(2*b)];
cs.gfac = o.M.N.wb / o.M.N.Bc(b) * dG;
cs.t0 = PULSE.t0;  cs.dur = PULSE.dur;
end

function s = pshape(t, cs)
if t >= cs.t0 && t <= cs.t0 + cs.dur
    s = 0.5*(1 - cos(2*pi*(t - cs.t0)/cs.dur));
else
    s = 0;
end
end

function f = frhs(t, x, cs)
f = full.rhs(x, cs.M);
s = pshape(t, cs);
if s > 0 && cs.gfac ~= 0
    f(cs.ivb) = f(cs.ivb) - s*cs.gfac*x(cs.ivb);
end
end

function [tt, XX] = rk4(cs, dt, T)
nst = round(T/dt);
keep = max(1, floor(nst/2000));
n = numel(cs.x0);
tt = zeros(1, ceil(nst/keep) + 1);  XX = zeros(n, ceil(nst/keep) + 1);
x = cs.x0;  tt(1) = 0;  XX(:, 1) = x;  m = 1;
for q = 1:nst
    t = (q-1)*dt;
    k1 = frhs(t, x, cs);
    k2 = frhs(t + dt/2, x + dt/2*k1, cs);
    k3 = frhs(t + dt/2, x + dt/2*k2, cs);
    k4 = frhs(t + dt, x + dt*k3, cs);
    x = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);
    if mod(q, keep) == 0
        m = m + 1;  tt(m) = q*dt;  XX(:, m) = x;
    end
end
if tt(m) < T - dt/2
    m = m + 1;  tt(m) = T;  XX(:, m) = x;
end
tt = tt(1:m);  XX = XX(:, 1:m);
end

function vamp = vterm_lin(v, o)
%VTERM_LIN first-order per-plant terminal-|V| deviation for a unit modal
%   coefficient: d|V| = (vd0*dvd + vq0*dvq)/|V0| per plant.
vamp = zeros(10, 1);
for p = 1:10
    b = o.plant.bus(p);
    i1 = o.M.ix_bus(2*b-1);  i2 = o.M.ix_bus(2*b);
    vd0 = o.x0(i1);  vq0 = o.x0(i2);
    V0 = hypot(vd0, vq0);
    vamp(p) = abs(vd0*real(v(i1)) + vq0*real(v(i2)))/max(V0, eps);
end
end

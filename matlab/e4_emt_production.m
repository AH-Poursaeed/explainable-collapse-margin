function e4_emt_production
%E4_EMT_PRODUCTION  Run the EMT checks, the boundary traces and the security map.
%   The integrator is the same checked ETDRK2 integrator as in fs4_emt.m
%   (Cox & Matthews 2002). This script runs its check ladder V0-V3 again
%   and saves the result.
%   Three parts:
%     1. gates    : V0 phi construction, V1 equilibrium hold, V2 dt halving,
%                   V3 cross-check with classical RK4  -> e4_gates.mat
%     2. straddle : terminal-voltage traces for k = 1.95/1.96/1.97/1.98/2.00,
%                   with 8 s windows on the collapse side so that the plot
%                   shows the whole collapse        -> e4_straddle.mat
%     3. security : outcome map over disturbance size x k on both sides of
%                   the fold (k* = 1.9730). The divergence detector looks only
%                   AFTER the pulse, because the forcing dip of large pulses
%                   would trip a detector over the whole window by mistake.
%                                                    -> e4_security_map.csv
%
%   Disturbance: a 20 ms raised-cosine load-conductance pulse at bus 31
%   (the reference disturbance), dG in system pu.
%   The script runs serially on 2 threads, so that it can share the machine
%   with a parallel batch.

maxNumCompThreads(2);
P = params.sys_const();                                        %#ok<NASGU>
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'final');
if ~exist(outdir, 'dir'), mkdir(outdir); end
dv = repmat({'GFL'}, 1, 10);  dv{9} = 'VSM';  dv{10} = 'VSM';  % buses 38, 39
PULSE = struct('bus', 31, 'dG', 0.60, 't0', 0.05, 'dur', 0.020);
DT = 5e-5;
TPOST = PULSE.t0 + PULSE.dur + 0.05;    % start of the classification window after the pulse

% ---- Part 1: integrator checks (gate ladder) ----------------------------------
fprintf('=== E4 part 1: integrator gates ===\n');
rng(1); Atest = randn(6) - 3*eye(6);
[Et, p1t, p2t] = phis(Atest, 0.01);
E_err = norm(Et - expm(Atest*0.01));
p1_err = norm(p1t - series_phi(Atest*0.01, 1));
p2_err = norm(p2t - series_phi(Atest*0.01, 2));
fprintf('[V0] phi gates: E %.2e, phi1 %.2e, phi2 %.2e (all < 1e-12)\n', ...
    E_err, p1_err, p2_err);
assert(E_err < 1e-12 && p1_err < 1e-12 && p2_err < 1e-12);

o = prep(dv, 1.95);
fprintf('[gate case] k=1.950 sigma_dom=%+.4f\n', o.S.sigma_dom);
cs = mkcase(o, PULSE, 0);
t1 = tic;
[~, X1] = etdrk2(cs, DT, 1.0, false);
drift = max(abs(X1(:, end) - o.x0));
fprintf('[V1] 1 s equilibrium hold: max drift %.3e (gate < 1e-8), %.1f s wall\n', drift, toc(t1));

cs = mkcase(o, PULSE, PULSE.dG);
[~, Xa] = etdrk2(cs, DT,   0.5, false);
[~, Xb] = etdrk2(cs, DT/2, 0.5, false);
dhalf = max(abs(Xa(:, end) - Xb(:, end)))/max(1, max(abs(Xb(:, end))));
fprintf('[V2] dt halving %g -> %g: end-state rel diff %.3e (gate < 1e-5)\n', DT, DT/2, dhalf);

t3 = tic;
[~, Xr] = rk4(cs, 5e-6, 0.12);
[~, Xe] = etdrk2(cs, DT, 0.12, false);
dref = max(abs(Xe(:, end) - Xr(:, end)))/max(abs(Xr(:, end)));
fprintf('[V3] ETD vs RK4(5e-6) at t=0.12 s: rel diff %.3e (gate < 1e-4), RK4 wall %.1f s\n', ...
    dref, toc(t3));
Vt = vterm(Xe, o);
dip = max(abs(Vt - Vt(:, 1)), [], 'all');
fprintf('[V3b] worst terminal |V| deviation in pulse window: %.4f pu\n', dip);
gates = struct('E_err', E_err, 'p1_err', p1_err, 'p2_err', p2_err, ...
    'drift', drift, 'dhalf', dhalf, 'dref', dref, 'dip', dip, 'dt', DT);
save(fullfile(outdir, 'e4_gates.mat'), '-struct', 'gates');

% ---- Part 2: traces on both sides of the boundary -----------------------------
fprintf('=== E4 part 2: boundary straddle (8 s collapse windows) ===\n');
ks   = [1.95 1.96 1.97 1.98 2.00];
Tsim = [3.0  6.0  4.0  8.0  8.0];
C = struct('k', {}, 'sigma_pred', {}, 'outcome', {}, 'tdiv', {}, ...
           'enddev', {}, 'wall_s', {}, 't', {}, 'Vterm', {});
for i = 1:numel(ks)
    o = prep(dv, ks(i));
    cs = mkcase(o, PULSE, PULSE.dG);
    tw = tic;
    [tt, XX] = etdrk2(cs, DT, Tsim(i), true);
    wall = toc(tw);
    Vt = vterm(XX, o);
    [oc, tdiv, endd] = classify(tt, Vt, XX, TPOST);
    C(end+1) = struct('k', ks(i), 'sigma_pred', o.S.sigma_dom, ...
        'outcome', oc, 'tdiv', tdiv, 'enddev', endd, 'wall_s', wall, ...
        't', tt, 'Vterm', Vt);                                 %#ok<AGROW>
    fprintf('[k=%.3f] sigma=%+.4f -> %-12s (end dev %.4g, tdiv %.2f s, wall %.0f s)\n', ...
        ks(i), o.S.sigma_dom, oc, endd, tdiv, wall);
end
straddle.C = C;  straddle.pulse = PULSE;  straddle.dt = DT;   %#ok<STRNU>
save(fullfile(outdir, 'e4_straddle.mat'), '-struct', 'straddle');
fprintf('saved e4_straddle.mat\n');

% ---- Part 3: security map -----------------------------------------------------
fprintf('=== E4 part 3: security map (disturbance size x k) ===\n');
kmap  = [1.90 1.93 1.95 1.96 1.965 1.97 1.975 1.98];
dGmap = [0.15 0.30 0.60 0.90 1.20 1.80];
TMAP = 8.0;
rows = {};
for i = 1:numel(kmap)
    o = prep(dv, kmap(i));
    fprintf('-- k=%.3f (sigma=%+.4f)\n', kmap(i), o.S.sigma_dom);
    for j = 1:numel(dGmap)
        PU = PULSE;  PU.dG = dGmap(j);
        cs = mkcase(o, PU, PU.dG);
        tw = tic;
        [tt, XX] = etdrk2(cs, DT, TMAP, true);
        wall = toc(tw);
        Vt = vterm(XX, o);
        [oc, tdiv, endd, dipj] = classify(tt, Vt, XX, TPOST);
        rows{end+1} = struct('k', kmap(i), 'dG', dGmap(j), ...
            'sigma_pred', o.S.sigma_dom, 'outcome', string(oc), ...
            'tdiv', tdiv, 'enddev', endd, 'pulse_dip', dipj, ...
            't_end', tt(end), 'wall_s', wall);                 %#ok<AGROW>
        fprintf('   dG=%.2f -> %-12s (tdiv %.2f, enddev %.4g, dip %.3f, wall %.0f s)\n', ...
            dGmap(j), oc, tdiv, endd, dipj, wall);
    end
end
Tm = struct2table([rows{:}]);
writetable(Tm, fullfile(outdir, 'e4_security_map.csv'));
fprintf('saved e4_security_map.csv (%d cells)\ne4 done.\n', height(Tm));
end

% ==============================================================================
function [oc, tdiv, endd, dip] = classify(tt, Vt, XX, tpost)
%CLASSIFY  Outcome of a run after the pulse.
%   The deviation is measured from the state before the pulse. Only
%   t >= tpost counts for the divergence test, because the forcing dip of
%   large pulses would trip a detector over the whole window by mistake.
%   A slow escape that the window cuts short (the terminal deviation still
%   grows at the end of the window, as at k = 2.00) is classified 'escaping'.
dev = max(abs(Vt - Vt(:, 1)), [], 1);
post = tt >= tpost;
dip = max(dev(~post & tt > 0), [], 'omitnan');
if isempty(dip), dip = NaN; end
endd = dev(end);
idx = find(post & dev > 0.5, 1);
bad = any(~isfinite(XX(:, end)));
if ~isempty(idx) || bad
    oc = 'collapse';
    if isempty(idx), tdiv = tt(end); else, tdiv = tt(idx); end
    return
end
tdiv = NaN;
% growth check over the last quarter of the window
i75 = find(tt >= tt(end)*0.75, 1);
d75 = median(dev(i75:min(i75+20, numel(dev))));
den = median(dev(max(1, numel(dev)-20):end));
if den > 1e-3 && den > 2*max(d75, eps)
    oc = 'escaping';
else
    oc = 'ride-through';
end
end

function o = prep(dv, k)
o = run_scenario(struct('devtype', {dv}, 'kL', 1.0, 'k', k));
assert(isempty(o.fail), 'run_scenario failed: %s', o.fail);
end

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

function [tt, XX] = etdrk2(cs, dt, T, stopdiv)
n = numel(cs.x0);
[E, P1, P2] = phis(cs.A, dt);
nst = round(T/dt);
keep = max(1, floor(nst/4000));
tt = zeros(1, ceil(nst/keep) + 1);  XX = zeros(n, ceil(nst/keep) + 1);
x = cs.x0;  tt(1) = 0;  XX(:, 1) = x;  m = 1;
x0 = cs.x0;  A = cs.A;
for q = 1:nst
    t = (q-1)*dt;
    dx = x - x0;
    r1 = frhs(t, x, cs) - A*dx;
    a  = x0 + E*dx + dt*(P1*r1);
    r2 = frhs(t + dt, a, cs) - A*(a - x0);
    x  = a + dt*(P2*(r2 - r1));
    if mod(q, keep) == 0
        m = m + 1;  tt(m) = q*dt;  XX(:, m) = x;
        if stopdiv && (any(~isfinite(x)) || max(abs(x - x0)) > 50)
            tt = tt(1:m);  XX = XX(:, 1:m);  return
        end
    end
end
tt = tt(1:m);  XX = XX(:, 1:m);
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
tt = tt(1:m);  XX = XX(:, 1:m);
end

function [E, P1, P2] = phis(A, dt)
%PHIS  E = expm(A dt), phi1 and phi2 from one expm of an augmented block matrix
%   with a nilpotent part.
A = full(A);
n = size(A, 1);
W = expm([A*dt, eye(n)*dt, zeros(n); zeros(n), zeros(n), eye(n)*dt; ...
          zeros(n), zeros(n), zeros(n)]);
E  = full(W(1:n, 1:n));
P1 = full(W(1:n, n+1:2*n))/dt;
P2 = full(W(1:n, 2*n+1:3*n))/dt^2;
end

function Pk = series_phi(Z, k)
n = size(Z, 1);
Pk = zeros(n);  term = eye(n);
for j = 0:30
    Pk = Pk + term/factorial(j + k);
    term = term*Z;
end
end

function Vt = vterm(XX, o)
Vt = zeros(10, size(XX, 2));
for p = 1:10
    b = o.plant.bus(p);
    vd = XX(o.M.ix_bus(2*b-1), :);  vq = XX(o.M.ix_bus(2*b), :);
    Vt(p, :) = sqrt(vd.^2 + vq.^2);
end
end

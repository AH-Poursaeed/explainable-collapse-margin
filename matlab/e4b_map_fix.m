function e4b_map_fix
%E4B_MAP_FIX  Re-run the dG >= 1.2 cells of the security map with a later divergence stop.
%   The divergence stop is off during the forcing window, and the state cap is higher.
%
%   Why: the raised-cosine conductance pulse acts on a bus with a
%   parasitic capacitance (Bc = 0.005 pu). At dG >= 1.2 the local bus
%   voltage collapses hard DURING the 20 ms forcing. The integrator stop at
%   max|x - x0| > 50 then fires at t ~ 0.06 s and ends the run before the
%   behavior after the pulse is visible. Two cells get the wrong class this
%   way. The physical question is the recovery AFTER the pulse. So this
%   script stops a run only for t > TPOST and at a cap of 2000, and it
%   classifies the run after the pulse as e4_emt_production does.
%
%   Output: results/final/e4_security_map_fixed.csv. It holds the dG <= 0.9
%   rows of the original map (unchanged) and the re-run dG >= 1.2 rows, with
%   a 'rerun' marker column. The original CSV stays as it is, as the record
%   of the first run.

maxNumCompThreads(2);
P = params.sys_const();                                        %#ok<NASGU>
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'final');
dv = repmat({'GFL'}, 1, 10);  dv{9} = 'VSM';  dv{10} = 'VSM';
PULSE = struct('bus', 31, 'dG', 0.60, 't0', 0.05, 'dur', 0.020);
DT = 5e-5;
TPOST = PULSE.t0 + PULSE.dur + 0.05;
TMAP = 8.0;
CAP = 2000;

T0 = readtable(fullfile(outdir, 'e4_security_map.csv'));
keep = T0(T0.dG < 1.1, :);
keep.rerun = zeros(height(keep), 1);

kmap  = unique(T0.k)';
dGmap = [1.20 1.80];
rows = {};
for i = 1:numel(kmap)
    o = prep(dv, kmap(i));
    fprintf('-- k=%.3f (sigma=%+.4f)\n', kmap(i), o.S.sigma_dom);
    for j = 1:numel(dGmap)
        PU = PULSE;  PU.dG = dGmap(j);
        cs = mkcase(o, PU, PU.dG);
        tw = tic;
        [tt, XX] = etdrk2(cs, DT, TMAP, TPOST, CAP);
        wall = toc(tw);
        Vt = vterm(XX, o);
        [oc, tdiv, endd, dipj] = classify(tt, Vt, XX, TPOST);
        rows{end+1} = struct('k', kmap(i), 'dG', dGmap(j), ...
            'sigma_pred', o.S.sigma_dom, 'outcome', string(oc), ...
            'tdiv', tdiv, 'enddev', endd, 'pulse_dip', dipj, ...
            't_end', tt(end), 'wall_s', wall, 'rerun', 1);   %#ok<AGROW>
        fprintf('   dG=%.2f -> %-12s (tdiv %.2f, enddev %.4g, dip %.3f, wall %.0f s)\n', ...
            dGmap(j), oc, tdiv, endd, dipj, wall);
    end
end
Tr = struct2table([rows{:}]);
Tm = [keep; Tr(:, keep.Properties.VariableNames)];
Tm = sortrows(Tm, {'k', 'dG'});
writetable(Tm, fullfile(outdir, 'e4_security_map_fixed.csv'));
fprintf('saved e4_security_map_fixed.csv (%d cells)\ne4b done.\n', height(Tm));
end

% ==============================================================================
function [oc, tdiv, endd, dip] = classify(tt, Vt, XX, tpost)
dev = max(abs(Vt - Vt(:, 1)), [], 1);
post = tt >= tpost;
dip = max(dev(~post & tt > 0), [], 'omitnan');
if isempty(dip), dip = NaN; end
endd = dev(end);
idx = find(post & dev > 0.5, 1);
bad = any(~isfinite(XX(:, end))) || tt(end) < tpost;
if ~isempty(idx) || bad
    oc = 'collapse';
    if isempty(idx), tdiv = tt(end); else, tdiv = tt(idx); end
    return
end
tdiv = NaN;
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

function [tt, XX] = etdrk2(cs, dt, T, tpost, cap)
%ETDRK2  ETDRK2 time steps. The divergence stop acts only AFTER tpost, at the cap given.
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
    if any(~isfinite(x))
        m = m + 1;  tt(m) = q*dt;  XX(:, m) = x;
        tt = tt(1:m);  XX = XX(:, 1:m);  return
    end
    if mod(q, keep) == 0
        m = m + 1;  tt(m) = q*dt;  XX(:, m) = x;
        if t > tpost && max(abs(x - x0)) > cap
            tt = tt(1:m);  XX = XX(:, 1:m);  return
        end
    end
end
tt = tt(1:m);  XX = XX(:, 1:m);
end

function [E, P1, P2] = phis(A, dt)
A = full(A);
n = size(A, 1);
W = expm([A*dt, eye(n)*dt, zeros(n); zeros(n), zeros(n), eye(n)*dt; ...
          zeros(n), zeros(n), zeros(n)]);
E  = full(W(1:n, 1:n));
P1 = full(W(1:n, n+1:2*n))/dt;
P2 = full(W(1:n, 2*n+1:3*n))/dt^2;
end

function Vt = vterm(XX, o)
Vt = zeros(10, size(XX, 2));
for p = 1:10
    b = o.plant.bus(p);
    vd = XX(o.M.ix_bus(2*b-1), :);  vq = XX(o.M.ix_bus(2*b), :);
    Vt(p, :) = sqrt(vd.^2 + vq.^2);
end
end

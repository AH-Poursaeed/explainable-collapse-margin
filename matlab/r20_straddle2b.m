function r20_straddle2b
%r20_straddle2b  Two longer or deeper collapse runs for ray 11.
%   The point of ray 11 beyond the boundary grows at sigma = +0.209 (time
%   constant ~4.8 s). That is too slow for the 10 s window: the end
%   deviation is 8.7e-4 and still growing, below the 1e-3 divergence
%   floor. So this script adds two runs, with the same method and gates as
%   r20_straddle2:
%     collapse30  the same point (k = 2.044855), 30 s window
%     collapse2pc k = kstar + 2% (deeper, faster growth), 12 s window
%   It appends to results/tables/r20/r20_straddle2.csv.
maxNumCompThreads(2);
P = params.sys_const();                                        %#ok<NASGU>
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'tables', 'r20');
fout = fullfile(outdir, 'r20_straddle2.csv');
assert(isfile(fout), 'base straddle CSV missing');

S = load(fullfile(here, '..', 'results', 'tables', 'r17', 'r17_rays_spec.mat'));
R = readtable(fullfile(here, '..', 'results', 'tables', 'r17', 'r17_rays.csv'));
ray = 11;
PULSE = struct('bus', 31, 'dG', 0.60, 't0', 0.05, 'dur', 0.020);
DT = 5e-5;
TPOST = PULSE.t0 + PULSE.dur + 0.05;

T0 = readtable(fout);
done = {};
for r = 1:height(T0), done{end+1} = sprintf('%d_%s', T0.rayid(r), T0.case_{r}); end %#ok<AGROW>

ci = S.cfgid_r(ray);
cfg = mkcfg(S.cfgs{ci}, S.outid_r(ray), S.kL_r(ray), S.kdis_r(ray,:), ...
            S.fplc_r(ray), S.fplj_r(ray,:), S.Hv_r(ray,:), S.Dv_r(ray,:), S.kbw_r(ray));
row = R(R.rayid == ray, :);
kstar = row.kstar;

cases = struct('name', {'collapse30', 'collapse2pc'}, ...
               'k', {kstar + 0.01*kstar, kstar + 0.02*kstar}, ...
               'T', {30.0, 12.0});
for c = 1:2
    key = sprintf('%d_%s', ray, cases(c).name);
    if any(strcmp(done, key)), fprintf('%s checkpointed, SKIP\n', key); continue; end
    cfg.k = cases(c).k;
    o = run_scenario(cfg);
    assert(isempty(o.fail), 'run_scenario failed: %s', o.fail);
    sig = o.S.sigma_dom;
    fprintf('ray %d %s: k=%.4f sigma_dom=%+.4f\n', ray, cases(c).name, cases(c).k, sig);
    assert(sig > 0, 'point not unstable');
    cs = mkcase(o, PULSE, PULSE.dG);
    tw = tic;
    [tt, XX] = etdrk2(cs, DT, cases(c).T, true);
    wall = toc(tw);
    Vt = vterm(XX, o);
    [oc, tdiv, endd, dipj] = classify(tt, Vt, XX, TPOST);
    fid = fopen(fout, 'a');
    fprintf(fid, '%d,%s,%.6f,%.6f,%s,%.3f,%.6g,%.4f,%.3f,%.1f\n', ...
        ray, cases(c).name, cases(c).k, sig, oc, tdiv, endd, dipj, tt(end), wall);
    fclose(fid);
    fprintf('ray %d %s -> %s (tdiv %.2f, enddev %.4g, wall %.0f s)\n', ...
        ray, cases(c).name, oc, tdiv, endd, wall);
end
fprintf('r20_straddle2b complete.\n');
end

% ---- mkcfg: copied unchanged from r17_rays.m --------------------------------
function cfg = mkcfg(gfmb, outid, kL, kdis, fplc, fplj, Hvs, Dvs, kbw)
devtype = repmat({'GFL'}, 1, 10);
buses = 30:39;
gslot = 0;
tune = repmat(struct('f_pll', [], 'Hv', [], 'Dv', [], 'kbw', kbw), 10, 1);
for b = 1:10
    if ismember(buses(b), gfmb)
        devtype{b} = 'VSM';
        gslot = gslot + 1;
        tune(b).Hv = Hvs(min(gslot, 4));
        tune(b).Dv = Dvs(min(gslot, 4));
    else
        tune(b).f_pll = min(max(fplc*fplj(b), 6), 45);
    end
end
cfg = struct('devtype', {devtype}, 'kL', kL, 'kdisp', kdis, 'tune', tune);
if outid > 0, cfg.outage = outid; end
end

% ==== local functions copied unchanged from e4_emt_production.m ====
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

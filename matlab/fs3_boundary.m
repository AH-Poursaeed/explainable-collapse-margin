function fs3_boundary
%FS3_BOUNDARY  Feasibility test of the full-model collapse boundary.
%   It tests how well the Section-III method predicts the full-model
%   collapse boundary. For each stress ray: (i) full-model k*
%   (sigma_dom = 0 crossing) by scan + bisection; (ii) the lossy multiport
%   pencil's predicted k*; (iii) the index values (PWSGSI, lam_min, gSCR)
%   just on the stable side of the boundary. Rays: k-sweep at kL in
%   {0.8, 1.0, 1.2} and a load-growth ray (kL-sweep at k = 1).
%   Output: results/feasibility/fs3_boundary.csv (+ per-ray scan log).

P = params.sys_const();
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'feasibility');
if ~exist(outdir, 'dir'), mkdir(outdir); end

dv = repmat({'GFL'}, 1, 10);  dv{9} = 'VSM';  dv{10} = 'VSM';   % buses 38, 39

rows = {};
scanlog = {};

% ---- k-sweep rays ----------------------------------------------------------
for kL = [0.8 1.0 1.2]
    cfgf = @(k) struct('devtype', {dv}, 'kL', kL, 'k', k);
    [ks, sf, sp, log1] = boundary_pair(cfgf, 'k', 1.0, 0.1, 3.0, P);
    scanlog = [scanlog(:); log1(:)];                           %#ok<AGROW>
    r = pack_ray(sprintf('k-sweep kL=%.1f', kL), 'k', ks, sf, sp, cfgf, P);
    rows{end+1} = r;                                           %#ok<AGROW>
    fprintf('[ray kL=%.1f] %s\n', kL, jsonencode(r));
end

% ---- load-growth ray at k = 1 ----------------------------------------------
cfgf = @(kL) struct('devtype', {dv}, 'kL', kL, 'k', 1.0);
[ks, sf, sp, log2] = boundary_pair(cfgf, 'kL', 1.0, 0.1, 3.0, P);
scanlog = [scanlog(:); log2(:)];
r = pack_ray('load-growth k=1.0', 'kL', ks, sf, sp, cfgf, P);
rows{end+1} = r;
fprintf('[ray load-growth] %s\n', jsonencode(r));

T = struct2table([rows{:}]);
writetable(T, fullfile(outdir, 'fs3_boundary.csv'));
Ts = struct2table([scanlog{:}]);
writetable(Ts, fullfile(outdir, 'fs3_scanlog.csv'));
fprintf('fs3 done: %d rays\n', height(T));
end

% -----------------------------------------------------------------------------
function [kstar, sig_full_fun, sig_pen_fun, log] = boundary_pair(cfgf, axname, k0, dk, kmax, P)
%BOUNDARY_PAIR scan the ray upward from k0 until full-model instability or
%   PF failure, then bisect the full-model crossing to 1e-3.
log = {};
klo = k0;  khi = NaN;  k = k0;
sig_full_fun = @(k) full_sigma(cfgf(k), P);
sig_pen_fun  = @(k) pencil_sigma(cfgf(k), P);
while k <= kmax
    [s, ok] = sig_full_fun(k);
    log{end+1} = struct('axis', axname, 'k', k, 'sigma_full', s, 'pf_ok', ok); %#ok<AGROW>
    fprintf('  scan %s=%.3f  sigma=%.4f  ok=%d\n', axname, k, s, ok);
    if ~ok
        kstar = struct('kstar', NaN, 'mode', 'pf_infeasible_first', ...
                       'k_last_ok', klo);
        return
    end
    if s >= 0, khi = k; break; end
    klo = k;  k = k + dk;
end
if isnan(khi)
    kstar = struct('kstar', NaN, 'mode', 'no_crossing_in_range', 'k_last_ok', klo);
    return
end
while khi - klo > 1e-3
    km = 0.5*(klo + khi);
    [s, ok] = sig_full_fun(km);
    if ~ok
        % PF fails inside the bracket: report the smallest infeasible point
        khi = km;
        kstar = struct('kstar', NaN, 'mode', 'pf_infeasible_in_bracket', ...
                       'k_last_ok', klo);
        return
    end
    if s >= 0, khi = km; else, klo = km; end
end
kstar = struct('kstar', 0.5*(klo + khi), 'mode', 'crossing', 'k_last_ok', klo);
end

function r = pack_ray(name, axname, ks, sig_full_fun, sig_pen_fun, cfgf, P)
r = struct('ray', name, 'axis', axname, 'mode', ks.mode, 'kstar_full', ks.kstar);
r.kstar_pencil = NaN;  r.pencil_err_pct = NaN;
r.PWSGSI_star = NaN;  r.lam_min_star = NaN;  r.gSCR_star = NaN;
r.sigma_at_star_minus = NaN;
if ~strcmp(ks.mode, 'crossing'), return; end

% pencil crossing by bisection seeded around the full k*
[slo, ok] = sig_pen_fun(max(ks.k_last_ok - 0.4, 0.2));
klo = max(ks.k_last_ok - 0.4, 0.2);
khi = klo;  found = false;
for k = klo:0.05:(ks.kstar + 0.6)
    [s, ok] = sig_pen_fun(k);
    if ~ok, break; end
    if s >= 0, khi = k; found = true; break; end
    klo = k;
end
if found
    while khi - klo > 1e-3
        km = 0.5*(klo + khi);
        [s, ok] = sig_pen_fun(km);
        if ~ok, break; end
        if s >= 0, khi = km; else, klo = km; end
    end
    r.kstar_pencil = 0.5*(klo + khi);
    r.pencil_err_pct = 100*abs(r.kstar_pencil - r.kstar_full)/r.kstar_full;
end

% index values just on the stable side
try
    kprobe = ks.kstar - 0.005;
    cfg = cfgf(kprobe);
    o = run_scenario(cfg);
    if isempty(o.fail)
        R = reduced.chain(o.mpc, o.res, o.plant, P);
        r.PWSGSI_star = R.PWSGSI;  r.lam_min_star = R.lam_min;  r.gSCR_star = R.gSCR;
        r.sigma_at_star_minus = o.S.sigma_dom;
    end
catch
end
end

function [s, ok] = full_sigma(cfg, P) %#ok<INUSD>
% MATPOWER can throw (rather than flag) on reference-bus loss during
% Q-limit enforcement near voltage collapse; treat as PF failure.
try
    o = run_scenario(cfg);
    ok = isempty(o.fail);
    if ok, s = o.S.sigma_dom; else, s = NaN; end
catch
    ok = false;  s = NaN;
end
end

function [s, ok] = pencil_sigma(cfg, P)
try
    o = run_scenario(cfg, struct('linearize', false));
    ok = isempty(o.fail) || strcmp(o.fail, '');
    if ~ok, s = NaN; return; end
    R = reduced.chain(o.mpc, o.res, o.plant, P);
    PL = reduced.pencil_lossy(R, P, o.mpc, o.res, o.plant);
    s = PL.sigma_dom;
catch
    s = NaN;  ok = false;
end
end

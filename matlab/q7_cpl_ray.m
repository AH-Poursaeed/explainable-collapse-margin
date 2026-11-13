function q7_cpl_ray
%Q7_CPL_RAY  Effect of the load model on the collapse boundary (CPL ray).
%   The published model realizes loads as series-RL legs. In a 100%
%   power-electronic system, most loads are converter-interfaced
%   (constant-power-like). This script sweeps the design ray with a
%   fraction alpha of every load re-typed as a smoothed CPL (tau = 20 ms).
%   The power flow and the equilibrium are IDENTICAL for every alpha, so
%   any movement of k* comes only from the dynamic load model.
%
%   For alpha in {0, 0.3, 0.6, 1.0}: scan k upward from 1.5 (0.05 grid) and
%   bisect the sigma_dom = 0 crossing to 1e-3. Record k*, sigma at the
%   probe points, and the reduced-chain values (GRIP, lam_min, gSCR) at the
%   last stable scan point. By its DECLARED scope, the chain absorbs the
%   loads as constant-Z, so it does not see alpha; this gap is what the
%   script measures.
%   Also: tau sensitivity at alpha = 0.6 (tau = 100 ms), and the
%   fixed-point sensitivity sigma_dom(k = 1.90) vs alpha.
%
%   The alpha = 0 run is the published chain (sigma_dom(k = 1.0) = -4.0312,
%   k* = 1.9730 +/- 2e-3). The script prints the alpha = 0 results first; it
%   does not compare them with these values.
%
%   Output: results/final/tables/q7_cpl_ray.csv (+ q7_cpl_scanlog.csv)
%

maxNumCompThreads(2);
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'final', 'tables');
fout = fullfile(outdir, 'q7_cpl_ray.csv');
flog = fullfile(outdir, 'q7_cpl_scanlog.csv');
P = params.sys_const();
dv = repmat({'GFL'}, 1, 10);  dv{9} = 'VSM';  dv{10} = 'VSM';

runs = {  % alpha, tau
    0.0, NaN;
    0.3, 0.02;
    0.6, 0.02;
    1.0, 0.02;
    0.6, 0.10};

done = {};
if isfile(fout)
    T0 = readtable(fout);
    done = arrayfun(@(a, t) sprintf('%.2f_%.3f', a, t), T0.alpha, T0.tau, ...
        'UniformOutput', false);
    fid = fopen(fout, 'a');
    flg = fopen(flog, 'a');
else
    fid = fopen(fout, 'w');
    fprintf(fid, ['alpha,tau,kstar,k_last_stable,sigma_at_190,', ...
        'GRIP_last,lam_min_last,gSCR_last,resid_worst\n']);
    flg = fopen(flog, 'w');
    fprintf(flg, 'alpha,tau,k,sigma_dom,resid_pu\n');
end

for rr = 1:size(runs, 1)
    alpha = runs{rr, 1};  tau = runs{rr, 2};
    key = sprintf('%.2f_%.3f', alpha, tau);
    if any(strcmp(done, key)), fprintf('skip %s\n', key); continue; end
    fprintf('=== alpha = %.2f, tau = %g ===\n', alpha, tau);
    mkcfg = @(k) local_cfg(dv, k, alpha, tau);
    % fixed-point sensitivity at k = 1.90
    o = run_scenario(mkcfg(1.90));
    assert(isempty(o.fail), o.fail);
    sig190 = o.S.sigma_dom;
    resid_worst = o.residual_pu;
    fprintf('  sigma_dom(k=1.90) = %+.4f (resid %.2e)\n', sig190, o.residual_pu);
    % scan + bisect. If already unstable at the scan origin, walk DOWN to
    % find the stable side first (the all-CPL stiff limit can be unstable
    % well below the constant-Z fold).
    korig = 1.50;
    o = run_scenario(mkcfg(korig));
    assert(isempty(o.fail), o.fail);
    fprintf(flg, '%.2f,%.3f,%.4f,%.8g,%.3g\n', alpha, tau, korig, ...
        o.S.sigma_dom, o.residual_pu);
    fprintf('  scan k=%.3f sigma=%+.4f\n', korig, o.S.sigma_dom);
    klo = NaN;  khi = NaN;  slo = NaN;
    if o.S.sigma_dom < 0
        klo = korig;  slo = o.S.sigma_dom;  k = korig + 0.05;  up = true;
    else
        khi = korig;  k = korig - 0.05;  up = false;
    end
    while (up && k <= 2.6) || (~up && k >= 0.30)
        o = run_scenario(mkcfg(k));
        if ~isempty(o.fail)
            fprintf('  scan k=%.3f FAIL %s\n', k, o.fail);
            break
        end
        resid_worst = max(resid_worst, o.residual_pu);
        fprintf(flg, '%.2f,%.3f,%.4f,%.8g,%.3g\n', alpha, tau, k, ...
            o.S.sigma_dom, o.residual_pu);
        fprintf('  scan k=%.3f sigma=%+.4f\n', k, o.S.sigma_dom);
        if up
            if o.S.sigma_dom >= 0, khi = k; break; end
            klo = k;  slo = o.S.sigma_dom;  k = k + 0.05;
        else
            if o.S.sigma_dom < 0, klo = k; slo = o.S.sigma_dom; break; end
            khi = k;  k = k - 0.05;
        end
    end
    if ~(isfinite(khi) && isfinite(klo))
        % a valid result, not an error: no bracketed crossing in the scan range
        fprintf(fid, '%.2f,%.3f,NaN,NaN,%.8g,NaN,NaN,NaN,%.3g\n', ...
            alpha, tau, sig190, resid_worst);
        fprintf('  -> NO bracketed crossing in k range (all-%s); row recorded\n', ...
            ternary(isfinite(khi), 'unstable', 'stable'));
        continue
    end
    for it = 1:12
        km = (klo + khi)/2;
        if khi - klo < 1e-3, break; end
        o = run_scenario(mkcfg(km));
        if isempty(o.fail) && o.S.sigma_dom < 0
            klo = km;  slo = o.S.sigma_dom;
        else
            khi = km;
        end
        fprintf(flg, '%.2f,%.3f,%.4f,%.8g,%.3g\n', alpha, tau, km, ...
            o.S.sigma_dom, o.residual_pu);
    end
    kstar = (klo + khi)/2;
    % reduced-chain values at the last stable point (declared constant-Z scope)
    ol = run_scenario(mkcfg(klo));
    Rc = reduced.chain(ol.mpc, ol.res, ol.plant, P);
    fprintf(fid, '%.2f,%.3f,%.5f,%.5f,%.8g,%.8g,%.8g,%.8g,%.3g\n', ...
        alpha, tau, kstar, klo, sig190, Rc.PWSGSI, Rc.lam_min, Rc.gSCR, resid_worst);
    fprintf('  -> k* = %.4f (last stable %.4f, GRIP %.4f)\n', kstar, klo, Rc.PWSGSI);
end
fclose(fid);  fclose(flg);
fprintf('q7_cpl_ray complete.\n');
end

function cfg = local_cfg(dv, k, alpha, tau)
cfg = struct('devtype', {dv}, 'kL', 1.0, 'k', k);
if alpha > 0
    cfg.cpl = struct('alpha', alpha, 'tau', tau);
end
end

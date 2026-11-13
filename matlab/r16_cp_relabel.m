function r16_cp_relabel
%r16_cp_relabel  Re-label the FULL converged bank with a constant-power load.
%   The load model is the 60% smoothed constant-power load (alpha = 0.6,
%   tau = 20 ms), set as in local_cfg of q7_cpl_ray. Power flow and
%   equilibrium are IDENTICAL for every alpha, so the stored features do
%   not move. Only the dynamic label (sigma_dom / zeta_min / class) can
%   change.
%
%   GATE (runs first, and the script stops if it fails): a 50-row probe at
%   alpha = 0 must reproduce the stored sigma_dom to rel 1e-6 (denominator
%   floor 1e-3, as in audit_rows). The gate writes its record to
%   r16_cp_gate.txt. The full run starts only when the gate file exists.
%
%   Row-checkpointed: each row is appended to the file, a row whose id is
%   already there is skipped, and a stopped run can resume.
%   Outputs (results/tables/r16/):
%     r16_cp_relabel.csv  id, stored and CP labels per row, flip flags
%     r16_cp_metrics.csv  AUC(low metric -> unstable) for gSCR/SCR_min/
%                         lam_min/GRIP against the CP labels AND against the
%                         stored labels on the same rows (stored
%                         trust_dom == 1, fresh run ok), as in e5_inversion
%     r16_cp_summary.csv  n_ok/n_fail/flip counts/margin-shift quantiles
%

maxNumCompThreads(2);
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'tables', 'r16');
if ~exist(outdir, 'dir'), mkdir(outdir); end
fout = fullfile(outdir, 'r16_cp_relabel.csv');
fmet = fullfile(outdir, 'r16_cp_metrics.csv');
fsum = fullfile(outdir, 'r16_cp_summary.csv');
fgate = fullfile(outdir, 'r16_cp_gate.txt');

ALPHA = 0.6;  TAU = 0.02;
EDGES = [0 0.01 0.03 0.05 0.10];

T = readtable(fullfile(here, '..', 'data', 'dataset.csv'));
assert(height(T) == 7324, 'dataset drift: %d rows (expected 7324)', height(T));
S = table2struct(T);                       % never row-slice a table in a hot loop
N = numel(S);

% ---- gate: 50-row alpha = 0 probe --------------------------------------------
if ~isfile(fgate)
    fprintf('GATE: 50-row alpha = 0 sigma_dom reproduction probe...\n');
    rng(20260807, 'twister');
    pick = randperm(N, 50);
    worst = 0;  worst_id = NaN;
    for q = 1:50
        r = S(pick(q));
        o = run_scenario(bank.row2cfg(r));
        assert(isempty(o.fail), 'probe id %d fresh-run failed: %s', r.id, o.fail);
        sig = o.S.sigma_dom;
        if ~o.S.oob_stable, sig = max(sig, o.S.oob_maxRe); end
        d = abs(sig - r.sigma_dom) / max(1e-3, abs(sig));
        if d > worst, worst = d; worst_id = r.id; end
        if mod(q, 10) == 0, fprintf('  probe %d/50 (worst rel %.3g)\n', q, worst); end
    end
    assert(worst <= 1e-6, ...
        'GATE FAIL: worst rel sigma_dom mismatch %.3g at id %d (tol 1e-6)', ...
        worst, worst_id);
    fid = fopen(fgate, 'w');
    fprintf(fid, 'r16_cp_relabel gate PASS %s\nworst rel sigma_dom mismatch %.3g (id %d), 50 rows, tol 1e-6, floor 1e-3\n', ...
        char(datetime('now')), worst, worst_id);
    fclose(fid);
    fprintf('GATE PASS: worst rel mismatch %.3g (id %d)\n', worst, worst_id);
else
    fprintf('gate file exists, skipping probe\n');
end

% ---- full CP re-label sweep (row-checkpointed) -------------------------------
done = containers.Map('KeyType', 'double', 'ValueType', 'logical');
if isfile(fout)
    T0 = readtable(fout);
    for i = 1:height(T0), done(T0.id(i)) = true; end
    fprintf('resume: %d rows already done\n', done.Count);
else
    fid = fopen(fout, 'w');
    fprintf(fid, ['id,sigma_cz,zeta_cz,class_cz,sigma_cp,zeta_cp,fdom_cp,', ...
        'class_cp,oob_cp,trust_cp,resid_pu,flip_bin,dclass,fail\n']);
    fclose(fid);
end

t0 = tic;  n0 = done.Count;
for i = 1:N
    r = S(i);
    if isKey(done, r.id), continue; end
    cfg = bank.row2cfg(r);
    cfg.cpl = struct('alpha', ALPHA, 'tau', TAU);
    sig = NaN; zeta = NaN; fdom = NaN; cls = NaN; oobf = NaN; trust = NaN;
    resid = NaN; flip = NaN; dcls = NaN; failmsg = '';
    try
        o = run_scenario(cfg);
        if ~isempty(o.fail)
            failmsg = o.fail;
        elseif o.S.nzero ~= 1
            failmsg = 'zero_mode_count';
        else
            zeta = o.S.zeta_min;  sig = o.S.sigma_dom;  fdom = o.S.f_dom;
            trust = o.S.trust_dom;  resid = o.residual_pu;
            if ~o.S.oob_stable
                zeta = min(zeta, -1e-6);
                sig = max(sig, o.S.oob_maxRe);
                oobf = 1;
            else
                oobf = 0;
            end
            cls = sum(zeta >= EDGES);  if zeta <= 0, cls = 0; end
            flip = double((sig >= 0) ~= (r.sigma_dom >= 0));
            dcls = cls - r.class;
        end
    catch e
        failmsg = strrep(['error: ' e.message], ',', ';');
    end
    fid = fopen(fout, 'a');
    fprintf(fid, '%d,%.10g,%.10g,%d,%.10g,%.10g,%.10g,%g,%g,%g,%.3g,%g,%g,%s\n', ...
        r.id, r.sigma_dom, r.zeta_min, r.class, sig, zeta, fdom, cls, oobf, ...
        trust, resid, flip, dcls, failmsg);
    fclose(fid);
    done(r.id) = true;
    if mod(done.Count, 25) == 0
        el = toc(t0);  rate = (done.Count - n0) / el;
        fprintf('%d/%d rows (%.2f rows/s, ETA %.0f min)\n', done.Count, N, ...
            rate, (N - done.Count) / rate / 60);
    end
end
fprintf('sweep complete: %d rows\n', done.Count);

% ---- metrics + summary (only from the complete relabel file) -----------------
R = readtable(fout);
assert(height(R) == N, 'incomplete relabel file: %d of %d', height(R), N);
[~, loc] = ismember(R.id, T.id);
assert(all(loc > 0), 'id join failure');
gSCR = T.gSCR(loc);  SCRm = T.SCR_min(loc);  lam = T.lam_min(loc);
GRIP = T.PWSGSI(loc);  trust_cz = T.trust_dom(loc);

% Failed rows carry NaN sigma_cp by design. A test on the fail string
% would break when that column is ALL empty: readtable then gives NaN
% doubles, and string(NaN) never equals ''. That is the case n = 0.
okrow = isfinite(R.sigma_cp);
use = okrow & trust_cz == 1;               % same population as in e5_inversion
y_cp = R.sigma_cp(use) >= 0;
y_cz = R.sigma_cz(use) >= 0;
mets = {'gSCR', gSCR; 'SCR_min', SCRm; 'lam_min', lam; 'PWSGSI', GRIP};
fid = fopen(fmet, 'w');
fprintf(fid, 'metric,n,n_unstable_cp,AUC_cp,n_unstable_cz,AUC_cz\n');
for m = 1:size(mets, 1)
    x = mets{m, 2}(use);
    [~, ~, ~, a_cp] = perfcurve(y_cp, -x, true);
    [~, ~, ~, a_cz] = perfcurve(y_cz, -x, true);
    fprintf(fid, '%s,%d,%d,%.6f,%d,%.6f\n', mets{m, 1}, sum(use), ...
        sum(y_cp), a_cp, sum(y_cz), a_cz);
    fprintf('[%s] AUC(low->unstable) CP %.4f | CZ %.4f (n=%d)\n', ...
        mets{m, 1}, a_cp, a_cz, sum(use));
end
fclose(fid);

dm = -(R.sigma_cp(okrow) - R.sigma_cz(okrow));   % margin m = -sigma shift
fid = fopen(fsum, 'w');
fprintf(fid, ['n_rows,n_ok,n_fail,n_flip_bin,pct_flip_bin,n_class_change,', ...
    'median_dm,q05_dm,q95_dm,mean_dm,max_abs_dm,n_trust_used\n']);
fprintf(fid, '%d,%d,%d,%d,%.4f,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%d\n', ...
    N, sum(okrow), N - sum(okrow), sum(R.flip_bin(okrow) == 1), ...
    100 * sum(R.flip_bin(okrow) == 1) / N, sum(R.dclass(okrow) ~= 0), ...
    median(dm), quantile(dm, 0.05), quantile(dm, 0.95), mean(dm), ...
    max(abs(dm)), sum(use));
fclose(fid);
fprintf('r16_cp_relabel COMPLETE: %d ok, %d fail, %d binary flips (%.2f%%), median dm %.4g\n', ...
    sum(okrow), N - sum(okrow), sum(R.flip_bin(okrow) == 1), ...
    100 * sum(R.flip_bin(okrow) == 1) / N, median(dm));
end

function audit_rows(nrows)
%AUDIT_ROWS  Cold re-audit of random dataset rows.
%   Recomputes each row END-TO-END from scratch (power flow -> full model ->
%   reduced chain -> label) and compares every stored quantity with the
%   fresh value. Any mismatch beyond tolerance is a FAIL and is printed.

if nargin < 1, nrows = 10; end
P = params.sys_const();
here = fileparts(mfilename('fullpath'));
T = readtable(fullfile(here, '..', 'data', 'dataset.csv'));
rng(1234, 'twister');                       % audit seed, distinct from build
pick = randperm(height(T), nrows);

tolrel = 1e-6;
nbad = 0;
for q = 1:nrows
    r = T(pick(q), :);
    cfg = bank.row2cfg(r);
    o = run_scenario(cfg);
    if ~isempty(o.fail)
        fprintf('[FAIL] id %d: fresh run failed (%s) but stored ok\n', r.id, o.fail);
        nbad = nbad + 1; continue;
    end
    % the SAMPLED tuning reaches the analytic chain, so the
    % threshold columns can be checked as well
    R = reduced.chain(o.mpc, o.res, o.plant, P, cfg.tune);
    zeta = o.S.zeta_min;
    sig  = o.S.sigma_dom;
    if ~o.S.oob_stable
        % mirror the bank's out-of-band forcing on BOTH stored quantities
        % (the stored sigma carries the same adjustment)
        zeta = min(zeta, -1e-6);
        sig = max(sig, o.S.oob_maxRe);
    end
    edges = [0 0.01 0.03 0.05 0.10];
    cls = sum(zeta >= edges); if zeta <= 0, cls = 0; end
    % denominator floors: 1e-3 for sigma (near-marginal rows have sigma ~ 0
    % and a 1e-9 floor would amplify eig noise into false failures)
    checks = {
        'zeta_min', r.zeta_min, zeta, 1e-8, 1e-9;
        'sigma_dom', r.sigma_dom, sig, 1e-6, 1e-3;
        'lam_min', r.lam_min, R.lam_min, tolrel, 1e-9;
        'PWSGSI', r.PWSGSI, R.PWSGSI, tolrel, 1e-9;
        'Hbar', r.Hbar, R.Hbar, tolrel, 1e-9;
        'C_SGSI', r.C_SGSI, R.C_SGSI, 1e-4, 1e-9;
        'C_dc', r.C_dc, R.C_dc, 1e-4, 1e-9;
        'Yeff_max', max(r{1, startsWith(T.Properties.VariableNames, 'Yeff_')}), max(R.Yeff), tolrel, 1e-9;
        'class', r.class, cls, 0.4, 1;
    };
    rowbad = false;
    for c = 1:size(checks, 1)
        a = checks{c,2}; b = checks{c,3};
        if isnan(a) && isnan(b), continue; end        % both-NaN (e.g. C fields)
        d = abs(a - b)/max(checks{c,5}, abs(b));
        if strcmp(checks{c,1}, 'class'), d = abs(a - b); end
        if d > checks{c,4} || (isnan(a) ~= isnan(b))
            fprintf('[FAIL] id %d %s: stored %.6g vs fresh %.6g\n', ...
                r.id, checks{c,1}, a, b);
            rowbad = true;
        end
    end
    if ~strcmp(string(r.thr_rule), R.thr_rule)
        fprintf('[FAIL] id %d thr_rule: stored %s vs fresh %s\n', r.id, string(r.thr_rule), R.thr_rule);
        rowbad = true;
    end
    if ~rowbad
        fprintf('[PASS] id %d (class %d, zeta %.4f, lam_min %.4f, rule %s)\n', ...
            r.id, cls, zeta, R.lam_min, R.thr_rule);
    else
        nbad = nbad + 1;
    end
end
fprintf('cold row audit: %d/%d rows verified end-to-end\n', nrows-nbad, nrows);
assert(nbad == 0, 'AUDIT FAILURES: %d rows', nbad);
end

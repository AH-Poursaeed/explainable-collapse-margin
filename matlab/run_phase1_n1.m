function N1 = run_phase1_n1()
%RUN_PHASE1_N1  N-1 screening of the selected design (GFM at buses 38, 39).
%   Non-islanding single-branch outages at nominal load; gate: zeta_min > 3%
%   for every contingency; also records the non-islanding pool for the
%   scenario bank / OOD holdout.

P = params.sys_const();
base = struct('devtype', {[repmat({'GFL'},1,8), {'VSM','VSM'}]});
[mpc0, ~] = build_case(base, P);
define_constants;
nb = size(mpc0.bus, 1);
nbr = size(mpc0.branch, 1);

% non-islanding pool by connectivity test
pool = [];
for e = 1:nbr
    st = true(nbr, 1); st(e) = false;
    A = sparse([mpc0.branch(st,F_BUS); mpc0.branch(st,T_BUS)], ...
               [mpc0.branch(st,T_BUS); mpc0.branch(st,F_BUS)], 1, nb, nb);
    g = graph(A);
    if max(conncomp(g)) == 1
        pool(end+1) = e;                                %#ok<AGROW>
    end
end
N1.pool = pool;
fprintf('non-islanding N-1 pool: %d of %d branches\n', numel(pool), nbr);

N1.res = struct([]);
worst = inf;
for i = 1:numel(pool)
    cfg = base;  cfg.outage = pool(i);
    o = run_scenario(cfg);
    r.branch = pool(i);
    r.fbus = mpc0.branch(pool(i), F_BUS);
    r.tbus = mpc0.branch(pool(i), T_BUS);
    if isempty(o.fail)
        r.zeta_min = o.S.zeta_min;
        r.sigma_dom = o.S.sigma_dom;
        r.oob = o.S.oob_stable;
    else
        r.zeta_min = NaN; r.sigma_dom = NaN; r.oob = NaN; r.fail = o.fail;
    end
    if isempty(N1.res), N1.res = r; else, N1.res(end+1) = r; end
    worst = min(worst, r.zeta_min);
end
N1.worst_zeta = worst;
N1.pass = worst > 0.03;
fprintf('worst N-1 zeta_min = %.4f  -> gate (>3%%): %s\n', worst, string(N1.pass));
zz = [N1.res.zeta_min];
[~, iw] = min(zz);
fprintf('worst contingency: branch %d (%d-%d)\n', N1.res(iw).branch, N1.res(iw).fbus, N1.res(iw).tbus);
outdir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
save(fullfile(outdir, 'phase1_n1.mat'), '-struct', 'N1');
end

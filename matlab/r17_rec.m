function r17_rec
%r17_rec  Census of the physics-side recourse search on unstable held-out test rows.
%   The script runs the deployed physics-side recourse search of
%   q9_recourse_physics.m as a CENSUS. The rows are a stratified sample of
%   verified unsafe HELD-OUT TEST rows (class = 0, that is, truly unstable),
%   not the 12 marginal candidates. It reports the stabilization and
%   alarm-clearance rates, and the distributions of MW, evaluations and
%   wall time.
%
%   Search method: the same as in q9. It is a projected FD gradient descent
%   of sigma_dom over the same dispatch box, with +-0.02 pu central probes,
%   backtracking steps, up to 4 gradient passes and no radius cap. The
%   slack is pinned, with Sn-weighted MW conservation on the non-slack dP.
%   The truth uses the bank convention: out-of-band unstable modes force
%   sigma through oob_maxRe.
%
%   Strata by depth on the unstable side: sigma_dom in [0,0.5), [0.5,2),
%   [2,5) and [5,inf). The rows are taken one from each stratum in turn, so
%   a run that stops at the time limit still covers every stratum.
%   Deterministic under P.seed + 41.
%
%   Output: results/tables/r17/r17_rec.csv

maxNumCompThreads(2);
N_CENSUS = 108;
TBOX = 165*60;
t00 = tic;

P = params.sys_const();
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..');
outdir = fullfile(root, 'results', 'tables', 'r17');
if ~exist(outdir, 'dir'), mkdir(outdir); end
fout = fullfile(outdir, 'r17_rec.csv');

Tb = readtable(fullfile(root, 'data', 'dataset.csv'));
Tt = readtable(fullfile(root, 'results', 'ml_export', 'test_frame.csv'));
Tu = Tb(ismember(Tb.id, Tt.id) & Tb.class == 0, :);
fprintf('census base: %d unstable held-out test rows\n', height(Tu));

% ---- stratified order, one row from each stratum in turn (deterministic) ---
edges = [0, 0.5, 2, 5, inf];
rng(P.seed + 41, 'twister');
per = ceil(N_CENSUS/(numel(edges)-1));
strat = cell(numel(edges)-1, 1);
for b = 1:numel(edges)-1
    idb = find(Tu.sigma_dom >= edges(b) & Tu.sigma_dom < edges(b+1));
    idb = idb(randperm(numel(idb)));
    strat{b} = idb(1:min(per, numel(idb)));
    fprintf('stratum %d [%g,%g): %d rows, taking %d\n', b, edges(b), ...
        edges(b+1), numel(idb), numel(strat{b}));
end
order = [];  stratlab = [];
for j = 1:per
    for b = 1:numel(strat)
        if j <= numel(strat{b})
            order(end+1) = strat{b}(j);                       %#ok<AGROW>
            stratlab(end+1) = b;                              %#ok<AGROW>
        end
    end
end

ALARM = -2.19073;             % deployed out-of-fold (OOF) alarm on the margin, as in q9
DFD = 0.02;
define_constants;
mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1, ...
    'pf.tol', 1e-10, 'pf.nr.max_it', 40);

done = [];
if isfile(fout)
    T0 = readtable(fout);
    done = T0.id;
    fid = fopen(fout, 'a');
else
    fid = fopen(fout, 'w');
    fprintf(fid, ['id,stratum,sigma0_row,sigma0_chk,sigma_best,cost_pu,', ...
        'cost_mw,stabilised,clears_alarm,n_evals,wall_s,note\n']);
end

smoke = ~isempty(getenv('SMOKE_RUN'));
ncase = numel(order);  if smoke, ncase = 1; end

for q = 1:ncase
    row = Tu(order(q), :);
    id = row.id;
    if any(done == id), fprintf('skip id=%d (done)\n', id); continue; end
    if toc(t00) > TBOX
        fprintf('Time limit reached at case %d/%d (%.0f min) -- exiting cleanly\n', ...
            q, ncase, toc(t00)/60);
        break
    end
    tC = tic;
    fprintf('=== case %d/%d id=%d sigma0=%+.4f stratum %d ===\n', q, ncase, ...
        id, row.sigma_dom, stratlab(q));
    try
        cfg = bank.row2cfg(row);
        [mpc0, plant] = build_case(cfg, P);
        res0 = runpf(mpc0, mpopt);
        assert(res0.success == 1, 'base PF');
        sl = plant.slack;
        Sn = plant.Sn(:).';
        ns = setdiff(1:10, sl);
        P0 = zeros(1, 10);  Q0 = zeros(1, 10);
        for i = 1:10
            gi = find(mpc0.gen(:, GEN_BUS) == plant.bus(i), 1);
            P0(i) = res0.gen(gi, PG)/plant.Sn(i);
            Q0(i) = res0.gen(gi, QG)/plant.Sn(i);
        end
        p_hi = max(1.0, P0 + 1e-9);
        q_hi = max(0.8, abs(Q0) + 1e-9);
        nev = 0;
        evalc_ = @(d) eval_sigma(cfg, P, plant, res0, d, sl, mpopt);
        [sig0_chk, ok0] = evalc_(zeros(1, 20));  nev = nev + 1;
        assert(ok0, 'base rebuild');
        fprintf('  base sigma rebuilt %+.4f (row %+.4f)\n', sig0_chk, row.sigma_dom);
        best_d = zeros(1, 20);  best_sig = sig0_chk;
        note = "";
        for round_ = 1:4
            g = zeros(1, 20);
            for i = ns
                for c = [i, 10 + i]
                    dp = zeros(1, 20);  dm = zeros(1, 20);
                    dp(c) = DFD;  dm(c) = -DFD;
                    [sp_, okp] = evalc_(best_d + dp);
                    [sm_, okm] = evalc_(best_d + dm);
                    nev = nev + 2;
                    if okp && okm, g(c) = (sp_ - sm_)/(2*DFD); end
                end
            end
            if norm(g) < 1e-6, note = "zero gradient"; break; end
            improved = false;
            for step = [0.30 0.15 0.08 0.04 0.02]
                d_try = proj(best_d - step*g/norm(g), sl, ns, Sn, P0, Q0, p_hi, q_hi);
                [s_try, ok] = evalc_(d_try);  nev = nev + 1;
                if ok && s_try < best_sig - 1e-4
                    best_sig = s_try;  best_d = d_try;  improved = true;
                    fprintf('  round %d: step %.2f -> sigma %+.4f (cost %.3f)\n', ...
                        round_, step, best_sig, norm(best_d));
                    break
                end
            end
            if ~improved, note = "line search stalled"; break; end
            if best_sig <= ALARM, note = "alarm cleared"; break; end
        end
        cost = norm(best_d);
        cost_mw = sum(abs(best_d(1:10)).*Sn);
        fprintf(fid, '%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%d,%d,%d,%.1f,%s\n', ...
            id, stratlab(q), row.sigma_dom, sig0_chk, best_sig, cost, cost_mw, ...
            best_sig < 0, best_sig <= ALARM, nev, toc(tC), note);
        fprintf('  -> best sigma %+.4f cost %.3f pu / %.0f MW (stab %d, alarm %d, %d evals, %.0f s)\n', ...
            best_sig, cost, cost_mw, best_sig < 0, best_sig <= ALARM, nev, toc(tC));
    catch ME
        fprintf(fid, '%d,%d,%.6f,NaN,NaN,NaN,NaN,0,0,0,%.1f,FAILED %s\n', ...
            id, stratlab(q), row.sigma_dom, toc(tC), ...
            strrep(strrep(char(ME.message), ',', ';'), newline, ' '));
        fprintf('  FAILED: %s\n', ME.message);
    end
end
fclose(fid);
fprintf('r17_rec complete (%.1f min).\n', toc(t00)/60);
end

% ---------------------------------------------------------------------------
function d = proj(d, sl, ns, Sn, P0, Q0, p_hi, q_hi)
%PROJ  Projection as in q9: slack pinned, Sn-weighted MW conservation on
%   the non-slack dP, and a box on the P and Q levels.
d([sl, 10 + sl]) = 0;
for it = 1:4
    w = Sn(ns);
    d(ns) = d(ns) - w*(w*d(ns).')/(w*w.');
    Pn = P0 + d(1:10);
    Pn = min(max(Pn, 0), p_hi);
    d(1:10) = Pn - P0;
    d(sl) = 0;
end
Qn = Q0 + d(11:20);
Qn = min(max(Qn, -q_hi), q_hi);
d(11:20) = Qn - Q0;
d([sl, 10 + sl]) = 0;
end

function [sig, ok] = eval_sigma(cfg, P, plant, res0, d, sl, mpopt)
%EVAL_SIGMA  Same as in q9, plus the out-of-band (oob) truth convention of the bank.
define_constants;
sig = NaN;  ok = false;
try
    [mpc, ~] = build_case(cfg, P);
    if any(abs(d) > 1e-15)
    for i = 1:10
        if i == sl, continue; end
        gi = find(mpc.gen(:, GEN_BUS) == plant.bus(i), 1);
        mpc.gen(gi, PG) = res0.gen(gi, PG) + d(i)*plant.Sn(i);
        mpc.gen(gi, QG) = res0.gen(gi, QG) + d(10 + i)*plant.Sn(i);
        mpc.bus(mpc.bus(:, BUS_I) == plant.bus(i), BUS_TYPE) = 1;   % PQ
    end
    end
    mpc.bus(:, VM) = res0.bus(:, VM);
    mpc.bus(:, VA) = res0.bus(:, VA);
    res = runpf(mpc, mpopt);
    if res.success ~= 1, return; end
    M = full.build_model(mpc, res, plant, P, cfg.tune);
    [x0, M, ~] = full.init_state(M, mpc, res);
    L = full.linearize(M, x0, 'cs');
    resid = norm(full.rhs(x0, M), inf);  it = 0;
    while resid > 1e-13 && it < 4
        x0 = x0 - lsqminnorm(L.Asys, full.rhs(x0, M));
        resid = norm(full.rhs(x0, M), inf);  it = it + 1;
    end
    if it > 0, L = full.linearize(M, x0, 'cs'); end
    S = full.modal(L.Asys, P);
    sig = S.sigma_dom;
    if ~S.oob_stable, sig = max(sig, S.oob_maxRe); end
    ok = true;
catch ME_inner
    fprintf('    [eval fail] %s\n', ME_inner.message);
end
end

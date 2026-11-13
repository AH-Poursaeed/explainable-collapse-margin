function q9_recourse_physics
%Q9_RECOURSE_PHYSICS  Physics-side redispatch search for unsafe recourse cases.
%   It covers the 12 marginal unsafe recourse cases. The search of
%   e2_recourse_margin.py walks the SURROGATE with a step-norm prior capped
%   at 0.15 pu. Its 11 "no exit" outcomes are failures of the surrogate
%   search, not of the physics. This script searches the FULL model directly,
%   WITHOUT the 0.15 radius cap. It uses projected finite-difference
%   gradient descent of sigma_dom over the same dispatch box: Pn in
%   [0, max(1, P0)], |Qn| <= max(0.8, |Q0|), slack pinned to zero, and
%   Sn-weighted MW conservation over non-slack dP.
%
%   Per case: gradient via +-0.02 pu central differences (non-slack P and
%   Q), backtracking projected steps, re-gradient every 3 accepted steps,
%   at most 4 gradient passes. Records the best sigma reached, its cost, and
%   two levels: stabilized (sigma < 0) and alarm-clearing (sigma <= -2.191).
%
%   Output: results/final/tables/q9_recourse_physics2.csv (checkpointed).
%

maxNumCompThreads(2);
P = params.sys_const();
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..');
fout = fullfile(root, 'results', 'final', 'tables', 'q9_recourse_physics2.csv');
Tc = readtable(fullfile(root, 'results', 'final', 'e2_recourse_candidates.csv'));
Tb = readtable(fullfile(root, 'data', 'dataset.csv'));
define_constants;
mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1, ...
    'pf.tol', 1e-10, 'pf.nr.max_it', 40);
ALARM = -2.19073;             % the OOF alarm on the margin (e1_oof_MD_ebm)
DFD = 0.02;                   % probe size, converter pu (as in e2_sens_production.m)

done = [];
if isfile(fout)
    T0 = readtable(fout);
    done = T0.id;
    fid = fopen(fout, 'a');
else
    fid = fopen(fout, 'w');
    fprintf(fid, ['id,sigma0,found_surrogate,sigma_best,cost_pu,cost_mw,', ...
        'stabilised,clears_alarm,n_evals,note\n']);
end

for q = 1:height(Tc)
    r = Tc(q, :);
    id = r.id;
    if any(done == id), fprintf('skip id=%d (done)\n', id); continue; end
    row = Tb(Tb.id == id, :);
    if isnumeric(r.found), fnd = double(r.found ~= 0);
    else, fnd = double(strcmpi(strtrim(string(r.found)), "true"));
    end
    fprintf('=== case id=%d sigma0=%+.4f ===\n', id, r.sigma_dom0);
    try
        cfg = bank.row2cfg(row);
        [mpc0, plant] = build_case(cfg, P);
        res0 = runpf(mpc0, mpopt);
        assert(res0.success == 1, 'base PF');
        sl = r.slack_plant;                    % 1-based plant index
        Sn = plant.Sn(:).';                   % MVA
        ns = setdiff(1:10, sl);
        % base dispatch in converter pu (match the python feature convention)
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
        fprintf('  base sigma rebuilt %+.4f (row %+.4f)\n', sig0_chk, r.sigma_dom0);
        best_d = zeros(1, 20);  best_sig = sig0_chk;
        note = "";
        for round_ = 1:4
            % ---- FD gradient over non-slack (P then Q): raw single-plant
            % probes, as in e2_sens_production.m (the PF slack absorbs the
            % imbalance). The projection acts at STEP time, not on the probes.
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
            % ---- backtracking projected line search -------------------------
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
        fprintf(fid, '%d,%.6f,%d,%.6f,%.6f,%.2f,%d,%d,%d,%s\n', ...
            id, r.sigma_dom0, fnd, best_sig, cost, cost_mw, ...
            best_sig < 0, best_sig <= ALARM, nev, note);
        fprintf('  -> best sigma %+.4f cost %.3f pu / %.0f MW (stab %d, alarm %d, %d evals)\n', ...
            best_sig, cost, cost_mw, best_sig < 0, best_sig <= ALARM, nev);
    catch ME
        fprintf(fid, '%d,%.6f,%d,NaN,NaN,NaN,0,0,0,FAILED %s\n', ...
            id, r.sigma_dom0, fnd, strrep(char(ME.message), ',', ';'));
        fprintf('  FAILED: %s\n', ME.message);
    end
end
fclose(fid);
fprintf('q9_recourse_physics complete.\n');
end

% ---------------------------------------------------------------------------
function d = proj(d, sl, ns, Sn, P0, Q0, p_hi, q_hi)
%PROJ project a candidate (dP, dQ) onto the constraint set: slack pinned,
%   Sn-weighted MW conservation on non-slack dP, box on P and Q levels.
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
%EVAL_SIGMA rebuild the case with (P0+dP, Q0+dQ) pinned at non-slack plants
%   and return sigma_dom.
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
    % warm start from the base solution: small dispatch probes on
    % deep-stress cases fail from a flat/case start
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
    ok = true;
catch ME_inner
    fprintf('    [eval fail] %s\n', ME_inner.message);
end
end

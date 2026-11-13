function q1_hbar_drift_b
%Q1_HBAR_DRIFT_B  Second worker (B) for the A-level sweep of q1_hbar_drift.
%   Hbar was derived for the robustness of lambda_min to SINGLE-DEVICE
%   self-stiffness drift. This script tests that on the bank rows.
%
%   For every bank row: rebuild the reduced chain (trace-checked against the
%   stored lam_min/Hbar/PWSGSI). Then perturb each GFM device's stiffening
%   susceptance bin_i by +-10% and +-20% via the pass-through identity
%   Btilde_red = Bkron + Bin_sys (checked in build_B), and record:
%     - realized worst-case single-device erosion of lambda_min (10%, 20%)
%     - the first-order prediction delta*max_i(p_i*b_i,conv) of the first-order bound
%     - max participation over GFM devices and over all devices
%   Unlike q1_hbar_drift, this worker runs the bank rows in reverse order
%   and does not run the stratified 60-row full-model margin subset.
%
%   Output (results/final/tables/):
%     q1_hbar_drift_b.csv   per-row A-level results (checkpointed, resumable)
%

maxNumCompThreads(2);
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..');
outdir = fullfile(root, 'results', 'final', 'tables');
outcsv = fullfile(outdir, 'q1_hbar_drift_b.csv');
outmar = fullfile(outdir, 'q1_hbar_margin.csv');
P = params.sys_const();
DELTAS = [0.10 0.20];

Tb = readtable(fullfile(root, 'data', 'dataset.csv'));
fprintf('bank rows: %d\n', height(Tb));
% row-slicing a 134-column table inside a loop is very slow (~4 s/row,
% measured); so convert it once to a struct array (fast to index)
Sb = table2struct(Tb);

% ---- checkpoint resume ------------------------------------------------------
done = containers.Map('KeyType', 'double', 'ValueType', 'logical');
if isfile(outcsv)
    T0 = readtable(outcsv);
    for i = 1:height(T0), done(T0.id(i)) = true; end
    fprintf('resume: %d rows already done\n', done.Count);
    fid = fopen(outcsv, 'a');
else
    fid = fopen(outcsv, 'w');
    fprintf(fid, ['id,lam_min_row,lam_min_re,Hbar_row,Hbar_re,PWSGSI_row,trace_ok,', ...
        'nGFM,maxp_all,maxp_gfm,maxpb_gfm,bound_NcHbar,', ...
        'ero10,ero10_pred,ero20,lam_post10,lam_post20,worst_dev,ero10_any\n']);
end

mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1, 'pf.tol', 1e-12);
nfail = 0;
t0 = tic;
for r = height(Tb):-1:1
    id = Sb(r).id;
    if isKey(done, id), continue; end
    row = Sb(r);
    try
        cfg = bank.row2cfg(row);
        [mpc, plant] = build_case(cfg, P);
        res = runpf(mpc, mpopt);
        assert(res.success == 1, 'pf');
        R = reduced.chain(mpc, res, plant, P, cfg.tune);
        BB = R.BB;
        % trace check vs the stored row (rebuild must reproduce the bank)
        tr_ok = abs(R.lam_min - row.lam_min) < 1e-6 && abs(R.Hbar - row.Hbar) < 1e-6;
        isGFM = strcmp(plant.type(:), 'VSM');
        bconv = diag(BB.Bin_sys) ./ BB.Sb;              % converter-base bin_i
        p = BB.p(:);
        maxp_all = max(p);
        maxp_gfm = max(p(isGFM));
        maxpb = max(p(isGFM) .* bconv(isGFM));          % first-order worst dP/ddelta
        Nc = numel(p);
        bound = Nc^(-row.Hbar);                          % min-entropy floor on maxp
        % ---- perturbation sweep (pass-through identity: diagonal update) ----
        Sih = diag(1./sqrt(BB.Sb));
        base = BB.Bkron + BB.Bin_sys;
        lam_post = zeros(numel(DELTAS), 1);
        ero = zeros(numel(DELTAS), 1);
        widx = NaN;
        for d = 1:numel(DELTAS)
            worst = inf;
            for i = find(isGFM).'
                for sgn = [-1 1]
                    B2 = base;
                    B2(i, i) = B2(i, i) + sgn*DELTAS(d)*BB.Sb(i)*bconv(i);
                    A2 = Sih*((B2 + B2.')/2)*Sih;
                    lm = min(eig(A2));
                    if lm < worst, worst = lm; if d == 1, widx = i; end, end
                end
            end
            lam_post(d) = worst;
            ero(d) = R.lam_min - worst;                 % >0 means erosion
        end
        ero10_pred = DELTAS(1)*maxpb;                    % first-order bound
        % secondary: direct A_ii drift at ANY device bus (the object of the bound)
        A0 = Sih*((base + base.')/2)*Sih;
        worst_any = inf;
        for i = 1:Nc
            for sgn = [-1 1]
                A2 = A0;
                A2(i, i) = A2(i, i)*(1 + sgn*DELTAS(1));
                lm = min(eig(A2));
                if lm < worst_any, worst_any = lm; end
            end
        end
        ero10_any = R.lam_min - worst_any;
        fprintf(fid, '%d,%.8g,%.8g,%.8g,%.8g,%.8g,%d,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d,%.8g\n', ...
            id, row.lam_min, R.lam_min, row.Hbar, R.Hbar, row.PWSGSI, tr_ok, ...
            sum(isGFM), maxp_all, maxp_gfm, maxpb, bound, ...
            ero(1), ero10_pred, ero(2), lam_post(1), lam_post(2), widx, ero10_any);
    catch ME
        nfail = nfail + 1;
        fprintf('row id=%d FAILED: %s\n', id, ME.message);
    end
    if mod(r, 200) == 0
        fprintf('%d/%d rows, %.1f min elapsed, %d fails\n', r, height(Tb), toc(t0)/60, nfail);
    end
end
fclose(fid);
fprintf('A-level sweep done (%d fails). Elapsed %.1f min\n', nfail, toc(t0)/60);

fprintf('worker B: A-level sweep complete.\n');
end

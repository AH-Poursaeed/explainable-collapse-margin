function q1_hbar_drift
%Q1_HBAR_DRIFT  Empirical test of the entropy factor Hbar across the bank.
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
%   A stratified 60-row subset also re-solves the FULL model with the worst
%   device's inner effective inductance Leff scaled (the physical form of
%   the same drift) and records the realized movement of the margin.
%
%   Output (results/final/tables/):
%     q1_hbar_drift.csv   per-row A-level results (checkpointed, resumable)
%     q1_hbar_margin.csv  full-model margin subset
%

maxNumCompThreads(2);
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..');
outdir = fullfile(root, 'results', 'final', 'tables');
outcsv = fullfile(outdir, 'q1_hbar_drift.csv');
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
for r = 1:height(Tb)
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

% ---- full-model margin subset ------------------------------------------------
% Stratified across Hbar deciles among rows with nGFM >= 2 (for single-GFM
% rows, Hbar carries no information on drift across devices).
if ~isfile(outmar)
    Tq = readtable(outcsv);
    Tq = Tq(Tq.trace_ok == 1 & Tq.nGFM >= 2, :);
    edges = quantile(Tq.Hbar_row, 0:0.1:1);
    pickid = [];
    rng(42);
    for b = 1:10
        in = Tq.id(Tq.Hbar_row >= edges(b) & Tq.Hbar_row <= edges(b+1));
        in = in(randperm(numel(in)));
        pickid = [pickid; in(1:min(6, numel(in)))];     %#ok<AGROW>
    end
    fprintf('margin subset: %d rows\n', numel(pickid));
    fm = fopen(outmar, 'w');
    fprintf(fm, ['id,sigma_row,sigma_re,worst_dev,delta,', ...
        'sigma_kXminus,sigma_kXplus,lam_kXminus,lam_kXplus,', ...
        'bin0,bin_kXminus,bin_kXplus,ero_margin\n']);
    for q = 1:numel(pickid)
        id = pickid(q);
        row = Tb(Tb.id == id, :);
        wd = Tq.worst_dev(Tq.id == id);
        try
            cfg = bank.row2cfg(row);
            o = run_scenario(cfg);
            assert(isempty(o.fail), o.fail);
            sig0 = o.S.sigma_dom;
            [mpc0, plant0] = build_case(cfg, P);
            res0 = runpf(mpc0, mpopt);
            Rc0 = reduced.chain(mpc0, res0, plant0, P, cfg.tune);
            b0 = Rc0.BB.Bin_sys(wd, wd)/Rc0.BB.Sb(wd);
            sigs = nan(1, 2);  lams = nan(1, 2);  bins = nan(1, 2);
            for si = 1:2
                sgn = 2*si - 3;                          % -1, +1
                cfg2 = cfg;
                % physical drift: the worst device's LCL reactances scale by
                % kX AFTER the nameplate controller tune (the hardware-drift
                % hook of params.converter). kX = 1/(1 +- 0.10) moves
                % b ~ 1/Leff up or down. The REALIZED stiffness drift is
                % identified again, not assumed.
                cfg2.tune(wd).kX = 1/(1 + sgn*0.10);
                o2 = run_scenario(cfg2);
                if isempty(o2.fail), sigs(si) = o2.S.sigma_dom; end
                [mpc2, plant2] = build_case(cfg2, P);
                res2 = runpf(mpc2, mpopt);
                if res2.success == 1
                    Rc2 = reduced.chain(mpc2, res2, plant2, P, cfg2.tune);
                    lams(si) = Rc2.lam_min;
                    bins(si) = Rc2.BB.Bin_sys(wd, wd)/Rc2.BB.Sb(wd);
                end
            end
            fprintf(fm, '%d,%.8g,%.8g,%d,%.2f,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n', ...
                id, row.sigma_dom, sig0, wd, 0.10, sigs(1), sigs(2), ...
                lams(1), lams(2), b0, bins(1), bins(2), max(sigs) - sig0);
            fprintf('[margin %d/%d] id=%d sig0=%+.4f -> [%+.4f, %+.4f]\n', ...
                q, numel(pickid), id, sig0, sigs(1), sigs(2));
        catch ME
            fprintf('margin id=%d FAILED: %s\n', id, ME.message);
        end
    end
    fclose(fm);
end
fprintf('q1_hbar_drift complete.\n');
end

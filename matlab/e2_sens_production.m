function e2_sens_production
%E2_SENS_PRODUCTION  Faithfulness-at-scale ground truth.
%   Central-FD probes of every plant's P and Q on a sigma-stratified subset
%   of the HELD-OUT TEST rows. For EVERY solve it exports the full measurable
%   feature block AND the modal truth. So the model's TOTAL derivative along
%   the same physical probe can be compared with the analytic
%   d(sigma_dom)/d(P_i,Q_i).
%
%   Two probe granularities (paired rows):
%     fine   +-0.02 pu on N_FINE rows  (the standard step for the gradients)
%     coarse +-0.10 pu on a stratified subset of the SAME rows (at this step
%            the signed fidelity starts to rise above the model MAE)
%
%   Outputs (results/final/):
%     e2_probe_features.csv         (fine,   probe in {base,Pp_i,Pm_i,Qp_i,Qm_i})
%     e2_probe_features_coarse.csv  (coarse, same layout, base re-used)
%
%   Conventions: seed 42 (params.sys_const), test rows from
%   results/ml_export/test_frame.csv. Expect ~25% of the rows to be lost to
%   PF failures from Q-pinning (so the pick has headroom: 320 requested).

N_FINE = 320;                  % requested rows (headroom for ~25% loss)
N_COARSE = 120;                % subset of the same picks, +-0.10 pu
DP_FINE = 0.02;                % converter-base pu, central difference
DP_COARSE = 0.10;

P = params.sys_const();
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'final');
if ~exist(outdir, 'dir'), mkdir(outdir); end
T = readtable(fullfile(here, '..', 'data', 'dataset.csv'));
Tt = readtable(fullfile(here, '..', 'results', 'ml_export', 'test_frame.csv'));
T = T(ismember(T.id, Tt.id), :);
rng(P.seed, 'twister');

% sigma-stratified pick: deep-stable bulk, near-boundary shell (both sides),
% unstable side.
edges = [-inf, -3.5, -2, -1, 0, 2, inf];
pick = [];
per = ceil(N_FINE/(numel(edges)-1));
for b = 1:numel(edges)-1
    idb = find(T.sigma_dom > edges(b) & T.sigma_dom <= edges(b+1));
    n_b = min(numel(idb), per);
    if n_b > 0
        pick = [pick; idb(randperm(numel(idb), n_b))];         %#ok<AGROW>
    end
end
% coarse subset: evenly spaced through the band-grouped pick vector
% (preserves rough stratification), paired with the fine rows.
csel = unique(round(linspace(1, numel(pick), min(N_COARSE, numel(pick)))));
pickc = pick(csel);
fprintf('e2: %d fine rows (of requested %d), %d coarse rows; 41 probes each\n', ...
    numel(pick), N_FINE, numel(pickc));

probes_f = mkprobes();
npool = 4;
if ~isempty(getenv('E2_POOL')), npool = str2double(getenv('E2_POOL')); end
try
    if isempty(gcp('nocreate')), parpool('Processes', npool); end
catch
    fprintf('parpool unavailable; running serial\n');
end
fprintf('e2 pool size %d\n', npool);

% safe restart: every scenario's 41-probe block is saved to
% e2_ckpt/<phase>_<id>.mat when it is done. A restart computes only the
% missing rows, and a finished phase (final CSV on disk) is skipped whole.
ckdir = fullfile(outdir, 'e2_ckpt');
if ~exist(ckdir, 'dir'), mkdir(ckdir); end

t0 = tic;
fine_csv = fullfile(outdir, 'e2_probe_features.csv');
if exist(fine_csv, 'file')
    fprintf('fine set already on disk (%s) -- skipped\n', fine_csv);
else
    rowsc = cell(numel(pick), 1);
    parfor q = 1:numel(pick)
        rowsc{q} = one_row_ckpt(T(pick(q), :), probes_f, DP_FINE, P, ...
                                ckdir, 'fine');
    end
    L = vertcat(rowsc{:});
    writetable(struct2table(L), fine_csv);
    fprintf('e2_probe_features.csv written: %d lines (%.0f s)\n', ...
        numel(L), toc(t0));
end

t1 = tic;
coarse_csv = fullfile(outdir, 'e2_probe_features_coarse.csv');
if exist(coarse_csv, 'file')
    fprintf('coarse set already on disk -- skipped\n');
else
    rowsc = cell(numel(pickc), 1);
    parfor q = 1:numel(pickc)
        rowsc{q} = one_row_ckpt(T(pickc(q), :), probes_f, DP_COARSE, P, ...
                                ckdir, 'coarse');
    end
    L = vertcat(rowsc{:});
    writetable(struct2table(L), coarse_csv);
    fprintf('e2_probe_features_coarse.csv written: %d lines (%.0f s)\n', ...
        numel(L), toc(t1));
end
fprintf('e2 total %.0f s\n', toc(t0));
end

% -----------------------------------------------------------------------------
function out = one_row_ckpt(r, probes, dP, P, ckdir, phase)
%ONE_ROW_CKPT one_row with a per-scenario checkpoint (parfor-safe: one
%   distinct file per scenario id and phase).
fn = fullfile(ckdir, sprintf('%s_%d.mat', phase, r.id));
if exist(fn, 'file')
    S = load(fn, 'out');
    out = S.out;
    return
end
out = one_row(r, probes, dP, P);
save(fn, 'out');
end

% -----------------------------------------------------------------------------
function probes = mkprobes()
probes = {'base', 0, 0};
for i = 1:10
    probes(end+1, :) = {sprintf('Pp_%d', i),  i, 0};           %#ok<AGROW>
    probes(end+1, :) = {sprintf('Pm_%d', i), -i, 0};           %#ok<AGROW>
    probes(end+1, :) = {sprintf('Qp_%d', i), 0,  i};           %#ok<AGROW>
    probes(end+1, :) = {sprintf('Qm_%d', i), 0, -i};           %#ok<AGROW>
end
end

function out = one_row(r, probes, dP, P)
out = repmat(blank(r.id), size(probes, 1), 1);
for p = 1:size(probes, 1)
    ip = probes{p, 2};  iq = probes{p, 3};
    dPc = zeros(1, 10);  dQc = zeros(1, 10);
    if ip ~= 0, dPc(abs(ip)) = sign(ip)*dP; end
    if iq ~= 0, dQc(abs(iq)) = sign(iq)*dP; end
    s = solve_probe(r, dPc, dQc, P);
    s.id = r.id;  s.probe = string(probes{p, 1});
    out(p) = s;
end
end

function s = blank(id)
s = struct('id', id, 'probe', "", 'ok', false, 'sigma_dom', NaN, ...
    'zeta_min', NaN, 'lam_min', NaN, 'Hbar', NaN, 'PWSGSI', NaN, ...
    'Ptot', NaN, 'Qtot', NaN);
for i = 1:10
    s.(sprintf('Pinv_%d', i)) = NaN;  s.(sprintf('Qinv_%d', i)) = NaN;
    s.(sprintf('Vbus_%d', i)) = NaN;  s.(sprintf('Yeff_%d', i)) = NaN;
    s.(sprintf('Yhat_%d', i)) = NaN;
end
end

function s = solve_probe(r, dPc, dQc, P)
%SOLVE_PROBE one perturbed operating point: PF -> full model -> modal truth
%   + the exact bank feature block (build_bank conventions) + chain indices.
%   device_probe_solve.m repeats it statement for statement.
s = blank(r.id);
try
    cfg = bank.row2cfg(r);
    [mpc, plant] = build_case(cfg, P);
    define_constants;
    for i = 1:10
        if i == plant.slack, continue; end
        mpc.gen(i, PG) = mpc.gen(i, PG) + dPc(i)*plant.Sn(i);
    end
    mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1, 'pf.tol', 1e-10);
    if any(dQc)
        res0 = runpf(mpc, mpopt);
        if res0.success ~= 1, return; end
        for i = 1:10
            if i == plant.slack, continue; end
            mpc.bus(mpc.bus(:, BUS_I) == plant.bus(i), BUS_TYPE) = 1;   % PQ
            mpc.gen(i, QG) = res0.gen(i, QG) + dQc(i)*plant.Sn(i);
            mpc.gen(i, QMIN) = mpc.gen(i, QG);  mpc.gen(i, QMAX) = mpc.gen(i, QG);
        end
    end
    res = runpf(mpc, mpopt);
    if res.success ~= 1, return; end
    tune = cfg.tune;
    M = full.build_model(mpc, res, plant, P, tune);
    [x0, M, ~] = full.init_state(M, mpc, res);
    L = full.linearize(M, x0, 'cs');
    dx = - lsqminnorm(L.Asys, full.rhs(x0, M));
    x0 = x0 + dx;
    L = full.linearize(M, x0, 'cs');
    S = full.modal(L.Asys, P);
    R = reduced.chain(mpc, res, plant, P, tune);

    s.ok = true;
    s.sigma_dom = S.sigma_dom;
    z = S.zeta_min;
    if ~S.oob_stable, z = min(z, -1e-6); end
    s.zeta_min = z;
    s.lam_min = R.lam_min;  s.Hbar = R.Hbar;  s.PWSGSI = R.PWSGSI;

    Pg = res.gen(:, PG);  Qg = res.gen(:, QG);
    Vm = res.bus(ismember(res.bus(:, BUS_I), plant.bus), VM);
    Pconv = Pg ./ plant.Sn;  Qconv = Qg ./ plant.Sn;
    Sapp = sqrt(Pg.^2 + Qg.^2) ./ plant.Sn;
    Yht = Sapp(:)./Vm(:).^2;
    for i = 1:10
        s.(sprintf('Pinv_%d', i)) = Pconv(i);
        s.(sprintf('Qinv_%d', i)) = Qconv(i);
        s.(sprintf('Vbus_%d', i)) = Vm(i);
        s.(sprintf('Yeff_%d', i)) = R.Yeff(i);
        s.(sprintf('Yhat_%d', i)) = Yht(i);
    end
    s.Ptot = sum(Pg)/100;  s.Qtot = sum(Qg)/100;
catch
    % leave blank (ok = false)
end
end

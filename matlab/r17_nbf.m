function r17_nbf
%r17_nbf  Small-step injection probes on held-out test rows near the boundary.
%   The probes use a reduced step (+-0.005 pu) on NEAR-BOUNDARY held-out
%   test rows (|sigma_dom| <= 1). They measure faithfulness at the depth
%   where the +-0.02/+-0.10 probe steps of the paper lose scenarios. The
%   probe layout is the same as in e2_sens_production (base + P/Q +- on all
%   10 plants, 41 solves per scenario), so the e2r17 faithfulness code reads
%   the output unchanged.
%
%   What the script prints: the RUNNING yield of converged PF/init solves
%   after every scenario. The Python side computes rank statistics per
%   scenario, on the scenarios and probes whose solves converge.
%
%   Deterministic under P.seed + 42. Serial. Each scenario saves a .mat
%   checkpoint. The final CSV is built from all checkpoints on disk, so a
%   run that stops at the time limit still writes a partial CSV.
%
%   Outputs: results/tables/r17/r17_nbf_probes.csv
%            results/tables/r17/r17_nbf_yield.csv

maxNumCompThreads(2);
N_NBF = 60;
DP    = 0.005;
TBOX  = 160*60;
t00 = tic;

P = params.sys_const();
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'tables', 'r17');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ckdir = fullfile(outdir, 'r17_nbf_ckpt');
if ~exist(ckdir, 'dir'), mkdir(ckdir); end

T = readtable(fullfile(here, '..', 'data', 'dataset.csv'));
Tt = readtable(fullfile(here, '..', 'results', 'ml_export', 'test_frame.csv'));
T = T(ismember(T.id, Tt.id) & abs(T.sigma_dom) <= 1, :);
fprintf('near-boundary held-out pool: %d rows (|sigma| <= 1)\n', height(T));

% stratified pick, one row from each stratum in turn: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]
edges = [-1, -0.5, 0, 0.5, 1 + 1e-9];
rng(P.seed + 42, 'twister');
per = ceil(N_NBF/(numel(edges)-1));
strat = cell(numel(edges)-1, 1);
for b = 1:numel(edges)-1
    idb = find(T.sigma_dom >= edges(b) & T.sigma_dom < edges(b+1));
    idb = idb(randperm(numel(idb)));
    strat{b} = idb(1:min(per, numel(idb)));
    fprintf('stratum %d [%g,%g): %d rows, taking %d\n', b, edges(b), ...
        edges(b+1), numel(idb), numel(strat{b}));
end
order = [];
for j = 1:per
    for b = 1:numel(strat)
        if j <= numel(strat{b}), order(end+1) = strat{b}(j); end %#ok<AGROW>
    end
end

probes = mkprobes();
smoke = ~isempty(getenv('SMOKE_RUN'));
nrun = numel(order);  if smoke, nrun = 1; end

nok_tot = 0;  nprobe_tot = 0;
yield = [];
for q = 1:nrun
    r = T(order(q), :);
    fn = fullfile(ckdir, sprintf('nbf_%d.mat', r.id));
    if exist(fn, 'file')
        S = load(fn, 'out');  out = S.out;
        fprintf('row %d/%d id=%d: checkpoint exists, skip\n', q, nrun, r.id);
    else
        if toc(t00) > TBOX
            fprintf('Time limit reached at row %d/%d (%.0f min) -- assembling partial CSV\n', ...
                q, nrun, toc(t00)/60);
            break
        end
        tR = tic;
        out = one_row(r, probes, DP, P);
        save(fn, 'out');
        fprintf('row %d/%d id=%d sigma=%+.3f: %d/%d probes ok (%.0f s)\n', ...
            q, nrun, r.id, r.sigma_dom, nnz([out.ok]), numel(out), toc(tR));
    end
    nok = nnz([out.ok]);
    nok_tot = nok_tot + nok;  nprobe_tot = nprobe_tot + numel(out);
    yield = [yield; struct('id', r.id, 'sigma_dom', r.sigma_dom, ...
        'n_ok', nok, 'n_probe', numel(out))];                    %#ok<AGROW>
    fprintf('  RUNNING YIELD: %.1f%% (%d/%d solves)\n', ...
        100*nok_tot/nprobe_tot, nok_tot, nprobe_tot);
end

% assemble CSV from every checkpoint on disk (complete or partial)
d = dir(fullfile(ckdir, 'nbf_*.mat'));
rows = cell(numel(d), 1);
for i = 1:numel(d)
    S = load(fullfile(ckdir, d(i).name), 'out');
    rows{i} = S.out;
end
L = vertcat(rows{:});
writetable(struct2table(L), fullfile(outdir, 'r17_nbf_probes.csv'));
if ~isempty(yield)
    writetable(struct2table(yield), fullfile(outdir, 'r17_nbf_yield.csv'));
end
fprintf('r17_nbf: %d probe lines from %d scenarios written (%.1f min total)\n', ...
    numel(L), numel(d), toc(t00)/60);
end

% ===========================================================================
% The same code as in e2_sens_production (probe layout and solve steps).
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

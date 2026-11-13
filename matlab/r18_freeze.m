function r18_freeze(stage)
%r18_freeze  Build a fresh frozen scenario bank under a new master seed.
%   The bank is generated after every configuration choice was frozen. The
%   shipped models are scored once against this bank. No model or threshold
%   may be tuned on it.
%
%   The code copies build_bank.m (the published generator) exactly, with
%   three deviations:
%     (D1) the scenario sampling stream is seeded with FREEZE_SEED (main
%          round) and FREEZE_SEED+1 (top-up round), not P.seed and P.seed+1.
%     (D2) the 16-outage pool is still drawn under the published master
%          seed 42 (build_bank.m:46, "pool ALWAYS from the master seed").
%          The pool is a design constant of the published bank, not random
%          per bank, so the frozen bank samples the same design space.
%     (D3) outputs go to results/bank_r18f[_topup]/ and
%          data/dataset_r18freeze.csv. The code never writes data/dataset.csv.
%   The two rounds follow the published spec files:
%   main N=6000 + topup N=2500 = 8,500 draws (bank_spec.mat, measured).
%
%   Stages: 'repro'    reproduction gate. It re-runs 8 published rows (4 per
%                      round) through the copied chain and asserts a relative
%                      difference <= 1e-9 against data/dataset.csv (expected
%                      ~1e-12, the floor of the CSV round trip). This is a
%                      hard gate: generation does not start if it fails.
%           'main'     generate the 6000-draw main round (parfor in chunks,
%                      one checkpoint per chunk, safe to re-run).
%           'topup'    generate the 2500-draw top-up round.
%           'assemble' merge, run the class-0 integrity audit, write data/dataset_r18freeze.csv
%                      and .mat, and write the failure ledger to results/tables/r18/.
%           'all'      repro -> main -> topup -> assemble (default).

if nargin < 1, stage = 'all'; end
FREEZE_SEED = 20260812;               % new master seed, chosen before
                                      % generation and not used before
                                      % in this project
N_MAIN  = 6000;
N_TOPUP = 2500;
here    = fileparts(mfilename('fullpath'));
resdir  = fullfile(here, '..', 'results');
triadir = fullfile(resdir, 'tables', 'r18');
if ~exist(triadir, 'dir'), mkdir(triadir); end

t0 = tic;
switch stage
    case 'all'
        r18_freeze('repro');
        r18_freeze('main');
        r18_freeze('topup');
        r18_freeze('assemble');
        fprintf('frozen bank: all done in %.1f min\n', toc(t0)/60);
        return
    case 'repro'
        do_repro(here, resdir, triadir);
        return
    case 'main'
        do_round('main',  N_MAIN,  FREEZE_SEED,     'bank_r18f', here, resdir);
        return
    case 'topup'
        do_round('topup', N_TOPUP, FREEZE_SEED + 1, 'bank_r18f_topup', here, resdir);
        return
    case 'assemble'
        do_assemble(here, resdir, triadir);
        return
    otherwise
        error('r18_freeze:stage', 'unknown stage %s', stage);
end
end

% ==========================================================================
function do_repro(here, resdir, triadir)
% Reproduction gate: the copied one_scenario chain must reproduce published
% dataset rows from the published spec draws before any generation runs.
fprintf('REPRO GATE start (%s)\n', datetime('now'));
% Single-threaded BLAS: the published rows were computed inside parfor
% workers (1 computational thread each), and so are the rows of the fresh
% bank. With several threads, eig moves near-marginal rows by ~5e-9
% (measured, id 12499). With one thread the copy reproduces them to 1.8e-15.
nt0 = maxNumCompThreads(1);
cleanup_nt = onCleanup(@() maxNumCompThreads(nt0));
Ds = readtable(fullfile(here, '..', 'data', 'dataset.csv'));
rounds = struct( ...
    'name',  {'main', 'topup'}, ...
    'dir',   {'bank', 'bank_topup'}, ...
    'idoff', {0, 10000});
cmpf = {'sigma_dom','zeta_min','f_dom','lam_min','Hbar','PWSGSI', ...
        'gSCR','SCR_min','Ptot','Qtot','residual_pu'};
rows = {};
maxrel_all = 0;
for rr = 1:2
    S = load(fullfile(resdir, rounds(rr).dir, 'bank_spec.mat'));
    Dr = Ds(Ds.round == rr, :);
    % Deterministic picks: the first unstable row, the first class-5 row, the first
    % row with an outage and >=3 GFM units, and the last id of the round
    picks = [ ...
        Dr.id(find(Dr.class == 0, 1, 'first')); ...
        Dr.id(find(Dr.class == 5, 1, 'first')); ...
        Dr.id(find(Dr.outage > 0 & Dr.nGFM >= 3, 1, 'first')); ...
        Dr.id(end)];
    picks = unique(picks, 'stable');
    for pp = 1:numel(picks)
        id = picks(pp);
        i  = id - rounds(rr).idoff;      % index into the round's spec arrays
        r = one_scenario_r18(id, S.spec.cfgs{S.cfgid(i)}, S.outid(i), ...
            S.kL(i), S.kk(i), S.kdis(i,:), S.fplc(i), S.fplj(i,:), ...
            S.Hvs(i,:), S.Dvs(i,:), S.kbw(i), S.spec);
        assert(r.ok, 'repro row id %d did not converge in re-run', id);
        drow = Ds(Ds.id == id, :);
        maxrel = 0;
        for f = cmpf
            a = r.(f{1});  b = drow.(f{1});
            rel = abs(a - b) / max([1e-12, abs(a), abs(b)]);
            maxrel = max(maxrel, rel);
        end
        okcls = (r.class == drow.class) && (double(r.abstain) == drow.abstain);
        fprintf('  repro id %5d (round %d): maxrel %.3e, class %d==%d %s\n', ...
            id, rr, maxrel, r.class, drow.class, string(okcls));
        rows{end+1} = {id, rr, maxrel, okcls};                     %#ok<AGROW>
        maxrel_all = max(maxrel_all, maxrel);
        assert(okcls, 'REPRO GATE: class/abstain mismatch at id %d', id);
    end
end
T = cell2table(vertcat(rows{:}), 'VariableNames', ...
    {'id','round','maxrel','class_match'});
writetable(T, fullfile(triadir, 'r18_freeze_repro.csv'));
fprintf('REPRO GATE max relative diff over %d rows x %d fields: %.3e\n', ...
    height(T), numel(cmpf), maxrel_all);
assert(maxrel_all <= 1e-9, ...
    'REPRO GATE FAILED: %.3e > 1e-9 -- generation blocked', maxrel_all);
fprintf('REPRO GATE PASS\n');
end

% ==========================================================================
function do_round(mode, N, round_seed, outname, here, resdir)
% The sampling block is copied from build_bank.m. Only deviations D1-D3 differ.
P = params.sys_const();
assert(P.seed == 42, 'published master seed changed -- pool derivation unsafe');
topup = strcmp(mode, 'topup');
fprintf('ROUND %s: N=%d seed=%d -> %s (%s)\n', mode, N, round_seed, ...
    outname, datetime('now'));

% ---- named GFM configurations (buses) and sampling weights, as in build_bank.m
cfgs = { [39], [38], [38 39], [35 39], [34 39], [35 38 39], [33 38 39], [33 35 38 39] };
if topup
    cfgp = [0.28  0.14  0.22   0.10     0.05     0.11        0.05        0.05];
else
    cfgp = [0.12  0.06  0.24   0.10     0.08     0.16        0.10        0.14];
end
% ---- outage pool: published master seed (deviation D2, see the help text) --
n1 = load(fullfile(resdir, 'phase1_n1.mat'));
rng(P.seed, 'twister');                          % the pool always uses seed 42
pool = n1.pool(randperm(numel(n1.pool), 16));
ood_branches = pool(13:16);
% ---- fresh scenario stream (deviation D1) ----------------------------------
rng(round_seed, 'twister');

% ---- LHS over the continuous axes (ranges per mode as in build_bank.m) -----
X = lhsdesign(N, 32, 'criterion', 'none');
if topup
    kL = 0.9  + X(:,1)*(1.2 - 0.9);
    kk = 1.2  + X(:,2)*(2.3 - 1.2);
    Dvs = 40  + X(:,28:31)*60;
else
    kL = 0.6  + X(:,1)*(1.2 - 0.6);
    kk = 0.6  + X(:,2)*(1.9 - 0.6);
    Dvs = 40  + X(:,28:31)*160;
end
kdis = 0.75 + X(:,3:12)*0.5;
fplc = 8    + X(:,13)*(35 - 8);
fplj = 0.8  + X(:,14:23)*0.4;
Hvs  = 2    + X(:,24:27)*8;
kbw  = 0.7  + X(:,32)*0.7;
cfgid  = randsample(numel(cfgs), N, true, cfgp);
hasout = rand(N,1) < 0.5;
outid  = pool(randi(16, N, 1)).';
outid(~hasout) = 0;
id_off = 0; if topup, id_off = 10000; end

spec = struct('N', N, 'cfgs', {cfgs}, 'cfgp', cfgp, 'pool', pool, ...
    'ood_branches', ood_branches, 'seed', round_seed, 'mode', mode, ...
    'ranges', struct('kL', [min(kL) max(kL)], 'k', [min(kk) max(kk)], ...
    'kdisp', [0.75 1.25], 'fpll', [8 35], 'fpll_jit', [0.8 1.2], ...
    'Hv', [2 10], 'Dv', [min(Dvs(:)) max(Dvs(:))], 'kbw', [0.7 1.4]), ...
    'delta_abst', 0.0015, 'bands', [0 0.01 0.03 0.05 0.10]);
% provenance of the frozen bank (extra fields that the row chain never reads)
spec.freeze_seed = round_seed;
spec.freeze_frozen_against = 'the paper';
spec.freeze_generated = char(datetime('now'));

outdir = fullfile(resdir, outname);
if ~exist(outdir, 'dir'), mkdir(outdir); end
specfile = fullfile(outdir, 'bank_spec.mat');
if exist(specfile, 'file')
    old = load(specfile);
    assert(old.spec.freeze_seed == round_seed, ...
        'existing %s has different seed -- refusing to mix banks', specfile);
else
    save(specfile, 'spec', 'kL', 'kk', 'kdis', 'fplc', ...
        'fplj', 'Hvs', 'Dvs', 'kbw', 'cfgid', 'outid');
end

chunk = 250;
nchunk = ceil(N/chunk);
if isempty(gcp('nocreate')), parpool('Processes', 3); end   % host RAM sets the limit

for ch = 1:nchunk
    fch = fullfile(outdir, sprintf('chunk_%03d.mat', ch));
    if exist(fch, 'file'), fprintf('chunk %d exists, skip\n', ch); continue; end
    idx = (ch-1)*chunk + 1 : min(ch*chunk, N);
    rows = cell(numel(idx), 1);
    parfor q = 1:numel(idx)
        i = idx(q);
        rows{q} = one_scenario_r18(i + id_off, cfgs{cfgid(i)}, outid(i), kL(i), kk(i), ...
            kdis(i,:), fplc(i), fplj(i,:), Hvs(i,:), Dvs(i,:), kbw(i), spec);
    end
    S = [rows{:}];                                     %#ok<NASGU>
    save(fch, 'S');
    fprintf('chunk %d/%d done (%s)\n', ch, nchunk, datetime('now'));
end
fprintf('round %s complete: %d chunks in %s\n', mode, nchunk, outdir);
end

% ==========================================================================
function do_assemble(here, resdir, triadir)
% The logic of assemble_bank.m, unchanged, pointed at the frozen-bank folders
% and output paths (deviation D3). The class-0 integrity audit is the same.
dirs = {fullfile(resdir, 'bank_r18f'), fullfile(resdir, 'bank_r18f_topup')};
S = [];
for dd2 = 1:numel(dirs)
    dd = dir(fullfile(dirs{dd2}, 'chunk_*.mat'));
    for i = 1:numel(dd)
        d = load(fullfile(dirs{dd2}, dd(i).name));
        Snew = d.S;
        if ~isfield(Snew, 'round'), [Snew.round] = deal(1); end
        if ~isempty(S)
            if ~isfield(S, 'round'), [S.round] = deal(1); end
            assert(isempty(setxor(fieldnames(S), fieldnames(Snew))), ...
                'r18 assemble: field mismatch between rounds');
            Snew = orderfields(Snew, S);
        end
        S = [S, Snew];                                        %#ok<AGROW>
    end
end
assert(~isempty(S), 'no chunks found');
fprintf('collected %d rows; ok = %d, failed = %d\n', ...
    numel(S), nnz([S.ok]), nnz(~[S.ok]));
fails = {S(~[S.ok]).fail};
ledger = {};
if ~isempty(fails)
    [uf, ~, ic] = unique(fails);
    for u = 1:numel(uf)
        fprintf('  fail[%d]: %s\n', nnz(ic == u), uf{u});
        ledger{end+1} = {uf{u}, nnz(ic == u)};                 %#ok<AGROW>
    end
end

ok = S([S.ok]);

% ---- class-0 integrity audit, as in assemble_bank.m ------------------------
cls = [ok.class];  sig = [ok.sigma_dom];  zet = [ok.zeta_min];
bad1 = find((cls == 0) ~= (zet <= 0));
bad2 = find((zet <= 0) ~= (sig >= 0));
fprintf('class-0 audit: label-vs-zeta mismatches %d, zeta-vs-sigma mismatches %d (of %d rows)\n', ...
    numel(bad1), numel(bad2), numel(ok));
assert(isempty(bad1) && isempty(bad2), 'CLASS-0 AUDIT FAILED: %s | %s', ...
    mat2str(bad1(1:min(10,end))), mat2str(bad2(1:min(10,end))));

% ---- flat table, same schema as assemble_bank.m ----------------------------
scal = {'id','round','outage','kL','k','residual_pu','oob_flag','zeta_min','sigma_dom', ...
    'f_dom','class','abstain','lam_min','H','Hbar','PWSGSI','C_SGSI','C_interior', ...
    'C_dc','im_resid','w_c','gSCR','SCR_min','lam_gap','port_resid','Ptot','Qtot', ...
    'kbw','nGFM','oob_stable','nzero','trust_dom'};
T = table();
for f = scal
    T.(f{1}) = [ok.(f{1})]';
end
T.thr_rule = string({ok.thr_rule}');
T.cfgbuses = string({ok.cfgbuses}');
vec = {'Pinv','Qinv','Yeff','Vbus','Yhat','p_part','fpll','Hv','Dv','kdisp'};
for f = vec
    M = vertcat(ok.(f{1}));
    for j = 1:size(M, 2)
        T.(sprintf('%s_%d', f{1}, j)) = M(:, j);
    end
end

datadir = fullfile(here, '..', 'data');
writetable(T, fullfile(datadir, 'dataset_r18freeze.csv'));
save(fullfile(datadir, 'dataset_r18freeze.mat'), 'T');
fprintf('exported %d rows x %d cols -> data/dataset_r18freeze.csv\n', ...
    height(T), width(T));

fprintf('class support: ');
for c = 0:5, fprintf('%d:%d  ', c, nnz(T.class == c)); end
fprintf('| abstain flagged: %d\n', nnz(T.abstain));

% failure ledger file
reasons = strings(0, 1);  counts = zeros(0, 1);
for u = 1:numel(ledger)
    reasons(end+1, 1) = string(ledger{u}{1});                  %#ok<AGROW>
    counts(end+1, 1)  = ledger{u}{2};                          %#ok<AGROW>
end
reasons = [reasons; "TOTAL_sampled"; "TOTAL_converged"];
counts  = [counts;  numel(S);        numel(ok)];
L = table(reasons, counts, 'VariableNames', {'fail_reason','count'});
writetable(L, fullfile(triadir, 'r18_freeze_ledger.csv'));
fprintf('ledger -> r18/r18_freeze_ledger.csv\n');
end

% ==========================================================================
function r = one_scenario_r18(i, gfmbuses, outid, kL, k, kdis, fplc, fplj, Hvs, Dvs, kbw, spec)
% An exact copy of one_scenario in build_bank.m (only the name differs). The
% repro gate above checks the copy against published dataset rows.
P = params.sys_const();
% complete template (identical fields for every row, success or failure)
r = struct('id', i, 'round', 1 + strcmp(spec.mode, 'topup'), ...
    'ok', false, 'fail', '', 'cfgbuses', mat2str(gfmbuses), ...
    'outage', outid, 'kL', kL, 'k', k, ...
    'residual_pu', NaN, 'nzero', NaN, 'oob_stable', NaN, 'trust_dom', NaN, ...
    'oob_flag', NaN, 'zeta_min', NaN, 'sigma_dom', NaN, 'f_dom', NaN, ...
    'class', NaN, 'abstain', NaN, ...
    'lam_min', NaN, 'H', NaN, 'Hbar', NaN, 'PWSGSI', NaN, ...
    'C_SGSI', NaN, 'C_interior', NaN, 'C_dc', NaN, 'thr_rule', '', ...
    'im_resid', NaN, 'w_c', NaN, 'gSCR', NaN, 'SCR_min', NaN, ...
    'lam_gap', NaN, 'port_resid', NaN, ...
    'Pinv', nan(1,10), 'Qinv', nan(1,10), 'Yeff', nan(1,10), ...
    'Vbus', nan(1,10), 'Yhat', nan(1,10), 'Ptot', NaN, 'Qtot', NaN, ...
    'p_part', nan(1,10), 'fpll', nan(1,10), 'Hv', nan(1,10), 'Dv', nan(1,10), ...
    'kbw', kbw, 'kdisp', kdis, 'nGFM', numel(gfmbuses));
try
    devtype = repmat({'GFL'}, 1, 10);
    buses = 30:39;
    gslot = 0;
    tune = repmat(struct('f_pll', [], 'Hv', [], 'Dv', [], 'kbw', kbw), 10, 1);
    fpll_i = zeros(1,10); Hv_i = zeros(1,10); Dv_i = zeros(1,10);
    for b = 1:10
        if ismember(buses(b), gfmbuses)
            devtype{b} = 'VSM';
            gslot = gslot + 1;
            tune(b).Hv = Hvs(min(gslot, 4));
            tune(b).Dv = Dvs(min(gslot, 4));
            Hv_i(b) = tune(b).Hv;  Dv_i(b) = tune(b).Dv;
        else
            tune(b).f_pll = min(max(fplc*fplj(b), 6), 45);
            fpll_i(b) = tune(b).f_pll;
        end
    end
    cfg = struct('devtype', {devtype}, 'kL', kL, 'k', k, 'kdisp', kdis, 'tune', tune);
    if outid > 0, cfg.outage = outid; end

    o = run_scenario(cfg);
    if ~isempty(o.fail), r.fail = o.fail; return; end
    % The sampled tuning must reach the analytic chain. The threshold sweep
    % depends on f_pll/Hv/Dv/kbw: the s = 0 block provably does not depend on
    % the tuning, but the interior realness crossings do.
    R = reduced.chain(o.mpc, o.res, o.plant, P, tune);

    % ---- gates ----
    r.residual_pu = o.residual_pu;
    r.nzero = o.S.nzero;
    r.oob_stable = o.S.oob_stable;
    r.trust_dom = o.S.trust_dom;
    if o.S.nzero ~= 1, r.fail = 'zero_mode_count'; return; end

    % ---- truth ----
    zeta = o.S.zeta_min;
    sig  = o.S.sigma_dom;
    if ~o.S.oob_stable
        % out-of-band instability: flagged, treated as Unstable, reported
        zeta = min(zeta, -1e-6);
        sig = max(sig, o.S.oob_maxRe);
        r.oob_flag = 1;
    else
        r.oob_flag = 0;
    end
    r.zeta_min = zeta;
    r.sigma_dom = sig;
    r.f_dom = o.S.f_dom;
    edges = spec.bands;
    r.class = sum(zeta >= edges);          % 0..5 by the band edges, 0 when zeta <= 0 (next line)
    if zeta <= 0, r.class = 0; end
    % abstain marks a real band-edge ambiguity of the in-band zeta. An
    % out-of-band instability is always Unstable: the -1e-6 zeta only forces
    % the label and is not a measurement near a boundary.
    if o.S.oob_stable
        r.abstain = min(abs(zeta - edges)) < spec.delta_abst;
    else
        r.abstain = false;
    end

    % ---- indices ----
    r.lam_min = R.lam_min;  r.H = R.H;  r.Hbar = R.Hbar;  r.PWSGSI = R.PWSGSI;
    r.C_SGSI = R.C_SGSI;  r.C_interior = R.C_interior;  r.C_dc = R.C_dc;
    r.thr_rule = R.thr_rule;  r.im_resid = R.im_resid;  r.w_c = R.w_c;
    r.gSCR = R.gSCR;  r.SCR_min = R.SCR_min;
    r.lam_gap = R.lam_gap;  r.port_resid = R.port_resid;

    % ---- features ----
    define_constants;
    Pg = o.res.gen(:, PG);  Qg = o.res.gen(:, QG);
    Vm = o.res.bus(ismember(o.res.bus(:, BUS_I), o.plant.bus), VM);
    Pconv = Pg ./ o.plant.Sn;              % converter-base pu
    Qconv = Qg ./ o.plant.Sn;
    Sapp  = sqrt(Pg.^2 + Qg.^2) ./ o.plant.Sn;
    r.Pinv = Pconv(:).';  r.Qinv = Qconv(:).';
    r.Yeff = R.Yeff(:).';
    r.Vbus = Vm(:).';
    r.Yhat = (Sapp(:)./Vm(:).^2).';        % measurable surrogate |I|/|V|, converter pu
    r.Ptot = sum(Pg)/100;  r.Qtot = sum(Qg)/100;
    r.p_part = R.p(:).';

    % ---- design record ----
    r.fpll = fpll_i;  r.Hv = Hv_i;  r.Dv = Dv_i;  r.kbw = kbw;
    r.kdisp = kdis;
    r.nGFM = numel(gfmbuses);
    r.ok = true;
catch e
    r.fail = ['error: ' e.message];
end
end

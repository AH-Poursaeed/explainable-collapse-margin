function build_bank(N, outname, mode)
%BUILD_BANK  Scenario bank generator, deterministic under the master seed.
%   Latin hypercube over the continuous axes; stratified named GFM
%   configurations; N-1 outages from a fixed pool. Every scenario
%   runs power flow -> full model -> reduced chain -> features -> label.
%   Failures are logged with reasons, never silently dropped.
%
%   mode = 'main' (default) or 'topup'. The top-up is the documented second
%   bank round near the boundary, set off by the adaptive class-support
%   rule. The reason: the marginal shell zeta in (0, 5%) is thin under the
%   main LHS. After round one, the Very Weak/Weak/Moderate bands held
%   0.2/1.4/2.1 percent of the rows. The top-up biases k into [1.2, 2.3]
%   (the part above 1.9 belongs to the documented top-up range), kL into
%   [0.9, 1.2] and Dv into [40, 100]. It also gives more weight to the 1-3
%   GFM families whose marginal crossings lie inside the range. The schema
%   records this; top-up rows carry round = 2 and ids offset by 10000.
%
%   Chunked parfor with per-chunk checkpoints in results/<outname>/;
%   re-runnable (existing chunks are skipped). assemble_bank() merges.

if nargin < 1, N = 6000; end
if nargin < 2, outname = 'bank'; end
if nargin < 3, mode = 'main'; end
P = params.sys_const();
topup = strcmp(mode, 'topup');
% a top-up round must not point at the main round's directory: the chunk
% checkpointing would silently skip every chunk and the spec file would be
% overwritten with topup metadata.
assert(~topup || ~strcmp(outname, 'bank'), 'build_bank:outname', ...
    'topup mode requires a distinct outname (e.g. ''bank_topup'')');
if topup
    rng(P.seed + 1, 'twister');       % distinct deterministic stream
else
    rng(P.seed, 'twister');
end

% ---- named GFM configurations (buses) and sampling weights ------------------
cfgs = { [39], [38], [38 39], [35 39], [34 39], [35 38 39], [33 38 39], [33 35 38 39] };
if topup
    cfgp = [0.28  0.14  0.22   0.10     0.05     0.11        0.05        0.05];
else
    cfgp = [0.12  0.06  0.24   0.10     0.08     0.16        0.10        0.14];
end
% ---- outage pool: 16 deterministic picks from the non-islanding 35 ----------
n1 = load(fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'phase1_n1.mat'));
rng(P.seed, 'twister');                          % pool ALWAYS from the master seed
pool = n1.pool(randperm(numel(n1.pool), 16));
ood_branches = pool(13:16);                      % reserved as OOD topologies
if topup, rng(P.seed + 1, 'twister'); end

% ---- LHS over continuous axes ------------------------------------------------
% dims: 1 kL | 2 k | 3-12 kdisp | 13 fpll_common | 14-23 fpll_jitter |
%       24-27 Hv (per GFM slot) | 28-31 Dv | 32 kbw
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
% discrete draws (deterministic under the round's stream)
cfgid  = randsample(numel(cfgs), N, true, cfgp);
hasout = rand(N,1) < 0.5;
outid  = pool(randi(16, N, 1)).';
outid(~hasout) = 0;
id_off = 0; if topup, id_off = 10000; end

spec = struct('N', N, 'cfgs', {cfgs}, 'cfgp', cfgp, 'pool', pool, ...
    'ood_branches', ood_branches, 'seed', P.seed, 'mode', mode, ...
    'ranges', struct('kL', [min(kL) max(kL)], 'k', [min(kk) max(kk)], ...
    'kdisp', [0.75 1.25], 'fpll', [8 35], 'fpll_jit', [0.8 1.2], ...
    'Hv', [2 10], 'Dv', [min(Dvs(:)) max(Dvs(:))], 'kbw', [0.7 1.4]), ...
    'delta_abst', 0.0015, 'bands', [0 0.01 0.03 0.05 0.10]);

outdir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', outname);
if ~exist(outdir, 'dir'), mkdir(outdir); end
save(fullfile(outdir, 'bank_spec.mat'), 'spec', 'kL', 'kk', 'kdis', 'fplc', ...
    'fplj', 'Hvs', 'Dvs', 'kbw', 'cfgid', 'outid');

chunk = 250;
nchunk = ceil(N/chunk);
if isempty(gcp('nocreate')), parpool('Processes', 4); end

for ch = 1:nchunk
    fch = fullfile(outdir, sprintf('chunk_%03d.mat', ch));
    if exist(fch, 'file'), fprintf('chunk %d exists, skip\n', ch); continue; end
    idx = (ch-1)*chunk + 1 : min(ch*chunk, N);
    rows = cell(numel(idx), 1);
    parfor q = 1:numel(idx)
        i = idx(q);
        rows{q} = one_scenario(i + id_off, cfgs{cfgid(i)}, outid(i), kL(i), kk(i), ...
            kdis(i,:), fplc(i), fplj(i,:), Hvs(i,:), Dvs(i,:), kbw(i), spec);
    end
    S = [rows{:}];                                     %#ok<NASGU>
    save(fch, 'S');
    fprintf('chunk %d/%d done (%s)\n', ch, nchunk, datetime('now'));
end
fprintf('bank complete: %d chunks in %s\n', nchunk, outdir);
end

% ==========================================================================
function r = one_scenario(i, gfmbuses, outid, kL, k, kdis, fplc, fplj, Hvs, Dvs, kbw, spec)
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
    % the SAMPLED tuning must reach the analytic chain: the threshold sweep
    % depends on f_pll/Hv/Dv/kbw. The s = 0 block provably does not depend on
    % the tune, but the interior realness crossings do.
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
    r.class = sum(zeta >= edges);          % 0..5 by the band edges (zeta <= 0 -> 0, next line)
    if zeta <= 0, r.class = 0; end
    % abstain marks a true band-edge ambiguity of the in-band zeta. An
    % out-of-band instability is a DEFINITIVE Unstable: the -1e-6 zeta is a
    % forcing device, not a measurement near the boundary.
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
    r.Yhat = (Sapp(:)./Vm(:).^2).';        % measurable surrogate |I|/|V| conv pu
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

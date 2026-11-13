function r17_rays
%r17_rays  Find the collapse point k* by scan + bisection along NRAY stress rays.
%   The ray configurations are drawn from the bank sampling distributions
%   (main-round ranges). The index chain (GRIP, lam_min, gSCR, SCR_min) is
%   evaluated just on the stable side of each boundary. This turns the
%   printed 3-ray boundary band into a distribution with a stated n.
%
%   Steps for each ray: fix (config, kL, kdisp, tune, outage). March the
%   line-impedance scale k upward from 0.6 in 0.1 steps (the PF is warm
%   started from the previous point on the ray) until the band-limited
%   growth rate crosses zero, then bisect to 1e-3. If the ray is already
%   unstable at k = 0.6, march down to 0.2 and bisect (mode crossing_down).
%   The chain is evaluated at k* - 0.005 with the ray's sampled tune,
%   because the interior threshold depends on the tune.
%
%   Deterministic under P.seed + 40 (the ray stream, recorded in the spec
%   mat). Checkpointed per row: each ray adds one CSV line as soon as it
%   finishes, and a re-run skips the rays already in the CSV.
%
%   Output: results/tables/r17/r17_rays.csv (+ r17_rays_spec.mat)

maxNumCompThreads(2);
NRAY  = 30;
TBOX  = 165*60;                 % s, clean exit (3 h limit minus a margin)
t00 = tic;

P = params.sys_const();
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'tables', 'r17');
if ~exist(outdir, 'dir'), mkdir(outdir); end
fout = fullfile(outdir, 'r17_rays.csv');
fspec = fullfile(outdir, 'r17_rays_spec.mat');

% ---- ray designs: bank main-round distributions, own random stream --------
cfgs = { [39], [38], [38 39], [35 39], [34 39], [35 38 39], [33 38 39], [33 35 38 39] };
cfgp = [0.12  0.06  0.24   0.10     0.08     0.16        0.10        0.14];
n1 = load(fullfile(here, '..', 'results', 'phase1_n1.mat'));
rng(P.seed, 'twister');                      % the pool always uses the master seed
pool = n1.pool(randperm(numel(n1.pool), 16));

rng(P.seed + 40, 'twister');                 % the ray stream
kL_r   = 0.6  + rand(NRAY,1)*0.6;            % [0.6, 1.2]
kdis_r = 0.75 + rand(NRAY,10)*0.5;           % [0.75, 1.25]
fplc_r = 8    + rand(NRAY,1)*27;             % [8, 35]
fplj_r = 0.8  + rand(NRAY,10)*0.4;           % [0.8, 1.2]
Hv_r   = 2    + rand(NRAY,4)*8;              % [2, 10]
Dv_r   = 40   + rand(NRAY,4)*160;            % [40, 200]
kbw_r  = 0.7  + rand(NRAY,1)*0.7;            % [0.7, 1.4]
cfgid_r  = randsample(numel(cfgs), NRAY, true, cfgp);
hasout_r = rand(NRAY,1) < 0.5;
outid_r  = pool(randi(16, NRAY, 1)).';
outid_r(~hasout_r) = 0;
save(fspec, 'NRAY', 'kL_r', 'kdis_r', 'fplc_r', 'fplj_r', 'Hv_r', 'Dv_r', ...
     'kbw_r', 'cfgid_r', 'outid_r', 'cfgs', 'cfgp', 'pool');

done = [];
if isfile(fout)
    T0 = readtable(fout);
    done = T0.rayid;
    fid = fopen(fout, 'a');
else
    fid = fopen(fout, 'w');
    fprintf(fid, ['rayid,cfgbuses,nGFM,outage,kL,kbw,fplc,mode,kstar,', ...
        'k_last_ok,n_evals,wall_s,k_probe,sigma_probe,GRIP_star,', ...
        'lam_min_star,gSCR_star,SCR_min_star,Hbar_star\n']);
end

smoke = ~isempty(getenv('SMOKE_RUN'));
nray_run = NRAY;  if smoke, nray_run = 1; end

for ray = 1:nray_run
    if any(done == ray), fprintf('ray %d done, skip\n', ray); continue; end
    if toc(t00) > TBOX
        fprintf('Time limit reached at ray %d (%.0f min elapsed) -- exiting cleanly\n', ...
            ray, toc(t00)/60);
        break
    end
    tR = tic;
    gfmb = cfgs{cfgid_r(ray)};
    cfg = mkcfg(gfmb, outid_r(ray), kL_r(ray), kdis_r(ray,:), fplc_r(ray), ...
                fplj_r(ray,:), Hv_r(ray,:), Dv_r(ray,:), kbw_r(ray));
    fprintf('=== ray %d: cfg %s outage %d kL %.3f ===\n', ray, ...
        mat2str(gfmb), outid_r(ray), kL_r(ray));
    st = trace_ray(cfg, P);
    % chain probe just on the stable side
    probe = struct('k', NaN, 'sigma', NaN, 'GRIP', NaN, 'lam_min', NaN, ...
                   'gSCR', NaN, 'SCR_min', NaN, 'Hbar', NaN);
    if strncmp(st.mode, 'crossing', 8)
        probe = chain_probe(cfg, st.kstar - 0.005, P);
        if ~probe.ok && st.kstar - 0.010 > 0.2
            probe = chain_probe(cfg, st.kstar - 0.010, P);   % one step deeper
        end
    end
    fprintf(fid, '%d,"%s",%d,%d,%.6f,%.6f,%.4f,%s,%.6f,%.6f,%d,%.1f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
        ray, mat2str(gfmb), numel(gfmb), outid_r(ray), kL_r(ray), ...
        kbw_r(ray), fplc_r(ray), st.mode, st.kstar, st.k_last_ok, ...
        st.n_evals, toc(tR), probe.k, probe.sigma, probe.GRIP, ...
        probe.lam_min, probe.gSCR, probe.SCR_min, probe.Hbar);
    fprintf('ray %d: %s kstar=%.4f GRIP*=%.4f lam_min*=%.4f (%d evals, %.0f s)\n', ...
        ray, st.mode, st.kstar, probe.GRIP, probe.lam_min, st.n_evals, toc(tR));
end
fclose(fid);
fprintf('r17_rays complete (%.1f min).\n', toc(t00)/60);
end

% ===========================================================================
function cfg = mkcfg(gfmb, outid, kL, kdis, fplc, fplj, Hvs, Dvs, kbw)
%MKCFG  Build the tune as one_scenario of the bank does (same slot logic).
devtype = repmat({'GFL'}, 1, 10);
buses = 30:39;
gslot = 0;
tune = repmat(struct('f_pll', [], 'Hv', [], 'Dv', [], 'kbw', kbw), 10, 1);
for b = 1:10
    if ismember(buses(b), gfmb)
        devtype{b} = 'VSM';
        gslot = gslot + 1;
        tune(b).Hv = Hvs(min(gslot, 4));
        tune(b).Dv = Dvs(min(gslot, 4));
    else
        tune(b).f_pll = min(max(fplc*fplj(b), 6), 45);
    end
end
cfg = struct('devtype', {devtype}, 'kL', kL, 'kdisp', kdis, 'tune', tune);
if outid > 0, cfg.outage = outid; end
end

% ---------------------------------------------------------------------------
function st = trace_ray(cfg, P)
%TRACE_RAY  Scan + bisect the sigma = 0 crossing on the k axis, with warm starts.
st = struct('mode', 'unresolved', 'kstar', NaN, 'k_last_ok', NaN, 'n_evals', 0);
wsD = struct('res', []);                     % carries the warm start (updated at each call)
K0 = 0.6;  KMAX = 2.6;  KMIN = 0.2;  DK = 0.1;

[s0, ok0, wsD] = eval_k(cfg, K0, P, wsD);  st.n_evals = st.n_evals + 1;
if ~ok0
    st.mode = 'pf_infeasible_at_k0';
    return
end
if s0 < 0
    % march up until a crossing, a PF failure or KMAX
    klo = K0;  khi = NaN;
    k = K0 + DK;
    while k <= KMAX + 1e-12
        [s, ok, wsD] = eval_k(cfg, k, P, wsD);  st.n_evals = st.n_evals + 1;
        fprintf('  scan k=%.3f sigma=%s ok=%d\n', k, fmt(s), ok);
        if ~ok
            st.mode = 'pf_infeasible_first';  st.k_last_ok = klo;
            % The infeasibility edge is not bisected as a proxy for k*.
            % It is recorded as its own mode, with no crossing row.
            return
        end
        if s >= 0, khi = k; break; end
        klo = k;
        k = k + DK;
    end
    if isnan(khi)
        st.mode = 'no_crossing_le_kmax';  st.k_last_ok = klo;
        return
    end
    dirmode = 'crossing';
else
    % already unstable at K0: march down to find the stable side
    khi = K0;
    k = K0 - DK;  klo = NaN;
    while k >= KMIN - 1e-12
        [s, ok, wsD] = eval_k(cfg, k, P, wsD);  st.n_evals = st.n_evals + 1;
        fprintf('  scan-down k=%.3f sigma=%s ok=%d\n', k, fmt(s), ok);
        if ~ok
            st.mode = 'pf_infeasible_down';  st.k_last_ok = NaN;
            return
        end
        if s < 0, klo = k; break; end
        khi = k;
        k = k - DK;
    end
    if isnan(klo)
        st.mode = 'unstable_to_kmin';  st.k_last_ok = NaN;
        return
    end
    dirmode = 'crossing_down';
end

% bisect [klo, khi] to 1e-3 (klo stable, khi unstable)
while khi - klo > 1e-3
    km = 0.5*(klo + khi);
    [s, ok, wsD] = eval_k(cfg, km, P, wsD);  st.n_evals = st.n_evals + 1;
    if ~ok
        st.mode = 'pf_infeasible_in_bracket';  st.k_last_ok = klo;
        return
    end
    if s >= 0, khi = km; else, klo = km; end
end
st.mode = dirmode;
st.kstar = 0.5*(klo + khi);
st.k_last_ok = klo;
end

function s = fmt(v)
if isnan(v), s = 'NaN'; else, s = sprintf('%+.4f', v); end
end

% ---------------------------------------------------------------------------
function [sig, ok, wsD] = eval_k(cfg, k, P, wsD)
%EVAL_K  One point on the ray: PF (warm started when possible) -> full model
%   -> init + polish (same steps as run_scenario) -> band-limited truth with
%   the bank's oob convention. A failed evaluation returns ok = false.
sig = NaN;  ok = false;
try
    c = cfg;  c.k = k;
    [mpc, plant] = build_case(c, P);
    mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1, ...
                     'pf.tol', 1e-12, 'pf.nr.max_it', 40);
    res = [];
    if ~isempty(wsD.res)
        mpcw = mpc;
        mpcw.bus(:, 8) = wsD.res.bus(:, 8);      % VM
        mpcw.bus(:, 9) = wsD.res.bus(:, 9);      % VA
        try
            r1 = runpf(mpcw, mpopt);
            if r1.success == 1, res = r1; end
        catch
        end
    end
    if isempty(res)
        r1 = runpf(mpc, mpopt);                  % cold fallback
        if r1.success ~= 1, return; end
        res = r1;
    end
    tune = c.tune;
    M = full.build_model(mpc, res, plant, P, tune);
    [x0, M, ~] = full.init_state(M, mpc, res);
    L = full.linearize(M, x0, 'cs');
    resid = norm(full.rhs(x0, M), inf);  it = 0;
    while resid > 1e-13 && it < 4
        x0 = x0 - lsqminnorm(L.Asys, full.rhs(x0, M));
        resid = norm(full.rhs(x0, M), inf);  it = it + 1;
    end
    if it > 0, L = full.linearize(M, x0, 'cs'); end
    residual_pu = norm(full.rhs(x0, M)./M.rowscale, inf);
    if ~(resid < P.tol.resid && residual_pu < 1e-12), return; end
    S = full.modal(L.Asys, P);
    if S.nzero ~= 1, return; end
    sig = S.sigma_dom;
    if ~S.oob_stable, sig = max(sig, S.oob_maxRe); end
    ok = true;
    wsD.res = res;                                % carry the warm start
catch
end
end

% ---------------------------------------------------------------------------
function pr = chain_probe(cfg, kprobe, P)
%CHAIN_PROBE  Index chain at the stable-side point, with the ray's tune.
pr = struct('ok', false, 'k', kprobe, 'sigma', NaN, 'GRIP', NaN, ...
            'lam_min', NaN, 'gSCR', NaN, 'SCR_min', NaN, 'Hbar', NaN);
try
    c = cfg;  c.k = kprobe;
    o = run_scenario(c);
    if ~isempty(o.fail), return; end
    R = reduced.chain(o.mpc, o.res, o.plant, P, c.tune);
    pr.sigma = o.S.sigma_dom;
    pr.GRIP = R.PWSGSI;  pr.lam_min = R.lam_min;
    pr.gSCR = R.gSCR;  pr.SCR_min = R.SCR_min;  pr.Hbar = R.Hbar;
    pr.ok = true;
catch
end
end

function e6_timing
%E6_TIMING  Screening-layer economics, MATLAB side.
%   Median wall time per operating point for each analysis stage, from 10
%   repetitions on the design case (k = 1.2, kL = 1.0, the mid-stress
%   point). The computational threads are capped at 2 to compare with the
%   batch environment.
%   Stages: power flow; GRIP index chain (reduced.chain); lossy multiport
%   pencil + eig; full model build + init + linearize (CS) + eig. The
%   EMT time per case is the median wall time of the e4 security map.
%   Output: results/final/tables/e6_timing_matlab.csv

maxNumCompThreads(2);
P = params.sys_const();
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'final', 'tables');
if ~exist(outdir, 'dir'), mkdir(outdir); end
dv = repmat({'GFL'}, 1, 10);  dv{9} = 'VSM';  dv{10} = 'VSM';
cfg = struct('devtype', {dv}, 'kL', 1.0, 'k', 1.2);
NREP = 10;

t_pf = zeros(NREP, 1); t_chain = t_pf; t_pencil = t_pf; t_full = t_pf;
t_idx = t_pf;
% device stiffening identified once (static device data); the per-point
% index cost is the susceptance rebuild + Schur + 10x10 eig
[mpc0, plant0] = build_case(cfg, P);
mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1, ...
    'pf.tol', 1e-12);
res0 = runpf(mpc0, mpopt);
R0 = reduced.chain(mpc0, res0, plant0, P);
bin_conv = diag(R0.BB.Bin_sys) ./ R0.BB.Sb;
for q = 1:NREP
    [mpc, plant] = build_case(cfg, P);
    tic; res = runpf(mpc, mpopt); t_pf(q) = toc;
    assert(res.success == 1);
    tic;
    BB = reduced.build_B(mpc, res, plant, bin_conv);
    pp = max(BB.p, 1e-300);
    Hn = -sum(pp.*log(pp))/log(numel(pp));
    grip = BB.lam_min * Hn;                                     %#ok<NASGU>
    t_idx(q) = toc;
    tic; R = reduced.chain(mpc, res, plant, P); t_chain(q) = toc;
    tic; PL = reduced.pencil_lossy(R, P, mpc, res, plant); t_pencil(q) = toc; %#ok<NASGU>
    tic;
    M = full.build_model(mpc, res, plant, P, repmat(struct(), 10, 1));
    [x0, M, ~] = full.init_state(M, mpc, res);
    L = full.linearize(M, x0, 'cs');
    x0 = x0 - lsqminnorm(L.Asys, full.rhs(x0, M));
    L = full.linearize(M, x0, 'cs');
    S = full.modal(L.Asys, P);                                  %#ok<NASGU>
    t_full(q) = toc;
end

% EMT per case from the security map (median wall of the 8 s cells)
emt_med = NaN;
try
    Tm = readtable(fullfile(here, '..', 'results', 'final', ...
        'e4_security_map.csv'));
    emt_med = median(Tm.wall_s);
catch
end

rows = {
 'power flow (MATPOWER, Q-lims)', median(t_pf);
 'GRIP index (susceptance rebuild + Schur + 10x10 eig; device data static)', median(t_idx);
 'full index chain incl. analytic thresholds (reduced.chain)', median(t_chain);
 'lossy multiport pencil + eig', median(t_pencil);
 'full model: build + init + linearise (CS) + eig (348 states)', median(t_full);
 'nonlinear EMT, 8 s window (ETDRK2, 50 us)', emt_med;
};
T = cell2table(rows, 'VariableNames', {'stage', 'median_wall_s'});
writetable(T, fullfile(outdir, 'e6_timing_matlab.csv'));
disp(T);
fprintf('e6_timing_matlab.csv written\n');
end

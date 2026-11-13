function e9_ma2024_threshold(outdir, job, r_first, r_last, peer_csv, stage)
%E9_MA2024_THRESHOLD  The stability threshold CgSCR of Ma et al. (2024) on the reduced chain.
%   The script computes CgSCR (eq. 40 of the source) on this project's reduced
%   chain, row by row. On every row it also traces the chain's own threshold
%   C_SGSI against the bank.
%
%   Source: F. Ma, H. Xin, D. Wu, Y. Liu, X. Chen, "Assessing small-signal
%   grid strength of 100% inverter-based power systems", IEEE Trans. Power
%   Del., vol. 39, no. 5, 2024 (preprint doi 10.36227/techrxiv.21120517.v8,
%   eqs. 12-41, Appendices B and C).
%
%   Per scenario
%   1. The case and the power flow: bank.row2cfg, build_case and runpf, with
%      the options of r2_regen_thresholds.m (the script that wrote the bank's
%      threshold columns). Then reduced.chain with the scenario's tune gives
%      the ports of all plants, B_in, B_kron, the stiffness matrix A and the
%      chain's own threshold C_SGSI (for the trace of the conventions).
%   2. The network as the source publishes it. The load susceptance is taken
%      out of B_lap, because the source's loads play no role in its
%      small-signal model (printed p. 4). Then a Kron reduction onto the plant
%      buses, as in e9_ma2024_index.m. S = sym(B + Bin_sys). B1 is the GFL
%      block, B2 the GFL x GFM block and B4t = B4 + B_in the GFM block.
%   3. The index (eqs. 33, 39): Y_eq = S_b^-1 (B1 - B2 B4t^-1 B2'). gSCR is
%      the smallest eigenvalue of the symmetric form
%      S_b^-1/2 (B1 - B2 B4t^-1 B2') S_b^-1/2, and phi is its unit
%      eigenvector. Then p_1i = phi_i^2 (eq. 34). u1 = S_b^-1/2 phi and
%      v1 = S_b^1/2 phi are the right and the left eigenvectors of Y_eq
%      (v1' u1 = 1).
%   4. The GFM perturbation (eqs. 30, 31, 35) at s:
%        D(s)     = blkdiag_j { S_bm,j F(s)^-1 Y_delta,j(s) }      2m x 2m
%        K        = kron(B4t^-1, I2)
%        Delta(s) = -(I + K D(s))^-1 K D(s) K
%        N(s)     = -kron(S_b^-1 B2, I2) Delta(s) kron(B2', I2)    2n x 2n
%        delta(s) = kron(v1, I2)' N(s) kron(u1, I2)                2 x 2
%   5. The threshold (eq. 40): G(s) = sum_i p_1i F(s)^-1 Y_GFL,i(s) - delta(s).
%      CgSCR is the real positive eigenvalue of G(j w_c). reduced.threshold
%      finds it without change when its input is F(jw) G(jw): its F^-1 Y_agg
%      is then G, and this includes its exact static root at w = 0. The
%      script keeps the family that binds and w_c. The rule (eq. 41): stable
%      if and only if gSCR >= CgSCR. beta = (gSCR - CgSCR)/CgSCR.
%
%   Conventions (a gate checks each one, none is assumed)
%     ports    reduced.port: common frame, injected current (dI = Y dV),
%              converter base. The source's Y_GFL,i is the same object
%              (eq. 17: global frame, injected, the inverter's own base).
%     network  I = kron(B, F(s)) V (eq. 21); reduced.Fmat is eq. 14.
%     Y_delta  The source's virtual apparatus is an ABSORBED admittance:
%              -Y_GFM = Y_dq + Y_delta (eq. 20) with Y_dq = B_in F(s)
%              (eq. 15). So eq. 27 carries +S_bm Y_delta on the GFM diagonal.
%              reduced.Ydelta is the INJECTED increment of the same loop
%              (its Ydq is -F/(w0 Leff)). The project's reduced
%              characteristic (reduced.dominant_root) carries
%              -S_bm R Ydelta R' there. So Y_delta(source) = -R Ydelta R',
%              with R = rot2(th0) as in chain.m's yagg_s. The map gate
%              checks that eq. 27, built this way, is the project's reduced
%              characteristic. map_rel_flip is the residual with the other
%              sign.
%   ASSUMPTIONS
%     - The network as published keeps the line charging and the declared
%       bus shunt capacitance. The pages of the source that were read do not
%       settle this.
%     - The project's GFL port is the GFL admittance of the bank's plants.
%     - Y_delta is the project's closed form with the s = 0 Thevenin fit
%       B_in = 1/(w0 Leff). It is not derived again from the source's eq. 45.
%
%   Gates computed per row
%     trace    lam_min, classical gSCR, GRIP and the chain's C_SGSI,
%              C_interior, C_dc, w_c and rule, beside the bank's values
%     agg_id   this script's aggregation with the chain's weights against
%              the chain's own yagg_s, at s = 0 and on the imaginary axis
%     index    the eigenvalue of the eigenvector form against the expression
%              of e9_ma2024_index.m, the sum of p_1i, v1' u1 and the
%              eigen-residuals
%     limit    delta(0) from the exact s = 0 limit against delta(1e-6 j)
%     schur    M(s) + N(s) (eqs. 28-31) against the Schur complement of the
%              eq. 27 matrix onto the GFL block
%     proj     G(s) against lambda_1 I - kron(v1, I2)' (M + N) kron(u1, I2)
%     map      eq. 27 with Y_delta = -R Ydelta R' against the reduced
%              characteristic of reduced.dominant_root (loads absorbed)
%   Diagnostics (reported, not gates)
%     device   R Ydelta(s) R' against the device's own synchronization
%              increment, the VSM port with the swing live minus the frozen
%              port: cosine at s = 0 and at 1 Hz, relative error at s = 0
%     limit9   the limit gate repeated at s = 1e-9 j (a first-order
%              difference falls a thousandfold)
%     cg_dsdl  Re(ds/dlambda) of the binding root of det(lambda I - G(s)) =
%              0: negative when a larger gSCR moves that root into the left
%              half plane, the orientation that the rule of eq. 41 assumes
%     variants on 'bank' rows of stage 'full': CgSCR with the other sign of
%              Y_delta, and with delta = 0 (the nominal subsystem, eq. 34)
%
%   Usage
%     e9_ma2024_threshold(outdir, frame, r_first, r_last, peer_csv, stage)
%     e9_ma2024_threshold(outdir, 'rays', [], [], '', stage)
%
%   frame is 'bank' (every row of data/dataset.csv) or 'test' (the rows of
%   the test frame, results/tables/r13/r13_tier_battery_rows.csv). The rows
%   run in file order from r_first to r_last. r_first > r_last runs
%   downward. peer_csv is the other worker's rows file. Its rows are skipped,
%   and the script reads it again at most every 5 s. stage is one of:
%     'conv'  the trace only, and the chain's threshold rerun through this
%             script's aggregation (C_mine)
%     'full'  the trace and the threshold of the source, as-published network
%     'twin'  as 'full' on the loads-absorbed network (a sensitivity variant)
%   'rays' walks the design ray of fs3_boundary.m and the bank rays of
%   r20_straddle2.m (specifications of r17_rays.m) along the line-impedance
%   scale k. On each ray it runs a probe at the artifact's stable-side point
%   k* - 0.005, then a grid k = 0.20:0.05:3.00 up to its first failed point.
%   Then it bisects every sign change of gSCR - CgSCR to 1e-5 in k.
%   Writes only under outdir: e9thr_<frame>_<stage>_<r_first>_<r_last>.csv
%   or e9thr_rays_<stage>.csv. Per row or point, the script appends one line
%   and closes the file. A rerun resumes. A last line cut by a kill is
%   dropped. The script refuses a file with another header, a malformed
%   line, another stage, or rows of another version of this script.

here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..');
if nargin < 1 || isempty(outdir), outdir = fullfile(tempdir, 'e9_ma2024_threshold'); end
if nargin < 2 || isempty(job), job = 'bank'; end
if nargin < 3, r_first = []; end
if nargin < 4, r_last = []; end
if nargin < 5 || isempty(peer_csv), peer_csv = ''; end
if nargin < 6 || isempty(stage), stage = 'full'; end
assert(any(strcmp(job, {'bank', 'test', 'rays'})), 'e9thr:job', 'unknown job %s', job);
assert(any(strcmp(stage, {'conv', 'full', 'twin'})), 'e9thr:stage', 'unknown stage %s', stage);
if ~isfolder(outdir), mkdir(outdir); end

maxNumCompThreads(2);
P = params.sys_const();
code_id = code_hash([mfilename('fullpath') '.m']);
mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1, 'pf.tol', 1e-12);
fprintf('e9_ma2024_threshold: job %s, stage %s, code_id %s, MATLAB %s\n', job, stage, ...
    code_id, version);
if strcmp(job, 'rays')
    assert(~strcmp(stage, 'conv'), 'e9thr:stage', 'the rays run stage full or twin');
    run_rays(outdir, stage, root, P, mpopt, code_id);
else
    run_rows(outdir, job, r_first, r_last, peer_csv, stage, root, P, mpopt, code_id);
end
end

% ---------------------------------------------------------------------------
function run_rows(outdir, frame_name, r_first, r_last, peer_csv, stage, root, P, mpopt, code_id)
% The rows of a frame in file order (or downward), checkpointed per row.
Tb = readtable(fullfile(root, 'data', 'dataset.csv'));
Sbank = table2struct(Tb);   % row access on the wide table is slow
ids = [Sbank.id];
assert(all(ids >= 1 & ids == round(ids)) && numel(unique(ids)) == numel(ids), ...
    'e9thr:ids', 'bank ids must be distinct positive integers');
if strcmp(frame_name, 'test')
    tf = readtable(test_frame_file(root));
    [isin, loc] = ismember(tf.id, ids);
    assert(all(isin), 'e9thr:frame', '%d test-frame ids are not in the bank', nnz(~isin));
    frame = Sbank(loc);
else
    frame = Sbank;
end
nrow = numel(frame);
if isempty(r_first), r_first = 1; end
if isempty(r_last), r_last = nrow; end
assert(all([r_first r_last] >= 1 & [r_first r_last] <= nrow) && ...
    r_first == round(r_first) && r_last == round(r_last), 'e9thr:range', ...
    'row range %g..%g is outside 1..%d', r_first, r_last, nrow);
cols = columns();
outcsv = fullfile(outdir, sprintf('e9thr_%s_%s_%d_%d.csv', frame_name, stage, r_first, r_last));
doneOwn = false(max(ids), 1);
doneOwn(resume_own(outcsv, cols, code_id, stage)) = true;
donePeer = false(max(ids), 1);
fprintf('frame %s: %d rows, range %d..%d; rows file %s (%d rows already in it)\n', ...
    frame_name, nrow, r_first, r_last, outcsv, nnz(doneOwn));
if ~isempty(peer_csv), fprintf('peer file %s\n', peer_csv); end

with_var = strcmp(frame_name, 'bank') && strcmp(stage, 'full');
step = 1;
if r_last < r_first, step = -1; end
order = r_first:step:r_last;
PEER_REFRESH_S = 5;
tPeer = -inf;
[ncomp, nown, npeer, nfail] = deal(0);
t0 = tic;
for k = 1:numel(order)
    row = frame(order(k));
    if doneOwn(row.id)
        nown = nown + 1;
        continue
    end
    if ~isempty(peer_csv) && ~donePeer(row.id) && toc(t0) - tPeer > PEER_REFRESH_S
        donePeer(peer_ids(peer_csv, cols, code_id, stage)) = true;
        tPeer = toc(t0);
    end
    if donePeer(row.id)
        npeer = npeer + 1;
        continue
    end
    line = one_row(row, order(k), stage, with_var, P, mpopt, code_id);
    if isempty(line)
        nfail = nfail + 1;
    else
        append_line(outcsv, line);   % an I/O failure stops the worker
        doneOwn(row.id) = true;
        ncomp = ncomp + 1;
    end
    if mod(ncomp, 20) == 0 && ~isempty(line)
        fprintf(['position %d of %d (frame row %d), %.0f s: computed %d, skipped own %d, ', ...
            'skipped peer %d, failed %d\n'], k, numel(order), order(k), toc(t0), ncomp, ...
            nown, npeer, nfail);
    end
end
fprintf(['FINISHED %s %s range %d..%d: computed %d, skipped own %d, skipped peer %d, ', ...
    'failed %d, %.0f s\n'], frame_name, stage, r_first, r_last, ncomp, nown, npeer, nfail, ...
    toc(t0));
end

function f = test_frame_file(root)
% The held-out test frame, with the fallback of e9_ma2024_index.py.
f = fullfile(root, 'results', 'tables', 'r13', 'r13_tier_battery_rows.csv');
if ~isfile(f)
    f = fullfile(root, 'results', 'tables', 'r15', 'r13_tier_battery_rows.csv');
end
end

function line = one_row(row, pos, stage, with_var, P, mpopt, code_id)
% One frame row; an empty line reports a failed row (printed, then skipped).
line = '';
t0 = tic;
try
    cfg = bank.row2cfg(row);
    X = scenario(cfg, stage, with_var, P, mpopt);
    st = struct('lam_min', row.lam_min, 'gSCR', row.gSCR, 'PWSGSI', row.PWSGSI, ...
        'C_SGSI', row.C_SGSI, 'C_int', row.C_interior, 'C_dc', row.C_dc, ...
        'w_c', row.w_c, 'rule', rule_code(row.thr_rule));
    line = format_row(row.id, pos, stage, st, X, toc(t0), code_id);
catch err
    fprintf('row id %d failed: %s\n', row.id, err.message);
end
end

% ---------------------------------------------------------------------------
function X = scenario(cfg, stage, with_var, P, mpopt)
% The chain and, unless the stage is 'conv', the threshold of the source.
X = blank_result();
[mpc, plant] = build_case(cfg, P);
res = runpf(mpc, mpopt);
assert(res.success == 1, 'e9thr:pf', 'power flow failed');
tune = repmat(struct(), 10, 1);        % the chain's default: design-default converters
if isfield(cfg, 'tune') && ~isempty(cfg.tune), tune = cfg.tune; end
wb = P.wb;
tc = tic;
R = reduced.chain(mpc, res, plant, P, tune);
X.t_chain = toc(tc);
isG = strcmp(R.plant_type(:), 'GFL');
X.nGFL = nnz(isG);
X.nGFM = nnz(~isG);
X.lam_min = R.lam_min;  X.gSCR = R.gSCR;  X.PWSGSI = R.PWSGSI;
X.Hbar = R.Hbar;  X.SCR_min = R.SCR_min;
X.C_SGSI = R.C_SGSI;  X.C_int = R.C_interior;  X.C_dc = R.C_dc;
X.w_c = R.w_c;  X.rule = rule_code(R.thr_rule);
svec = check_points();

% the agg_id gate: this script's aggregation with the chain's weights against
% the chain's own yagg_s (R.Yagg_s)
yproj = @(s) plant_sum(s, R.ports, R.p, isG, R.Leff, wb);
X.agg_id_rel = max_rel_diff(yproj, R.Yagg_s, svec);
if strcmp(stage, 'conv')
    T = reduced.threshold(@(w) yproj(1j*w), wb);
    X.C_mine = T.C_SGSI;  X.rule_mine = rule_code(T.rule);  X.w_c_mine = T.w_c;
    return
end

% the index and the threshold of the source
assert(any(isG) && any(~isG), 'e9thr:partition', 'the row has no GFL or no GFM plant');
N = ma_network(mpc, res, R.BB, ~isG, strcmp(stage, 'twin'));
X.gscr = N.gscr;  X.gscr_g2b = N.gscr_g2b;  X.lam_gap = N.lam_gap;
X.psum = sum(N.p1);  X.v1u1 = N.v1.' * N.u1;  X.eig_res = N.eig_res;
Gfun = @(s) ma_G(s, N, R, wb, -1, true);
tm = tic;
T = reduced.threshold(@(w) reduced.Fmat(1j*w, wb) * Gfun(1j*w), wb);
X.t_ma = toc(tm);
X.cgscr = T.C_SGSI;  X.cg_w_c = T.w_c;  X.cg_rule = rule_code(T.rule);
X.cg_found = double(T.found);  X.cg_n_reject = T.n_reject;  X.cg_n_cand = size(T.all, 1);
X.cg_im_resid = T.im_resid;  X.cg_C_int = T.C_interior;  X.cg_C_dc = T.C_dc;
X.beta = (X.gscr - X.cgscr) / X.cgscr;   % NaN when no threshold is found
if T.found
    X.cg_dsdl = crossing_rate(X.cgscr, 1j*T.w_c, Gfun);
end

% gates and diagnostics
d0 = ma_delta(0, N, R, wb, -1);
X.limit_rel = norm(ma_delta(1e-6j, N, R, wb, -1) - d0, 'fro') / norm(d0, 'fro');
% the same at s = 1e-9 j: a difference of first order in s falls a thousandfold
X.limit_rel9 = norm(ma_delta(1e-9j, N, R, wb, -1) - d0, 'fro') / norm(d0, 'fro');
[X.schur_rel, X.proj_rel] = algebra_gates(svec, N, R, wb, Gfun);
[X.map_rel, X.map_rel_flip] = map_gate(svec, R, isG, wb);
[X.dev_cos_min, X.dev_cos1_min, X.dev_rel_max] = device_check(mpc, res, plant, P, tune, R);
if with_var
    Tf = reduced.threshold(@(w) reduced.Fmat(1j*w, wb) * ma_G(1j*w, N, R, wb, +1, true), wb);
    X.cgscr_flip = Tf.C_SGSI;
    Tn = reduced.threshold(@(w) reduced.Fmat(1j*w, wb) * ma_G(1j*w, N, R, wb, -1, false), wb);
    X.cgscr_nom = Tn.C_SGSI;
end
end

function sv = check_points()
% s = 0 and five points on the imaginary axis inside the sweep band of
% reduced.threshold (2 rad/s to 2 pi 55 rad/s)
sv = [0, 1j*[2, 2*pi*1, 2*pi*10, 2*pi*30, 2*pi*55]];
end

% ---------------------------------------------------------------------------
function Y = plant_sum(s, ports, w, isG, Leff, wb)
% chain.m's yagg_s restated: the sum over the plants, in plant order, of
% w_i times the GFL port or the rotated closed-form Y_delta of a GFM plant.
Y = zeros(2, 2);
for i = 1:numel(ports)
    if isG(i)
        Y = Y + w(i)*ports{i}.Y(s);
    else
        Y = Y + w(i)*ydelta_common(s, ports{i}, Leff(i), wb);
    end
end
end

function Y = ydelta_common(s, prt, Leff, wb)
% reduced.Ydelta (the injected increment of the synchronization loop, in the
% plant's local frame) rotated to the common frame, as in chain.m's yagg_s.
Yd_loc = reduced.Ydelta(s, Leff, prt.v0, prt.i0, prt.cvt, wb);
Rth = reduced.rot2(prt.th0);
Y = Rth*Yd_loc*Rth.';
end

function N = ma_network(mpc, res, BB, isF, loads_absorbed)
% The Kron-reduced plant-bus susceptance of the source (as published: the
% load susceptance out of B_lap; or the chain's B_kron for the twin), its
% partition, the index of eq. 39 and the eigenvectors of eq. 34.
define_constants;
if loads_absorbed
    Bk = BB.Bkron;
else
    nb = size(mpc.bus, 1);
    noload = BB.Blap;
    Vm = res.bus(:, VM);
    Qd = res.bus(:, QD) / mpc.baseMVA;
    noload(sub2ind([nb nb], 1:nb, 1:nb)) = diag(BB.Blap) - Qd ./ Vm.^2;
    cb = BB.cb;
    intr = setdiff((1:nb)', cb);
    Bk = noload(cb, cb) - noload(cb, intr) * (noload(intr, intr) \ noload(intr, cb));
end
S = (Bk + BB.Bin_sys + (Bk + BB.Bin_sys).') / 2;
isG = ~isF;
N.idxG = find(isG);  N.idxF = find(isF);
N.B1 = S(isG, isG);  N.B2 = S(isG, isF);  N.B4t = S(isF, isF);
N.SbG = BB.Sb(isG);  N.SbF = BB.Sb(isF);
Sch = N.B1 - N.B2 * (N.B4t \ N.B2.');
Schs = (Sch + Sch.') / 2;
Sih = diag(1 ./ sqrt(N.SbG));
N.gscr_g2b = min(eig(Sih * ((Sch + Sch.') / 2) * Sih));   % e9_ma2024_index.m, index_parts
Ys = Sih * Schs * Sih;
[Phi, Dg] = eig((Ys + Ys.') / 2);
[lam, o] = sort(diag(Dg));
phi = Phi(:, o(1)) / norm(Phi(:, o(1)));
N.gscr = lam(1);
N.lam_gap = NaN;
if numel(lam) > 1, N.lam_gap = lam(2) - lam(1); end
N.p1 = phi.^2;
N.u1 = phi ./ sqrt(N.SbG);
N.v1 = phi .* sqrt(N.SbG);
N.Yeq = diag(1 ./ N.SbG) * Schs;
nY = norm(N.Yeq);
N.eig_res = max(norm(N.Yeq*N.u1 - N.gscr*N.u1) / (nY*norm(N.u1)), ...
    norm(N.v1.'*N.Yeq - N.gscr*N.v1.') / (nY*norm(N.v1)));
m = numel(N.idxF);
N.K = kron(N.B4t \ eye(m), eye(2));
N.Lb = kron(diag(1 ./ N.SbG) * N.B2, eye(2));
N.Rb = kron(N.B2.', eye(2));
N.V1 = kron(N.v1, eye(2));
N.U1 = kron(N.u1, eye(2));
end

function G = ma_G(s, N, R, wb, sgn, with_delta)
% The matrix of eq. 40: sum_i p_1i F(s)^-1 Y_GFL,i(s) - delta_GFM(s).
F = reduced.Fmat(s, wb);
Y = zeros(2, 2);
for q = 1:numel(N.idxG)
    Y = Y + N.p1(q) * R.ports{N.idxG(q)}.Y(s);
end
G = F \ Y;
if with_delta
    G = G - ma_delta(s, N, R, wb, sgn);
end
end

function [d, Nm, Dd] = ma_delta(s, N, R, wb, sgn)
% delta_GFM(s) of eqs. 30, 31 and 35. sgn = -1 is the source's absorbed
% Y_delta = -R Ydelta R' (see the header); sgn = +1 the other sign.
F = reduced.Fmat(s, wb);
m = numel(N.idxF);
Dd = zeros(2*m);
for q = 1:m
    j = N.idxF(q);
    ii = 2*q-1:2*q;
    Dd(ii, ii) = N.SbF(q) * (F \ (sgn * ydelta_common(s, R.ports{j}, R.Leff(j), wb)));
end
KD = N.K * Dd;
Delta = -((eye(2*m) + KD) \ (KD * N.K));
Nm = -N.Lb * Delta * N.Rb;
d = N.V1.' * Nm * N.U1;
end

function r = crossing_rate(lam, sc, Gfun)
% Re(ds/dlambda) of the root s(lambda) of f = det(lambda I - G(s)) through
% (lam, sc), by implicit differentiation: df/dlambda = 2 lambda - trace G(sc)
% exactly; df/ds by a central difference along the real axis (G is analytic
% there and h is small against every time constant of the ports).
h = 1e-4;
f = @(s) det(lam*eye(2) - Gfun(s));
dfds = (f(sc + h) - f(sc - h)) / (2*h);
r = real(-(2*lam - trace(Gfun(sc))) / dfds);
end

% ---------------------------------------------------------------------------
function [schur_rel, proj_rel] = algebra_gates(svec, N, R, wb, Gfun)
% eqs. 28-31 against the Schur complement of eq. 27 onto the GFL block, and
% the projection of eq. 35, at the check points.
n = numel(N.idxG);
m = numel(N.idxF);
g = 1:2*n;
f = 2*n + (1:2*m);
schur_rel = 0;
proj_rel = 0;
for s = svec
    F = reduced.Fmat(s, wb);
    YG = zeros(2*n);
    DG = zeros(2*n);
    for q = 1:n
        ii = 2*q-1:2*q;
        Yq = R.ports{N.idxG(q)}.Y(s);
        YG(ii, ii) = N.SbG(q) * Yq;
        DG(ii, ii) = F \ Yq;
    end
    YD = zeros(2*m);
    for q = 1:m
        ii = 2*q-1:2*q;
        j = N.idxF(q);
        YD(ii, ii) = N.SbF(q) * (-ydelta_common(s, R.ports{j}, R.Leff(j), wb));
    end
    % eq. 27, left-multiplied by kron(I, F^-1) and by S_b^-1 on the GFL rows
    H27 = [-YG + kron(N.B1, F), kron(N.B2, F); kron(N.B2.', F), YD + kron(N.B4t, F)];
    Lm = blkdiag(kron(diag(1 ./ N.SbG), eye(2)), eye(2*m)) * kron(eye(n + m), inv(F));
    Tm = Lm * H27;
    Sch = Tm(g, g) - Tm(g, f) * (Tm(f, f) \ Tm(f, g));
    [~, Nm] = ma_delta(s, N, R, wb, -1);
    MN = kron(N.Yeq, eye(2)) - DG + Nm;        % M(s) + N(s), eqs. 29-31
    schur_rel = max(schur_rel, norm(Sch - MN, 'fro') / norm(MN, 'fro'));
    Gs = Gfun(s);
    Gp = N.gscr * eye(2) - N.V1.' * MN * N.U1;
    proj_rel = max(proj_rel, norm(Gs - Gp, 'fro') / norm(Gs, 'fro'));
end
end

function [rel_m, rel_p] = map_gate(svec, R, isG, wb)
% The source's eq. 27 in plant order on the chain's network (B_mod = B_kron
% + Bin_sys) against the reduced characteristic of reduced.dominant_root,
% -Y_Gdelta(s) + kron(Btilde_red, F(s)): with Y_delta = -R Ydelta R' (rel_m,
% the gate) and with the other sign (rel_p).
BB = R.BB;
np = numel(R.ports);
Bmod = BB.Bkron + BB.Bin_sys;
rel_m = 0;
rel_p = 0;
for s = svec
    F = reduced.Fmat(s, wb);
    YGd = zeros(2*np);
    Y27m = zeros(2*np);
    Y27p = zeros(2*np);
    for i = 1:np
        ii = 2*i-1:2*i;
        if isG(i)
            Yi = BB.Sb(i) * R.ports{i}.Y(s);
            YGd(ii, ii) = Yi;  Y27m(ii, ii) = -Yi;  Y27p(ii, ii) = -Yi;
        else
            Yd = BB.Sb(i) * ydelta_common(s, R.ports{i}, R.Leff(i), wb);
            YGd(ii, ii) = Yd;  Y27m(ii, ii) = -Yd;  Y27p(ii, ii) = Yd;
        end
    end
    H19 = -YGd + kron(BB.Btilde_red, F);
    nrm = norm(H19, 'fro');
    rel_m = max(rel_m, norm(Y27m + kron(Bmod, F) - H19, 'fro') / nrm);
    rel_p = max(rel_p, norm(Y27p + kron(Bmod, F) - H19, 'fro') / nrm);
end
end

function [cos0, cos1, rel0] = device_check(mpc, res, plant, P, tune, R)
% The closed-form R Ydelta(s) R' against the synchronization increment of the
% device itself: the VSM port with the swing live minus the port with it
% frozen (both reduced.port, injected, common frame), at s = 0 and at 1 Hz.
define_constants;
V = res.bus(:, VM) .* exp(1j * res.bus(:, VA) * pi / 180);
Sg = plant_injections(mpc, res, plant);
s1 = 2j*pi;
cos0 = inf;
cos1 = inf;
rel0 = 0;
for i = find(strcmp(plant.type(:), 'VSM')).'
    c = params.converter('VSM', plant.Sn(i), P, tune(min(i, numel(tune))));
    c.Ssys = c.Sn;
    sp = struct('cvt', c, 'k', dev.state_map('VSM'), 'nx', c.nx, 'Ssys', c.Sn);
    Sconv = Sg(i) * mpc.baseMVA / plant.Sn(i);
    live = reduced.port('VSM', sp, V(plant.bus(i)), Sconv, false);
    frz = R.ports{i};
    E0 = live.Y(0) - frz.Y(0);
    Y0 = ydelta_common(0, frz, R.Leff(i), P.wb);
    E1 = live.Y(s1) - frz.Y(s1);
    Y1 = ydelta_common(s1, frz, R.Leff(i), P.wb);
    cos0 = min(cos0, cosine(E0, Y0));
    cos1 = min(cos1, cosine(E1, Y1));
    rel0 = max(rel0, norm(E0 - Y0, 'fro') / norm(E0, 'fro'));
end
end

function c = cosine(A, B)
% The real Frobenius cosine of two (complex) matrices.
c = real(sum(conj(A(:)) .* B(:))) / (norm(A, 'fro') * norm(B, 'fro'));
end

function Sg = plant_injections(mpc, res, plant)
% Generator injections per plant, system pu (chain.m lines 19-25).
define_constants;
Sg = zeros(numel(plant.bus), 1) + 0j;
for gi = 1:size(res.gen, 1)
    pb = find(plant.bus == res.gen(gi, GEN_BUS), 1);
    if ~isempty(pb)
        Sg(pb) = Sg(pb) + (res.gen(gi, PG) + 1j*res.gen(gi, QG))/mpc.baseMVA;
    end
end
end

function r = max_rel_diff(fa, fb, svec)
% The largest relative Frobenius difference of two 2x2 functions of s.
r = 0;
for s = svec
    a = fa(s);
    b = fb(s);
    r = max(r, norm(a - b, 'fro') / max(norm(b, 'fro'), realmin));
end
end

function c = rule_code(r)
% The threshold family that binds: 1 static (w_c = 0), 2 interior, 0 none.
switch char(r)
    case 'dc', c = 1;
    case 'interior', c = 2;
    otherwise, c = 0;
end
end

% ---------------------------------------------------------------------------
function run_rays(outdir, stage, root, P, mpopt, code_id)
% The three rays along k, checkpointed per point.
cols = ray_columns();
fout = fullfile(outdir, sprintf('e9thr_rays_%s.csv', stage));
cache = ray_cache(fout, cols, code_id, stage);
rays = ray_specs(root);
KGRID = (4:60) / 20;          % k = 0.20, 0.25, ..., 3.00
KTOL = 1e-5;
t0 = tic;
for q = 1:numel(rays)
    ray = rays(q);
    fprintf('=== ray %s: k* %.6f (artifact), probe at k = %.6f ===\n', ray.name, ...
        ray.kstar, ray.kprobe);
    eval_point(ray, 'probe', ray.kprobe, cache, fout, cols, stage, P, mpopt, code_id);
    kk = zeros(1, 0);
    dd = zeros(1, 0);
    for k = KGRID
        v = eval_point(ray, 'grid', k, cache, fout, cols, stage, P, mpopt, code_id);
        if ~v.ok
            fprintf('ray %s: the grid stops at its first failed point, k = %.2f\n', ray.name, k);
            break
        end
        kk(end+1) = k;   %#ok<AGROW>
        dd(end+1) = v.d; %#ok<AGROW>
    end
    fin = isfinite(dd);
    kf = kk(fin);
    up = dd(fin) >= 0;            % gSCR >= CgSCR: the stable side of the rule
    chg = find(up(1:end-1) ~= up(2:end));
    fprintf('ray %s: %d grid points, %d without CgSCR, %d sign changes of gSCR - CgSCR\n', ...
        ray.name, numel(kk), nnz(~fin), numel(chg));
    for c = chg
        klo = kf(c);
        khi = kf(c + 1);
        ulo = up(c);
        while khi - klo > KTOL
            km = (klo + khi) / 2;
            v = eval_point(ray, 'bisect', km, cache, fout, cols, stage, P, mpopt, code_id);
            if ~v.ok || ~isfinite(v.d)
                fprintf('ray %s: bisection stopped at k = %.6f (no CgSCR or a failed point)\n', ...
                    ray.name, km);
                break
            end
            if (v.d >= 0) == ulo
                klo = km;
            else
                khi = km;
            end
        end
        fprintf('ray %s: sign change in [%.6f, %.6f]\n', ray.name, klo, khi);
    end
end
fprintf('FINISHED rays %s: %.0f s\n', stage, toc(t0));
end

function rays = ray_specs(root)
% The design ray of fs3_boundary.m and the bank rays of r20_straddle2.m, with
% the full model's k* and the stable-side probe readings from their artifacts.
rays = struct('name', {}, 'cfg', {}, 'kstar', {}, 'kprobe', {}, 'ref', {});
% design ray (fs3_boundary.m, its k-sweep at load scale 1.0): GFM plants at
% buses 38 and 39, design-default tune, no outage; probe at k* - 0.005
kL = 1.0;
F3 = readtable(fullfile(root, 'results', 'feasibility', 'fs3_boundary.csv'), 'TextType', 'char');
r = F3(strcmp(F3.ray, sprintf('k-sweep kL=%.1f', kL)), :);
assert(height(r) == 1 && strcmp(r.mode{1}, 'crossing'), 'e9thr:ray', ...
    'the design ray is not in fs3_boundary.csv');
dv = repmat({'GFL'}, 1, 10);
dv{9} = 'VSM';
dv{10} = 'VSM';
rays(end+1) = struct('name', 'design', 'cfg', struct('devtype', {dv}, 'kL', kL), ...
    'kstar', r.kstar_full, 'kprobe', r.kstar_full - 0.005, ...
    'ref', [r.lam_min_star, r.gSCR_star, r.PWSGSI_star]);
% bank rays: the rays of the EMT straddle on bank-drawn rays (r20_straddle2.m),
% specifications from r17_rays_spec.mat, k* and probe from r17_rays.csv
S = load(fullfile(root, 'results', 'tables', 'r17', 'r17_rays_spec.mat'));
R17 = readtable(fullfile(root, 'results', 'tables', 'r17', 'r17_rays.csv'), 'TextType', 'char');
E20 = readtable(fullfile(root, 'results', 'tables', 'r20', 'r20_straddle2.csv'), 'TextType', 'char');
for ray = unique(E20.rayid).'
    ci = S.cfgid_r(ray);
    cfg = mkcfg(S.cfgs{ci}, S.outid_r(ray), S.kL_r(ray), S.kdis_r(ray, :), S.fplc_r(ray), ...
        S.fplj_r(ray, :), S.Hv_r(ray, :), S.Dv_r(ray, :), S.kbw_r(ray));
    row = R17(R17.rayid == ray, :);
    assert(height(row) == 1 && abs(row.kL - S.kL_r(ray)) < 1e-6 && ...
        row.nGFM == numel(S.cfgs{ci}) && strcmp(row.mode{1}, 'crossing'), 'e9thr:ray', ...
        'ray %d: specification and r17_rays.csv disagree', ray);
    rays(end+1) = struct('name', sprintf('bank%d', ray), 'cfg', cfg, 'kstar', row.kstar, ...
        'kprobe', row.k_probe, 'ref', [row.lam_min_star, row.gSCR_star, row.GRIP_star]); %#ok<AGROW>
end
end

function cfg = mkcfg(gfmb, outid, kL, kdis, fplc, fplj, Hvs, Dvs, kbw)
% Copied without change from r17_rays.m (how the bank's one_scenario builds
% the tune).
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

function v = eval_point(ray, kind, k, cache, fout, cols, stage, P, mpopt, code_id)
% One point of a ray; a point already in the file is read back, not rerun.
key = sprintf('%s|%.17g', ray.name, k);
if isKey(cache, key)
    v = cache(key);
    return
end
t0 = tic;
cfg = ray.cfg;
cfg.k = k;
X = blank_result();
ok = 0;
try
    X = scenario(cfg, stage, false, P, mpopt);
    ok = 1;
catch err
    fprintf('ray %s k %.6f failed: %s\n', ray.name, k, err.message);
end
append_line(fout, format_ray(ray, kind, k, ok, stage, X, toc(t0), code_id, numel(cols)));
v = struct('ok', ok, 'd', X.gscr - X.cgscr);
% containers.Map is a handle: the caller's cache holds the new point
cache(key) = v; %#ok<NASGU>
end

function cache = ray_cache(fout, cols, code_id, stage)
% The points already in the ray file, for the resume.
cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
header = strjoin(cols, ',');
if ~isfile(fout)
    write_text(fout, sprintf('%s\n', header));
    return
end
lines = complete_lines(fout, header);
col = @(name) find(strcmp(cols, name));
for q = 1:numel(lines)
    f = strsplit(lines{q}, ',', 'CollapseDelimiters', false);
    if numel(f) ~= numel(cols)
        error('e9thr:corrupt', '%s line %d is malformed', fout, q + 1);
    end
    if ~strcmp(f{end}, code_id) || ~strcmp(f{col('stage')}, stage)
        error('e9thr:identity', '%s holds points of another code version or stage', fout);
    end
    k = str2double(f{col('k')});
    d = str2double(f{col('gscr')}) - str2double(f{col('cgscr')});
    cache(sprintf('%s|%.17g', f{col('ray')}, k)) = struct('ok', str2double(f{col('ok')}), 'd', d);
end
end

% ---------------------------------------------------------------------------
function X = blank_result()
% Every quantity of a scenario, NaN until computed.
names = {'nGFL', 'nGFM', 'lam_min', 'gSCR', 'PWSGSI', 'Hbar', 'SCR_min', 'C_SGSI', ...
    'C_int', 'C_dc', 'w_c', 'rule', 'agg_id_rel', 'C_mine', 'rule_mine', 'w_c_mine', ...
    'gscr', 'gscr_g2b', 'lam_gap', 'psum', 'v1u1', 'eig_res', 'cgscr', 'cg_w_c', ...
    'cg_rule', 'cg_found', 'cg_n_reject', 'cg_n_cand', 'cg_im_resid', 'cg_C_int', ...
    'cg_C_dc', 'beta', 'cg_dsdl', 'limit_rel', 'limit_rel9', 'schur_rel', 'proj_rel', ...
    'map_rel', 'map_rel_flip', 'dev_cos_min', 'dev_cos1_min', 'dev_rel_max', 'cgscr_flip', ...
    'cgscr_nom', 't_chain', 't_ma'};
for q = 1:numel(names)
    X.(names{q}) = NaN;
end
end

function c = columns()
c = {'id', 'pos', 'stage', 'nGFL', 'nGFM', ...
    'lam_min_row', 'lam_min_re', 'gSCR_row', 'gSCR_re', 'PWSGSI_row', 'PWSGSI_re', ...
    'C_SGSI_row', 'C_SGSI_re', 'C_int_row', 'C_int_re', 'C_dc_row', 'C_dc_re', ...
    'w_c_row', 'w_c_re', 'rule_row', 'rule_re', ...
    'agg_id_rel', 'C_mine', 'rule_mine', 'w_c_mine', ...
    'gscr', 'gscr_g2b', 'lam_gap', 'psum', 'v1u1', 'eig_res', ...
    'cgscr', 'cg_w_c', 'cg_rule', 'cg_found', 'cg_n_reject', 'cg_n_cand', ...
    'cg_im_resid', 'cg_C_int', 'cg_C_dc', 'beta', 'cg_dsdl', ...
    'limit_rel', 'limit_rel9', 'schur_rel', 'proj_rel', 'map_rel', 'map_rel_flip', ...
    'dev_cos_min', 'dev_cos1_min', 'dev_rel_max', 'cgscr_flip', 'cgscr_nom', ...
    't_chain', 't_ma', 't_row', 'code_id'};
end

function line = format_row(id, pos, stage, st, X, t_row, code_id)
v = [X.nGFL, X.nGFM, st.lam_min, X.lam_min, st.gSCR, X.gSCR, st.PWSGSI, X.PWSGSI, ...
    st.C_SGSI, X.C_SGSI, st.C_int, X.C_int, st.C_dc, X.C_dc, st.w_c, X.w_c, ...
    st.rule, X.rule, X.agg_id_rel, X.C_mine, X.rule_mine, X.w_c_mine, ...
    X.gscr, X.gscr_g2b, X.lam_gap, X.psum, X.v1u1, X.eig_res, ...
    X.cgscr, X.cg_w_c, X.cg_rule, X.cg_found, X.cg_n_reject, X.cg_n_cand, ...
    X.cg_im_resid, X.cg_C_int, X.cg_C_dc, X.beta, X.cg_dsdl, ...
    X.limit_rel, X.limit_rel9, X.schur_rel, X.proj_rel, X.map_rel, X.map_rel_flip, ...
    X.dev_cos_min, X.dev_cos1_min, X.dev_rel_max, X.cgscr_flip, X.cgscr_nom, ...
    X.t_chain, X.t_ma, t_row];
assert(numel(v) + 4 == numel(columns()), 'e9thr:columns', 'a row has %d fields', numel(v) + 4);
line = sprintf('%d,%d,%s,%s%s\n', id, pos, stage, sprintf('%.17g,', v), code_id);
end

function c = ray_columns()
c = {'ray', 'kind', 'k', 'ok', 'stage', 'nGFL', 'nGFM', ...
    'lam_min', 'gSCR_cl', 'PWSGSI', 'Hbar', 'SCR_min', 'C_SGSI', 'rule', 'w_c', ...
    'gscr', 'cgscr', 'cg_w_c', 'cg_rule', 'cg_found', 'cg_n_reject', 'cg_C_int', ...
    'cg_C_dc', 'beta', 'cg_dsdl', 'limit_rel', 'limit_rel9', 'schur_rel', 'proj_rel', ...
    'map_rel', 'dev_cos_min', 'kstar_ref', 'lam_min_ref', 'gSCR_ref', 'GRIP_ref', ...
    't_chain', 't_ma', 't_row', 'code_id'};
end

function line = format_ray(ray, kind, k, ok, stage, X, t_row, code_id, ncol)
v = [X.nGFL, X.nGFM, X.lam_min, X.gSCR, X.PWSGSI, X.Hbar, X.SCR_min, X.C_SGSI, ...
    X.rule, X.w_c, X.gscr, X.cgscr, X.cg_w_c, X.cg_rule, X.cg_found, X.cg_n_reject, ...
    X.cg_C_int, X.cg_C_dc, X.beta, X.cg_dsdl, X.limit_rel, X.limit_rel9, X.schur_rel, ...
    X.proj_rel, X.map_rel, X.dev_cos_min, ray.kstar, ray.ref, X.t_chain, X.t_ma, t_row];
assert(numel(v) + 6 == ncol, 'e9thr:columns', 'a ray point has %d fields', numel(v) + 6);
line = sprintf('%s,%s,%.17g,%d,%s,%s%s\n', ray.name, kind, k, ok, stage, ...
    sprintf('%.17g,', v), code_id);
end

% ---------------------------------------------------------------------------
function ids = resume_own(outcsv, cols, code_id, stage)
% Prepare the worker's own rows file and return the ids already in it. A
% missing file is created with the header; a foreign header, a malformed
% line, another stage or a row of another code version is an error, and the
% file is not appended to.
header = strjoin(cols, ',');
ids = zeros(0, 1);
if ~isfile(outcsv)
    write_text(outcsv, sprintf('%s\n', header));
    return
end
[ids, codes, stages, okfmt] = parse_lines(complete_lines(outcsv, header), numel(cols));
if ~all(okfmt)
    error('e9thr:corrupt', '%s holds %d malformed lines', outcsv, nnz(~okfmt));
end
if ~all(strcmp(codes, code_id)) || ~all(strcmp(stages, stage))
    error('e9thr:identity', '%s holds rows of another code version or stage', outcsv);
end
end

function lines = complete_lines(f, header)
% The data lines of a file of this script. A last line cut by a kill (no
% trailing newline) is dropped through a temporary file and a rename; a
% file with another header is refused.
txt = fileread(f);
cut = find(txt == newline, 1, 'last');
if isempty(cut)
    error('e9thr:corrupt', '%s has no complete header line', f);
end
if cut < numel(txt)
    fprintf('dropping the cut last line of %s\n', f);
    tmp = [f '.tmp'];
    write_text(tmp, txt(1:cut));
    movefile(tmp, f, 'f');
    txt = txt(1:cut);
end
lines = strsplit(txt(1:end-1), newline, 'CollapseDelimiters', false);
if ~strcmp(lines{1}, header)
    error('e9thr:header', '%s has another header; refusing to append', f);
end
lines = lines(2:end);
end

function ids = peer_ids(peer_csv, cols, code_id, stage)
% Ids of the complete rows of the peer's file written by this code version
% at this stage. The peer's file is only read, never changed; a read that
% fails (absent or held) returns no ids and is retried at the next refresh.
ids = zeros(0, 1);
if ~isfile(peer_csv), return; end
try
    txt = fileread(peer_csv);
catch
    return
end
cut = find(txt == newline, 1, 'last');
if isempty(cut), return; end
lines = strsplit(txt(1:cut-1), newline, 'CollapseDelimiters', false);
if ~strcmp(lines{1}, strjoin(cols, ',')), return; end
[pid, codes, stages, okfmt] = parse_lines(lines(2:end), numel(cols));
ids = pid(okfmt & strcmp(codes, code_id) & strcmp(stages, stage));
end

function [ids, codes, stages, okfmt] = parse_lines(body, ncol)
% Split complete data lines; a line is well formed when it has ncol fields
% and a positive integer id.
n = numel(body);
ids = nan(n, 1);
codes = repmat({''}, n, 1);
stages = repmat({''}, n, 1);
okfmt = false(n, 1);
for k = 1:n
    f = strsplit(body{k}, ',', 'CollapseDelimiters', false);
    if numel(f) == ncol
        v = str2double(f{1});
        if isfinite(v) && v >= 1 && v == round(v)
            ids(k) = v;
            codes{k} = f{end};
            stages{k} = f{3};
            okfmt(k) = true;
        end
    end
end
end

function append_line(f, line)
% Append one complete line and close the file, so every finished row is on
% disk before the next one starts.
for attempt = 1:10
    fid = fopen(f, 'a');
    if fid >= 0
        nbytes = fprintf(fid, '%s', line);
        status = fclose(fid);
        if nbytes ~= numel(line) || status ~= 0
            error('e9thr:io', 'incomplete write to %s', f);
        end
        return
    end
    pause(0.5);
end
error('e9thr:io', 'cannot open %s for appending', f);
end

function write_text(f, txt)
fid = fopen(f, 'w');
if fid < 0, error('e9thr:io', 'cannot write %s', f); end
fprintf(fid, '%s', txt);
fclose(fid);
end

function h = code_hash(f)
% First 16 hex digits of the SHA-256 of a text file with carriage returns
% removed, so the identity does not depend on the line-ending convention.
fid = fopen(f, 'r');
b = fread(fid, inf, '*uint8');
fclose(fid);
b(b == 13) = [];
md = java.security.MessageDigest.getInstance('SHA-256');
d = typecast(md.digest(typecast(b, 'int8')), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
h = h(1:16);
end

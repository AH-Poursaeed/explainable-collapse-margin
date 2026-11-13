function r17_delay
%r17_delay  Add a one-sample + PWM actuation delay and trace the design ray again.
%   The delay is added to both converter classes as a first-order Pade
%   stage between the inner-loop voltage command and the LCL. Then the
%   flagship design ray is traced again. Td = n_ts/fsw with fsw = 5 kHz.
%   The paper's delay for digital control with PWM is 1.5 Ts (one sample of
%   computation + a half-sample ZOH). The values 0 (identity gate), 1.0 Ts
%   and 2.0 Ts bracket it.
%
%   Pade(1,1): e^(-s Td) ~ (1 - s Td/2)/(1 + s Td/2). Each voltage channel
%   implements it as  zdot = (2/Td)(u - z),  v_applied = 2 z - u. The stage
%   is all-pass and exact at DC, so the equilibrium does not change and
%   z0 = u0 initializes it exactly. It adds two states per converter. The
%   published packages +dev/+full stay unchanged. The device algebra is
%   copied exactly into local functions, with one substitution
%   (v_applied -> LCL). Gates A and B check the copies against full.rhs to
%   1e-12, at the equilibrium and at a perturbed point. vapp = 2z - u with
%   z = u is exact in IEEE arithmetic, so an exact copy gives a zero difference.
%
%   Stages:
%     recon (default) -- the 1 h check: the copy-fidelity gates, the residual
%       gate of the delayed init/polish, and a modal comparison at the
%       flagship base point. Writes r17_delay_recon.json.
%     main -- flagship-ray traces at Td/Ts in {0, 1.0, 1.5, 2.0}. Each trace
%       scans and bisects the band-limited sigma = 0 crossing (warm-started
%       PF) and records the super-synchronous gate (oob_maxRe) at every
%       evaluation. Td = 0 runs the published pipeline and must reproduce the
%       fs3 record k* = 1.973046875 to the bisection tolerance.
%       Writes r17_delay_trace.csv + r17_delay_summary.csv.
%
%   Design ray: buses 38/39 VSM (largest-rating GFM slack), kL = 1.0,
%   default tune (fs3_boundary configuration).

maxNumCompThreads(2);
P = params.sys_const();
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'tables', 'r17');
if ~exist(outdir, 'dir'), mkdir(outdir); end

stage = getenv('DELAY_STAGE');
if isempty(stage), stage = 'recon'; end

FSW = 5000;                       % params.converter switching class [Hz]
TS = 1/FSW;
KSTAR_REF = 1.973046875;          % fs3_boundary.csv, k-sweep kL=1.0 record

dv = repmat({'GFL'}, 1, 10);  dv{9} = 'VSM';  dv{10} = 'VSM';
cfg = struct('devtype', {dv}, 'kL', 1.0, 'k', 1.0);

switch stage
    case 'recon'
        recon(cfg, P, TS, outdir);
    case 'main'
        main_traces(cfg, P, TS, KSTAR_REF, outdir);
    otherwise
        error('r17_delay:stage', 'unknown stage %s', stage);
end
end

% ===========================================================================
function recon(cfg, P, TS, outdir)
t0 = tic;
rep = struct();
fprintf('=== r17_delay RECON (flagship base point k = 1.0) ===\n');
[mpc, res, plant, M, x0] = base_published(cfg, 1.0, P);
f_pub = full.rhs(x0, M);
rep.resid_pub = norm(f_pub, inf);
fprintf('published base: resid %.2e\n', rep.resid_pub);

Td = 1.5*TS;
[Md, x0d] = delay_model(M, x0, Td);

% ---- gate A: copy fidelity at the equilibrium -----------------------------
fd = rhs_d(x0d, Md);
gA = norm(fd(1:M.nx) - f_pub, inf);
gAz = norm(fd(M.nx+1:end), inf);
rep.gateA_body = gA;  rep.gateA_zrows = gAz;
fprintf('gate A (equilibrium copy fidelity): body %.2e, z rows %.2e\n', gA, gAz);

% ---- gate B: copy fidelity at a perturbed point ---------------------------
rng(P.seed + 44, 'twister');
xp = x0 + 1e-3*randn(M.nx, 1);
zp = eq_delay_states(xp, Md);
fp_pub = full.rhs(xp, M);
fp_d = rhs_d([xp; zp], Md);
gB = norm(fp_d(1:M.nx) - fp_pub, inf);
rep.gateB_body = gB;
fprintf('gate B (perturbed copy fidelity): body %.2e\n', gB);

% ---- gate C: delayed init + polish residual -------------------------------
[sigd, okd, aux] = eval_delayed(cfg, 1.0, Td, P, []);
rep.delayed_ok = okd;
rep.delayed_resid = aux.resid;
rep.delayed_sigma = sigd;
rep.delayed_oob_maxRe = aux.oob_maxRe;
S_pub = full.modal(full.linearize(M, x0, 'cs').Asys, P);
rep.pub_sigma = S_pub.sigma_dom;
rep.pub_oob_maxRe = S_pub.oob_maxRe;
fprintf(['gate C (delayed init/polish): ok %d, resid %.2e\n' ...
    'base sigma: published %+.6f -> delayed %+.6f (d = %+.2e)\n' ...
    'oob_maxRe: published %+.4f -> delayed %+.4f\n'], ...
    okd, aux.resid, S_pub.sigma_dom, sigd, sigd - S_pub.sigma_dom, ...
    S_pub.oob_maxRe, aux.oob_maxRe);

% ---- Td scaling check: the shift at 0.15 Ts should be ~10x smaller --------
[sig_s, ok_s] = eval_delayed(cfg, 1.0, 0.15*TS, P, []);
rep.small_Td_sigma = sig_s;  rep.small_Td_ok = ok_s;
fprintf('Td scaling: d_sigma(1.5Ts) = %+.3e, d_sigma(0.15Ts) = %+.3e\n', ...
    sigd - S_pub.sigma_dom, sig_s - S_pub.sigma_dom);

go = (gA < 1e-12) && (gB < 1e-12) && okd;
if go, rep.verdict = 'GO'; else, rep.verdict = 'NO-GO'; end
rep.wall_s = toc(t0);
fid = fopen(fullfile(outdir, 'r17_delay_recon.json'), 'w');
fprintf(fid, '%s', jsonencode(rep, 'PrettyPrint', true));
fclose(fid);
fprintf('RECON VERDICT: %s (%.0f s)\n', rep.verdict, rep.wall_s);
end

% ===========================================================================
function main_traces(cfg, P, TS, KSTAR_REF, outdir)
t0 = tic;
ftr = fullfile(outdir, 'r17_delay_trace.csv');
fsm = fullfile(outdir, 'r17_delay_summary.csv');
fid = fopen(ftr, 'w');
fprintf(fid, 'Td_ts,k,sigma_band,oob_maxRe,oob_stable,f_dom,ok\n');
rows = {};
for Tdts = [0, 1.0, 1.5, 2.0]
    Td = Tdts*TS;
    fprintf('=== trace Td = %.1f Ts ===\n', Tdts);
    st = trace_kstar(cfg, Td, P, fid, Tdts);
    st.Td_ts = Tdts;
    rows{end+1} = st;                                          %#ok<AGROW>
    fprintf('Td %.1f Ts: %s kstar = %.6f (oob violations %d, first at k = %.4f)\n', ...
        Tdts, st.mode, st.kstar, st.n_oob_viol, st.k_first_oob);
end
fclose(fid);
T = struct2table([rows{:}]);
T.kstar_ref = repmat(KSTAR_REF, height(T), 1);
T.dk_vs_ref = T.kstar - KSTAR_REF;
writetable(T, fsm);
d0 = abs(T.kstar(T.Td_ts == 0) - KSTAR_REF);
fprintf('Td=0 identity gate vs fs3 record: |dk| = %.2e (tol 2e-3)\n', d0);
fprintf('r17_delay main complete (%.1f min).\n', toc(t0)/60);
end

% ---------------------------------------------------------------------------
function st = trace_kstar(cfg, Td, P, fid, Tdts)
%TRACE_KSTAR  Scan (from 1.0 upward, step 0.1) and bisect (to 1e-3) the
%   band-limited crossing. Records oob at every evaluation. The PF is warm
%   started along the ray.
st = struct('mode', 'unresolved', 'kstar', NaN, 'n_evals', 0, ...
            'n_oob_viol', 0, 'k_first_oob', NaN);
ws = [];
KMAX = 2.4;  DK = 0.1;
klo = NaN;  khi = NaN;
k = 1.0;
while k <= KMAX + 1e-12
    [s, ok, aux, ws] = eval_any(cfg, k, Td, P, ws);
    st.n_evals = st.n_evals + 1;
    logrow(fid, Tdts, k, s, aux, ok);
    if ok && ~aux.oob_stable
        st.n_oob_viol = st.n_oob_viol + 1;
        if isnan(st.k_first_oob), st.k_first_oob = k; end
    end
    fprintf('  scan k=%.3f sigma=%s oob=%+.3f ok=%d\n', k, fmt(s), aux.oob_maxRe, ok);
    if ~ok
        st.mode = 'pf_infeasible_first';
        return
    end
    if s >= 0, khi = k; break; end
    klo = k;
    k = k + DK;
end
if isnan(khi), st.mode = 'no_crossing'; return; end
while khi - klo > 1e-3
    km = 0.5*(klo + khi);
    [s, ok, aux, ws] = eval_any(cfg, km, Td, P, ws);
    st.n_evals = st.n_evals + 1;
    logrow(fid, Tdts, km, s, aux, ok);
    if ok && ~aux.oob_stable
        st.n_oob_viol = st.n_oob_viol + 1;
        if isnan(st.k_first_oob) || km < st.k_first_oob, st.k_first_oob = km; end
    end
    if ~ok, st.mode = 'pf_infeasible_in_bracket'; return; end
    if s >= 0, khi = km; else, klo = km; end
end
st.mode = 'crossing';
st.kstar = 0.5*(klo + khi);
end

function logrow(fid, Tdts, k, s, aux, ok)
fprintf(fid, '%.2f,%.6f,%.8f,%.8f,%d,%.6f,%d\n', Tdts, k, s, ...
    aux.oob_maxRe, aux.oob_stable, aux.f_dom, ok);
end

function s = fmt(v)
if isnan(v), s = 'NaN'; else, s = sprintf('%+.4f', v); end
end

% ---------------------------------------------------------------------------
function [sig, ok, aux, ws] = eval_any(cfg, k, Td, P, ws)
%EVAL_ANY  Published pipeline at Td = 0, delayed pipeline otherwise.
if Td == 0
    [sig, ok, aux, ws] = eval_published(cfg, k, P, ws);
else
    [sig, ok, aux, ws] = eval_delayed_ws(cfg, k, Td, P, ws);
end
end

function [sig, ok, aux, ws] = eval_published(cfg, k, P, ws)
sig = NaN;  ok = false;
aux = struct('oob_maxRe', NaN, 'oob_stable', NaN, 'f_dom', NaN, 'resid', NaN);
try
    [mpc, res, plant, M, x0, ws] = solve_point(cfg, k, P, ws);   %#ok<ASGLU>
    L = full.linearize(M, x0, 'cs');
    S = full.modal(L.Asys, P);
    if S.nzero ~= 1, return; end
    sig = S.sigma_dom;                       % band-limited (design-ray form)
    aux.oob_maxRe = S.oob_maxRe;  aux.oob_stable = S.oob_stable;
    aux.f_dom = S.f_dom;
    ok = true;
catch
end
end

function [sig, ok, aux] = eval_delayed(cfg, k, Td, P, ws)
[sig, ok, aux] = eval_delayed_ws(cfg, k, Td, P, ws);
end

function [sig, ok, aux, ws] = eval_delayed_ws(cfg, k, Td, P, ws)
sig = NaN;  ok = false;
aux = struct('oob_maxRe', NaN, 'oob_stable', NaN, 'f_dom', NaN, 'resid', NaN);
try
    [~, ~, ~, M, x0, ws] = solve_point(cfg, k, P, ws);
    [Md, x0d] = delay_model(M, x0, Td);
    A = lin_d(Md, x0d);
    resid = norm(rhs_d(x0d, Md), inf);  it = 0;
    while resid > 1e-13 && it < 4
        x0d = x0d - lsqminnorm(A, rhs_d(x0d, Md));
        resid = norm(rhs_d(x0d, Md), inf);  it = it + 1;
        if it > 0, A = lin_d(Md, x0d); end
    end
    residual_pu = norm(rhs_d(x0d, Md)./Md.rowscale_d, inf);
    aux.resid = resid;
    if ~(resid < P.tol.resid && residual_pu < 1e-11), return; end
    S = full.modal(A, P);
    if S.nzero ~= 1, return; end
    sig = S.sigma_dom;
    aux.oob_maxRe = S.oob_maxRe;  aux.oob_stable = S.oob_stable;
    aux.f_dom = S.f_dom;
    ok = true;
catch
end
end

function [mpc, res, plant, M, x0, ws] = solve_point(cfg, k, P, ws)
%SOLVE_POINT  PF (with an optional warm start) + published model/init/polish.
c = cfg;  c.k = k;
[mpc, plant] = build_case(c, P);
mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1, ...
                 'pf.tol', 1e-12, 'pf.nr.max_it', 40);
res = [];
if ~isempty(ws)
    mpcw = mpc;
    mpcw.bus(:, 8) = ws.bus(:, 8);
    mpcw.bus(:, 9) = ws.bus(:, 9);
    try
        r1 = runpf(mpcw, mpopt);
        if r1.success == 1, res = r1; end
    catch
    end
end
if isempty(res)
    r1 = runpf(mpc, mpopt);
    assert(r1.success == 1, 'powerflow');
    res = r1;
end
tune = repmat(struct(), 10, 1);
if isfield(c, 'tune') && ~isempty(c.tune), tune = c.tune; end
M = full.build_model(mpc, res, plant, P, tune);
[x0, M, ~] = full.init_state(M, mpc, res);
L = full.linearize(M, x0, 'cs');
resid = norm(full.rhs(x0, M), inf);  it = 0;
while resid > 1e-13 && it < 4
    x0 = x0 - lsqminnorm(L.Asys, full.rhs(x0, M));
    resid = norm(full.rhs(x0, M), inf);  it = it + 1;
end
residual_pu = norm(full.rhs(x0, M)./M.rowscale, inf);
assert(resid < P.tol.resid && residual_pu < 1e-12, 'init_residual');
ws = res;
end

function [mpc, res, plant, M, x0] = base_published(cfg, k, P)
[mpc, res, plant, M, x0, ~] = solve_point(cfg, k, P, []);
end

% ===========================================================================
% Delayed-model assembly
function [Md, x0d] = delay_model(M, x0, Td)
Md = M;
ndev = numel(M.dev);
Md.Td = Td;
Md.ix_del = M.nx + (1:2*ndev);
Md.nx_d = M.nx + 2*ndev;
rs = [M.rowscale; ones(2*ndev, 1)*(2/Td)];
Md.rowscale_d = rs;
z = eq_delay_states(x0, Md);
x0d = [x0; z];
end

function z = eq_delay_states(x, Md)
%EQ_DELAY_STATES  z_i = the present commanded converter voltage (local
%   frame). This is the exact all-pass equilibrium (vapp = 2z - u = u at z = u).
ndev = numel(Md.dev);
z = zeros(2*ndev, 1);
if ~isreal(x), z = complex(z); end
nb = Md.N.nbus;
vb = reshape(x(Md.ix_bus), 2, nb);
for i = 1:ndev
    d = Md.dev(i);  b = d.bus;
    [~, ~, ~, u, ~] = dev_core(d.type, x(d.ix), vb(1,b), vb(2,b), ...
                               d.params, [0; 0], 1, false);
    z(2*i-1:2*i) = u;
end
end

function f = rhs_d(x, Md)
%RHS_D  full.rhs copied exactly, except that the device calls go to the
%   delay-aware local cores and the Pade z rows are appended.
N = Md.N;  wb = N.wb;
nbr = N.nbr; nb = N.nbus; nld = N.nload;

ibr = reshape(x(Md.ix_br),  2, nbr);
vb  = reshape(x(Md.ix_bus), 2, nb);
if nld > 0
    ild = reshape(x(Md.ix_load), 2, nld);
else
    ild = zeros(2, 0);
end

f = zeros(size(x));
if ~isreal(x), f = complex(f); end
Iinj = zeros(2, nb);
if ~isreal(x), Iinj = complex(Iinj); end

% ---- devices (delay-aware cores) ------------------------------------------
for i = 1:numel(Md.dev)
    d = Md.dev(i); b = d.bus;
    zi = x(Md.ix_del(2*i-1:2*i));
    [fd, ID, IQ, ~, fz] = dev_core(d.type, x(d.ix), vb(1,b), vb(2,b), ...
                                   d.params, zi, Md.Td, true);
    f(d.ix) = fd;
    f(Md.ix_del(2*i-1:2*i)) = fz;
    Iinj(1,b) = Iinj(1,b) + ID;
    Iinj(2,b) = Iinj(2,b) + IQ;
end

% ---- branches (as in full.rhs) --------------------------------------------
fF = N.f; tT = N.t; tap = N.tap;
for e = 1:nbr
    id = ibr(1,e); iq = ibr(2,e);
    vFd = vb(1,fF(e))/tap(e); vFq = vb(2,fF(e))/tap(e);
    vTd = vb(1,tT(e));        vTq = vb(2,tT(e));
    did = (wb/N.x(e))*(vFd - vTd - N.r(e)*id) + wb*iq;
    diq = (wb/N.x(e))*(vFq - vTq - N.r(e)*iq) - wb*id;
    f(Md.ix_br(2*e-1)) = did;
    f(Md.ix_br(2*e))   = diq;
    Iinj(1,fF(e)) = Iinj(1,fF(e)) - id/tap(e);
    Iinj(2,fF(e)) = Iinj(2,fF(e)) - iq/tap(e);
    Iinj(1,tT(e)) = Iinj(1,tT(e)) + id;
    Iinj(2,tT(e)) = Iinj(2,tT(e)) + iq;
end

% ---- series-RL loads (as in full.rhs) -------------------------------------
for l = 1:nld
    b = N.loadbus(l);
    id = ild(1,l); iq = ild(2,l);
    f(Md.ix_load(2*l-1)) = (wb/N.Xs(l))*(vb(1,b) - N.Rs(l)*id) + wb*iq;
    f(Md.ix_load(2*l))   = (wb/N.Xs(l))*(vb(2,b) - N.Rs(l)*iq) - wb*id;
    Iinj(1,b) = Iinj(1,b) - id;
    Iinj(2,b) = Iinj(2,b) - iq;
end

% ---- CPL legs (as in full.rhs; the design ray has none) -------------------
if isfield(N, 'ncpl') && N.ncpl > 0
    icp = reshape(x(Md.ix_cpl), 2, N.ncpl);
    for l = 1:N.ncpl
        b = N.cplbus(l);
        vd = vb(1,b); vq = vb(2,b);
        v2 = vd*vd + vq*vq;
        ird = (N.Pc(l)*vd + N.Qc(l)*vq)/v2;
        irq = (N.Pc(l)*vq - N.Qc(l)*vd)/v2;
        f(Md.ix_cpl(2*l-1)) = (ird - icp(1,l))/N.tauc;
        f(Md.ix_cpl(2*l))   = (irq - icp(2,l))/N.tauc;
        Iinj(1,b) = Iinj(1,b) - icp(1,l);
        Iinj(2,b) = Iinj(2,b) - icp(2,l);
    end
end

% ---- bus capacitors (as in full.rhs) --------------------------------------
for b = 1:nb
    vd = vb(1,b); vq = vb(2,b);
    f(Md.ix_bus(2*b-1)) = (wb/N.Bc(b))*(Iinj(1,b) - N.G(b)*vd) + wb*vq;
    f(Md.ix_bus(2*b))   = (wb/N.Bc(b))*(Iinj(2,b) - N.G(b)*vq) - wb*vd;
end
end

function A = lin_d(Md, x0d)
%LIN_D  Complex-step Jacobian of rhs_d (same method as full.linearize).
n = Md.nx_d;
A = zeros(n);
h = 1e-200;
for kk = 1:n
    xp = x0d; xp(kk) = xp(kk) + 1i*h;
    A(:, kk) = imag(rhs_d(xp, Md))/h;
end
end

% ===========================================================================
function [fd, ID, IQ, u, fz] = dev_core(type, xd, VDi, VQi, sp, z, Td, delay_on)
%DEV_CORE  dev.gfl_rhs / dev.gfm_vsm_rhs copied exactly, with one
%   substitution: the converter-side LCL rows receive v_applied (the Pade
%   output 2z - u when delay_on, or u itself otherwise). u is the commanded
%   vconv from the unchanged controller algebra. fz holds the Pade state
%   derivatives (zeros when delay_on = false).
k = sp.k; c = sp.cvt; wb = c.wb;
switch upper(type)
    case 'GFL'
        th = xd(k.theta); eps_ = xd(k.eps_pll); Pm = xd(k.Pm); Qm = xd(k.Qm);
        chid = xd(k.chi_d); chiq = xd(k.chi_q); gamd = xd(k.gam_d); gamq = xd(k.gam_q);
        ifd = xd(k.ifd); ifq = xd(k.ifq); vcd = xd(k.vcd); vcq = xd(k.vcq);
        igd = xd(k.igd); igq = xd(k.igq);

        ct = cos(th); st = sin(th);
        vpd =  VDi*ct + VQi*st;
        vpq = -VDi*st + VQi*ct;

        wpll = 1 + c.pll_Kp*vpq + c.pll_Ki*eps_;

        Ppcc = vpd*igd + vpq*igq;
        Qpcc = vpq*igd - vpd*igq;

        ifref_d =   c.Kpp*(sp.ref.Pstar - Pm) + c.Kip*chid;
        ifref_q = -(c.Kpq*(sp.ref.Qstar - Qm) + c.Kiq*chiq);

        vconv_d = c.Kpi*(ifref_d - ifd) + c.Kii*gamd + vcd - wpll*c.Xl1*ifq - c.Kad*(ifd - igd);
        vconv_q = c.Kpi*(ifref_q - ifq) + c.Kii*gamq + vcq + wpll*c.Xl1*ifd - c.Kad*(ifq - igq);
        u = [vconv_d; vconv_q];
        if delay_on
            vapp_d = 2*z(1) - vconv_d;  vapp_q = 2*z(2) - vconv_q;
            fz = [(2/Td)*(vconv_d - z(1)); (2/Td)*(vconv_q - z(2))];
        else
            vapp_d = vconv_d;  vapp_q = vconv_q;
            fz = [0; 0];
        end

        vmid_d = vcd + c.Rd*(ifd - igd);
        vmid_q = vcq + c.Rd*(ifq - igq);
        difd = (wb/c.Xl1)*(vapp_d - vmid_d - c.r1*ifd) + wpll*wb*ifq;
        difq = (wb/c.Xl1)*(vapp_q - vmid_q - c.r1*ifq) - wpll*wb*ifd;
        dvcd = (wb/c.Bcf)*(ifd - igd) + wpll*wb*vcq;
        dvcq = (wb/c.Bcf)*(ifq - igq) - wpll*wb*vcd;
        digd = (wb/c.Xl2)*(vmid_d - vpd - c.r2*igd) + wpll*wb*igq;
        digq = (wb/c.Xl2)*(vmid_q - vpq - c.r2*igq) - wpll*wb*igd;

        fd = zeros(sp.nx, 1);
        if ~isreal(xd) || ~isreal(VDi), fd = complex(fd); end
        fd(k.theta)   = (wpll - 1)*wb;
        fd(k.eps_pll) = vpq;
        fd(k.Pm)      = c.wc*(Ppcc - Pm);
        fd(k.Qm)      = c.wc*(Qpcc - Qm);
        fd(k.chi_d)   = sp.ref.Pstar - Pm;
        fd(k.chi_q)   = sp.ref.Qstar - Qm;
        fd(k.gam_d)   = ifref_d - ifd;
        fd(k.gam_q)   = ifref_q - ifq;
        fd(k.ifd) = difd; fd(k.ifq) = difq;
        fd(k.vcd) = dvcd; fd(k.vcq) = dvcq;
        fd(k.igd) = digd; fd(k.igq) = digq;

        IDl = igd*ct - igq*st;  IQl = igd*st + igq*ct;
        scl = c.Sn/c.Ssys;
        ID = IDl*scl;  IQ = IQl*scl;

    case 'VSM'
        th = xd(k.theta); w = xd(k.w); Pm = xd(k.Pm); Qm = xd(k.Qm);
        xivd = xd(k.xiv_d); xivq = xd(k.xiv_q); gamd = xd(k.gam_d); gamq = xd(k.gam_q);
        ifd = xd(k.ifd); ifq = xd(k.ifq); vcd = xd(k.vcd); vcq = xd(k.vcq);
        igd = xd(k.igd); igq = xd(k.igq);

        ct = cos(th); st = sin(th);
        vpd =  VDi*ct + VQi*st;
        vpq = -VDi*st + VQi*ct;

        Pc_ = vcd*igd + vcq*igq;
        Qc_ = vcq*igd - vcd*igq;

        Es = sp.ref.E0 + c.mq*(sp.ref.Qstar - Qm);
        vstar_d = Es - c.Rv*igd + c.Xv*igq;
        vstar_q =    - c.Rv*igq - c.Xv*igd;

        ifref_d = c.Kpv*(vstar_d - vcd) + c.Kiv*xivd + c.kff*(igd - w*c.Bcf*vcq);
        ifref_q = c.Kpv*(vstar_q - vcq) + c.Kiv*xivq + c.kff*(igq + w*c.Bcf*vcd);

        vconv_d = c.Kpi*(ifref_d - ifd) + c.Kii*gamd + vcd - w*c.Xl1*ifq - c.Kad*(ifd - igd);
        vconv_q = c.Kpi*(ifref_q - ifq) + c.Kii*gamq + vcq + w*c.Xl1*ifd - c.Kad*(ifq - igq);
        u = [vconv_d; vconv_q];
        if delay_on
            vapp_d = 2*z(1) - vconv_d;  vapp_q = 2*z(2) - vconv_q;
            fz = [(2/Td)*(vconv_d - z(1)); (2/Td)*(vconv_q - z(2))];
        else
            vapp_d = vconv_d;  vapp_q = vconv_q;
            fz = [0; 0];
        end

        vmid_d = vcd + c.Rd*(ifd - igd);
        vmid_q = vcq + c.Rd*(ifq - igq);
        difd = (wb/c.Xl1)*(vapp_d - vmid_d - c.r1*ifd) + w*wb*ifq;
        difq = (wb/c.Xl1)*(vapp_q - vmid_q - c.r1*ifq) - w*wb*ifd;
        dvcd = (wb/c.Bcf)*(ifd - igd) + w*wb*vcq;
        dvcq = (wb/c.Bcf)*(ifq - igq) - w*wb*vcd;
        digd = (wb/c.Xl2)*(vmid_d - vpd - c.r2*igd) + w*wb*igq;
        digq = (wb/c.Xl2)*(vmid_q - vpq - c.r2*igq) - w*wb*igd;

        fd = zeros(sp.nx, 1);
        if ~isreal(xd) || ~isreal(VDi), fd = complex(fd); end
        fd(k.theta) = (w - 1)*wb;
        fd(k.w)  = (sp.ref.Pstar - Pm - c.Dv*(w - 1)) / (2*c.Hv);
        fd(k.Pm) = c.wc*(Pc_ - Pm);
        fd(k.Qm) = c.wc*(Qc_ - Qm);
        fd(k.xiv_d) = vstar_d - vcd;  fd(k.xiv_q) = vstar_q - vcq;
        fd(k.gam_d) = ifref_d - ifd;  fd(k.gam_q) = ifref_q - ifq;
        fd(k.ifd) = difd; fd(k.ifq) = difq;
        fd(k.vcd) = dvcd; fd(k.vcq) = dvcq;
        fd(k.igd) = digd; fd(k.igq) = digq;

        IDl = igd*ct - igq*st;  IQl = igd*st + igq*ct;
        scl = c.Sn/c.Ssys;
        ID = IDl*scl;  IQ = IQl*scl;
    otherwise
        error('r17_delay:type', 'unknown type %s', type);
end
end

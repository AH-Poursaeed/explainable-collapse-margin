function r16_limiter_straddle
%r16_limiter_straddle  Run the design-ray boundary straddle with a converter current limit.
%   The reference is the published unlimited straddle (e4_straddle.mat).
%
%   Limiter (declared design): circular saturation of the current-reference
%   vector at Imax = 1.2 pu (converter base). This is standard headroom above
%   the ~1 pu operating currents, and the equilibrium margins are printed
%   before any run. The d and q parts are scaled in proportion. Anti-windup
%   uses conditional integration: the outer-loop integrators chi_d/chi_q
%   (GFL) and xiv_d/xiv_q (VSM) freeze while saturated, and the inner
%   current-loop integrator gam integrates the saturated reference error.
%   The ETDRK2 splitting keeps the unlimited equilibrium linearization as
%   its linear part. The limiter is inactive at equilibrium, so the
%   nonlinear remainder carries the saturation. The pulse forcing uses the
%   same structure.
%
%   Machinery: prep/mkcase/pshape/etdrk2/classify/vterm/phis are copied
%   exactly from e4_emt_production.m (the validated ETDRK2 core). The
%   device RHS copies (gfl_rhs_lim / vsm_rhs_lim) are dev.gfl_rhs /
%   dev.gfm_vsm_rhs, unchanged except for the marked limiter block.
%
%   GATES (the run stops if one fails):
%     G1 equivalence: at Imax = Inf the limited RHS equals full.rhs to
%        1e-13 (normalized by rowscale) on 20 perturbed states.
%     G2 straddle reproduction: runs with the limiter inactive at
%        k = 1.96/1.97/1.98 reproduce the published outcomes and divergence
%        onsets (ride-through / collapse 2.174 s / collapse 2.202 s).
%   Measured: the same three k with Imax = 1.2, plus the ringdown of the
%   representative row of the Poorly-damped band (id 10832, 4 s) with the
%   limiter armed.
%
%   Outputs (results/tables/r16/): r16_limiter_straddle.csv,
%   r16_limiter_scanlog.txt, r16_limiter_traces.mat.
%

maxNumCompThreads(2);
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'tables', 'r16');
if ~exist(outdir, 'dir'), mkdir(outdir); end
fcsv = fullfile(outdir, 'r16_limiter_straddle.csv');
flog = fullfile(outdir, 'r16_limiter_scanlog.txt');
fmat = fullfile(outdir, 'r16_limiter_traces.mat');
LOG = fopen(flog, 'a');

IMAX = 1.2;                       % converter-base pu, declared design constant
dv = repmat({'GFL'}, 1, 10);  dv{9} = 'VSM';  dv{10} = 'VSM';
PULSE = struct('bus', 31, 'dG', 0.60, 't0', 0.05, 'dur', 0.020);
DT = 5e-5;
TPOST = PULSE.t0 + PULSE.dur + 0.05;
lg(LOG, '=== r16_limiter_straddle %s ===\n', char(datetime('now')));

% ---- G1: equivalence of the limited RHS at Imax = Inf ------------------------
o = prep(dv, 1.97);
rng(20260807, 'twister');
worst = 0;
for q = 1:20
    xq = o.x0 + 1e-3*randn(size(o.x0));
    d = max(abs((rhs_lim(xq, o.M, Inf) - full.rhs(xq, o.M))./o.M.rowscale));
    worst = max(worst, d);
end
assert(worst < 1e-13, 'G1 FAIL: limited-RHS equivalence %.3g', worst);
lg(LOG, '[G1] PASS: Imax=Inf RHS equivalence, worst scaled |diff| %.2e\n', worst);

% equilibrium current-reference margins (the limiter must be inactive at rest)
mg = iref_ratio(o.x0, o.M);
lg(LOG, '[margins] equilibrium |i_ref| per device (conv pu): max %.4f, all < %.2f: %d\n', ...
    max(mg), IMAX, all(mg < IMAX));
assert(all(mg < IMAX), 'equilibrium current exceeds Imax: premise broken');

% ---- G2 + measured straddle --------------------------------------------------
ks   = [1.96 1.97 1.98];
Tsim = [6.0  4.0  8.0];
pub_out  = {'ride-through', 'collapse', 'collapse'};
pub_tdiv = [NaN 2.174 2.202];

rows = {};
traces = struct('tag', {}, 'k', {}, 'imax', {}, 't', {}, 'Vterm', {}, ...
    'satfrac', {}, 'irefmax', {});
for i = 1:numel(ks)
    o = prep(dv, ks(i));
    for pass = 1:2
        imax = Inf;  if pass == 2, imax = IMAX; end
        cs = mkcase(o, PULSE, PULSE.dG);
        cs.imax = imax;
        tw = tic;
        [tt, XX] = etdrk2(cs, DT, Tsim(i), true);
        wall = toc(tw);
        Vt = vterm(XX, o);
        [oc, tdiv, endd] = classify(tt, Vt, XX, TPOST);
        [satfrac, irefmax, t1sat] = sat_diag(tt, XX, o.M, IMAX);
        lg(LOG, '[k=%.3f imax=%g] %-12s tdiv=%.3f enddev=%.4g satfrac=%.3f irefmax=%.3f t1sat=%.3f (%.0f s wall)\n', ...
            ks(i), imax, oc, tdiv, endd, satfrac, irefmax, t1sat, wall);
        rows{end+1} = struct('run', "straddle", 'k', ks(i), 'id', 0, ...
            'imax', imax, 'sigma_pred', o.S.sigma_dom, 'outcome', string(oc), ...
            'tdiv', tdiv, 'enddev', endd, 'sat_frac_cols', satfrac, ...
            'iref_max_ratio', irefmax, 't_first_sat', t1sat, ...
            'wall_s', wall); %#ok<AGROW>
        traces(end+1) = struct('tag', sprintf('k%.2f_imax%g', ks(i), imax), ...
            'k', ks(i), 'imax', imax, 't', tt, 'Vterm', Vt, ...
            'satfrac', satfrac, 'irefmax', irefmax); %#ok<AGROW>
        if pass == 1
            assert(strcmp(oc, pub_out{i}), ...
                'G2 FAIL k=%.3f: outcome %s vs published %s', ks(i), oc, pub_out{i});
            if isfinite(pub_tdiv(i))
                assert(abs(tdiv - pub_tdiv(i)) < 1e-2, ...
                    'G2 FAIL k=%.3f: tdiv %.4f vs published %.4f', ...
                    ks(i), tdiv, pub_tdiv(i));
            end
            lg(LOG, '[G2] PASS k=%.3f reproduces published %s (tdiv %.3f)\n', ...
                ks(i), pub_out{i}, tdiv);
        end
    end
end

% ---- band ringdown, limiter armed (Poorly damped representative, id 10832) --
T = readtable(fullfile(here, '..', 'data', 'dataset.csv'));
r = T(T.id == 10832, :);
assert(height(r) == 1, 'ringdown rep 10832 not found');
ord = run_scenario(bank.row2cfg(r));
assert(isempty(ord.fail), 'ringdown rep run failed: %s', ord.fail);
lg(LOG, '[ringdown rep] id 10832 sigma %.4f (bank %.4f)\n', ...
    ord.S.sigma_dom, r.sigma_dom);
cs = mkcase(ord, PULSE, PULSE.dG);
cs.imax = IMAX;
tw = tic;
[tt, XX] = etdrk2(cs, DT, 4.0, true);
wall = toc(tw);
Vt = vterm(XX, ord);
[oc, tdiv, endd] = classify(tt, Vt, XX, TPOST);
[satfrac, irefmax, t1sat] = sat_diag(tt, XX, ord.M, IMAX);
lg(LOG, '[ringdown id 10832 imax=%.1f] %-12s enddev=%.4g satfrac=%.3f irefmax=%.3f (%.0f s wall)\n', ...
    IMAX, oc, endd, satfrac, irefmax, wall);
rows{end+1} = struct('run', "ringdown", 'k', r.k, 'id', 10832, ...
    'imax', IMAX, 'sigma_pred', ord.S.sigma_dom, 'outcome', string(oc), ...
    'tdiv', tdiv, 'enddev', endd, 'sat_frac_cols', satfrac, ...
    'iref_max_ratio', irefmax, 't_first_sat', t1sat, 'wall_s', wall);
traces(end+1) = struct('tag', 'ringdown10832', 'k', r.k, 'imax', IMAX, ...
    't', tt, 'Vterm', Vt, 'satfrac', satfrac, 'irefmax', irefmax);

Tb = struct2table([rows{:}]);
writetable(Tb, fcsv);
save(fmat, 'traces', 'PULSE', 'DT', 'IMAX');
lg(LOG, 'r16_limiter_straddle COMPLETE: %d rows -> %s\n', height(Tb), fcsv);
fclose(LOG);
end

function lg(LOG, varargin)
fprintf(1, varargin{:});
if LOG > 0, fprintf(LOG, varargin{:}); end
end

% ============================================================== machinery ====
% prep/mkcase/pshape/classify/vterm/phis/etdrk2: as in e4_emt_production,
% except frhs -> frhs_lim with cs.imax, and rhs_lim replaces full.rhs.
function o = prep(dv, k)
o = run_scenario(struct('devtype', {dv}, 'kL', 1.0, 'k', k));
assert(isempty(o.fail), 'run_scenario failed: %s', o.fail);
end

function cs = mkcase(o, PULSE, dG)
cs.M = o.M;  cs.x0 = o.x0;  cs.A = full(o.Asys);
b = PULSE.bus;
cs.ivb = [o.M.ix_bus(2*b-1); o.M.ix_bus(2*b)];
cs.gfac = o.M.N.wb / o.M.N.Bc(b) * dG;
cs.t0 = PULSE.t0;  cs.dur = PULSE.dur;
cs.imax = Inf;
end

function s = pshape(t, cs)
if t >= cs.t0 && t <= cs.t0 + cs.dur
    s = 0.5*(1 - cos(2*pi*(t - cs.t0)/cs.dur));
else
    s = 0;
end
end

function f = frhs_lim(t, x, cs)
f = rhs_lim(x, cs.M, cs.imax);
s = pshape(t, cs);
if s > 0 && cs.gfac ~= 0
    f(cs.ivb) = f(cs.ivb) - s*cs.gfac*x(cs.ivb);
end
end

function [tt, XX] = etdrk2(cs, dt, T, stopdiv)
n = numel(cs.x0);
[E, P1, P2] = phis(cs.A, dt);
nst = round(T/dt);
keep = max(1, floor(nst/4000));
tt = zeros(1, ceil(nst/keep) + 1);  XX = zeros(n, ceil(nst/keep) + 1);
x = cs.x0;  tt(1) = 0;  XX(:, 1) = x;  m = 1;
x0 = cs.x0;  A = cs.A;
for q = 1:nst
    t = (q-1)*dt;
    dx = x - x0;
    r1 = frhs_lim(t, x, cs) - A*dx;
    a  = x0 + E*dx + dt*(P1*r1);
    r2 = frhs_lim(t + dt, a, cs) - A*(a - x0);
    x  = a + dt*(P2*(r2 - r1));
    if mod(q, keep) == 0
        m = m + 1;  tt(m) = q*dt;  XX(:, m) = x;
        if stopdiv && (any(~isfinite(x)) || max(abs(x - x0)) > 50)
            tt = tt(1:m);  XX = XX(:, 1:m);  return
        end
    end
end
tt = tt(1:m);  XX = XX(:, 1:m);
end

function [oc, tdiv, endd, dip] = classify(tt, Vt, XX, tpost)
dev = max(abs(Vt - Vt(:, 1)), [], 1);
post = tt >= tpost;
dip = max(dev(~post & tt > 0), [], 'omitnan');
if isempty(dip), dip = NaN; end
endd = dev(end);
idx = find(post & dev > 0.5, 1);
bad = any(~isfinite(XX(:, end)));
if ~isempty(idx) || bad
    oc = 'collapse';
    if isempty(idx), tdiv = tt(end); else, tdiv = tt(idx); end
    return
end
tdiv = NaN;
i75 = find(tt >= tt(end)*0.75, 1);
d75 = median(dev(i75:min(i75+20, numel(dev))));
den = median(dev(max(1, numel(dev)-20):end));
if den > 1e-3 && den > 2*max(d75, eps)
    oc = 'escaping';
else
    oc = 'ride-through';
end
end

function Vt = vterm(XX, o)
Vt = zeros(10, size(XX, 2));
for p = 1:10
    b = o.plant.bus(p);
    vd = XX(o.M.ix_bus(2*b-1), :);  vq = XX(o.M.ix_bus(2*b), :);
    Vt(p, :) = sqrt(vd.^2 + vq.^2);
end
end

function [E, P1, P2] = phis(A, dt)
A = full(A);
n = size(A, 1);
W = expm([A*dt, eye(n)*dt, zeros(n); zeros(n), zeros(n), eye(n)*dt; ...
          zeros(n), zeros(n), zeros(n)]);
E  = full(W(1:n, 1:n));
P1 = full(W(1:n, n+1:2*n))/dt;
P2 = full(W(1:n, 2*n+1:3*n))/dt^2;
end

% ================================================== limited RHS (full copy) ==
function f = rhs_lim(x, M, Imax)
%RHS_LIM  full.rhs, unchanged except that the device calls go through the
%   limited device RHS copies. At Imax = Inf it agrees with full.rhs to
%   1e-13, scaled by rowscale (gate G1).
N = M.N;  wb = N.wb;
nbr = N.nbr; nb = N.nbus; nld = N.nload;
ibr = reshape(x(M.ix_br),  2, nbr);
vb  = reshape(x(M.ix_bus), 2, nb);
if nld > 0
    ild = reshape(x(M.ix_load), 2, nld);
else
    ild = zeros(2, 0);
end
f = zeros(size(x));
Iinj = zeros(2, nb);
if ~isreal(x), Iinj = complex(Iinj); end
for i = 1:numel(M.dev)
    d = M.dev(i); b = d.bus;
    switch d.type
        case 'GFL'
            [fd, ID, IQ] = gfl_rhs_lim(x(d.ix), vb(1,b), vb(2,b), d.params, Imax);
        case 'VSM'
            [fd, ID, IQ] = vsm_rhs_lim(x(d.ix), vb(1,b), vb(2,b), d.params, Imax);
    end
    f(d.ix) = fd;
    Iinj(1,b) = Iinj(1,b) + ID;
    Iinj(2,b) = Iinj(2,b) + IQ;
end
fF = N.f; tT = N.t; tap = N.tap;
for e = 1:nbr
    id = ibr(1,e); iq = ibr(2,e);
    vFd = vb(1,fF(e))/tap(e); vFq = vb(2,fF(e))/tap(e);
    vTd = vb(1,tT(e));        vTq = vb(2,tT(e));
    did = (wb/N.x(e))*(vFd - vTd - N.r(e)*id) + wb*iq;
    diq = (wb/N.x(e))*(vFq - vTq - N.r(e)*iq) - wb*id;
    f(M.ix_br(2*e-1)) = did;
    f(M.ix_br(2*e))   = diq;
    Iinj(1,fF(e)) = Iinj(1,fF(e)) - id/tap(e);
    Iinj(2,fF(e)) = Iinj(2,fF(e)) - iq/tap(e);
    Iinj(1,tT(e)) = Iinj(1,tT(e)) + id;
    Iinj(2,tT(e)) = Iinj(2,tT(e)) + iq;
end
for l = 1:nld
    b = N.loadbus(l);
    id = ild(1,l); iq = ild(2,l);
    f(M.ix_load(2*l-1)) = (wb/N.Xs(l))*(vb(1,b) - N.Rs(l)*id) + wb*iq;
    f(M.ix_load(2*l))   = (wb/N.Xs(l))*(vb(2,b) - N.Rs(l)*iq) - wb*id;
    Iinj(1,b) = Iinj(1,b) - id;
    Iinj(2,b) = Iinj(2,b) - iq;
end
if isfield(N, 'ncpl') && N.ncpl > 0
    icp = reshape(x(M.ix_cpl), 2, N.ncpl);
    for l = 1:N.ncpl
        b = N.cplbus(l);
        vd = vb(1,b); vq = vb(2,b);
        v2 = vd*vd + vq*vq;
        ird = (N.Pc(l)*vd + N.Qc(l)*vq)/v2;
        irq = (N.Pc(l)*vq - N.Qc(l)*vd)/v2;
        f(M.ix_cpl(2*l-1)) = (ird - icp(1,l))/N.tauc;
        f(M.ix_cpl(2*l))   = (irq - icp(2,l))/N.tauc;
        Iinj(1,b) = Iinj(1,b) - icp(1,l);
        Iinj(2,b) = Iinj(2,b) - icp(2,l);
    end
end
for b = 1:nb
    vd = vb(1,b); vq = vb(2,b);
    f(M.ix_bus(2*b-1)) = (wb/N.Bc(b))*(Iinj(1,b) - N.G(b)*vd) + wb*vq;
    f(M.ix_bus(2*b))   = (wb/N.Bc(b))*(Iinj(2,b) - N.G(b)*vq) - wb*vd;
end
end

function [fd, ID, IQ] = gfl_rhs_lim(xd, VDi, VQi, sp, Imax)
%GFL_RHS_LIM  dev.gfl_rhs, unchanged, plus the marked limiter block.
k = sp.k; c = sp.cvt; wb = c.wb;
th = xd(k.theta); eps = xd(k.eps_pll); Pm = xd(k.Pm); Qm = xd(k.Qm); %#ok<NASGU>
chid = xd(k.chi_d); chiq = xd(k.chi_q); gamd = xd(k.gam_d); gamq = xd(k.gam_q);
ifd = xd(k.ifd); ifq = xd(k.ifq); vcd = xd(k.vcd); vcq = xd(k.vcq);
igd = xd(k.igd); igq = xd(k.igq);
ct = cos(th); st = sin(th);
vpd =  VDi*ct + VQi*st;
vpq = -VDi*st + VQi*ct;
wpll = 1 + c.pll_Kp*vpq + c.pll_Ki*xd(k.eps_pll);
Ppcc = vpd*igd + vpq*igq;
Qpcc = vpq*igd - vpd*igq;
ifref_d =   c.Kpp*(sp.ref.Pstar - Pm) + c.Kip*chid;
ifref_q = -(c.Kpq*(sp.ref.Qstar - Qm) + c.Kiq*chiq);
% ---- LIMITER: circular saturation + conditional anti-windup ----------------
sat = false;
imag2 = sqrt(ifref_d^2 + ifref_q^2);
if imag2 > Imax
    sc = Imax/imag2;
    ifref_d = ifref_d*sc;  ifref_q = ifref_q*sc;
    sat = true;
end
% ----------------------------------------------------------------------------
vconv_d = c.Kpi*(ifref_d - ifd) + c.Kii*gamd + vcd - wpll*c.Xl1*ifq - c.Kad*(ifd - igd);
vconv_q = c.Kpi*(ifref_q - ifq) + c.Kii*gamq + vcq + wpll*c.Xl1*ifd - c.Kad*(ifq - igq);
vmid_d = vcd + c.Rd*(ifd - igd);
vmid_q = vcq + c.Rd*(ifq - igq);
difd = (wb/c.Xl1)*(vconv_d - vmid_d - c.r1*ifd) + wpll*wb*ifq;
difq = (wb/c.Xl1)*(vconv_q - vmid_q - c.r1*ifq) - wpll*wb*ifd;
dvcd = (wb/c.Bcf)*(ifd - igd) + wpll*wb*vcq;
dvcq = (wb/c.Bcf)*(ifq - igq) - wpll*wb*vcd;
digd = (wb/c.Xl2)*(vmid_d - vpd - c.r2*igd) + wpll*wb*igq;
digq = (wb/c.Xl2)*(vmid_q - vpq - c.r2*igq) - wpll*wb*igd;
fd = zeros(sp.nx, 1);
fd(k.theta)   = (wpll - 1)*wb;
fd(k.eps_pll) = vpq;
fd(k.Pm)      = c.wc*(Ppcc - Pm);
fd(k.Qm)      = c.wc*(Qpcc - Qm);
if sat
    fd(k.chi_d) = 0;                 % conditional integration (anti-windup)
    fd(k.chi_q) = 0;
else
    fd(k.chi_d) = sp.ref.Pstar - Pm;
    fd(k.chi_q) = sp.ref.Qstar - Qm;
end
fd(k.gam_d)   = ifref_d - ifd;       % integrates the (saturated) reference error
fd(k.gam_q)   = ifref_q - ifq;
fd(k.ifd) = difd; fd(k.ifq) = difq;
fd(k.vcd) = dvcd; fd(k.vcq) = dvcq;
fd(k.igd) = digd; fd(k.igq) = digq;
IDl = igd*ct - igq*st;  IQl = igd*st + igq*ct;
scl = c.Sn/c.Ssys;
ID = IDl*scl;  IQ = IQl*scl;
end

function [fd, ID, IQ] = vsm_rhs_lim(xd, VDi, VQi, sp, Imax)
%VSM_RHS_LIM  dev.gfm_vsm_rhs, unchanged, plus the marked limiter block.
k = sp.k; c = sp.cvt; wb = c.wb;
th = xd(k.theta); w = xd(k.w); Pm = xd(k.Pm); Qm = xd(k.Qm);
xivd = xd(k.xiv_d); xivq = xd(k.xiv_q); gamd = xd(k.gam_d); gamq = xd(k.gam_q);
ifd = xd(k.ifd); ifq = xd(k.ifq); vcd = xd(k.vcd); vcq = xd(k.vcq);
igd = xd(k.igd); igq = xd(k.igq);
ct = cos(th); st = sin(th);
vpd =  VDi*ct + VQi*st;
vpq = -VDi*st + VQi*ct;
P = vcd*igd + vcq*igq;
Q = vcq*igd - vcd*igq;
Es = sp.ref.E0 + c.mq*(sp.ref.Qstar - Qm);
vstar_d = Es - c.Rv*igd + c.Xv*igq;
vstar_q =    - c.Rv*igq - c.Xv*igd;
ifref_d = c.Kpv*(vstar_d - vcd) + c.Kiv*xivd + c.kff*(igd - w*c.Bcf*vcq);
ifref_q = c.Kpv*(vstar_q - vcq) + c.Kiv*xivq + c.kff*(igq + w*c.Bcf*vcd);
% ---- LIMITER: circular saturation + conditional anti-windup ----------------
sat = false;
imag2 = sqrt(ifref_d^2 + ifref_q^2);
if imag2 > Imax
    sc = Imax/imag2;
    ifref_d = ifref_d*sc;  ifref_q = ifref_q*sc;
    sat = true;
end
% ----------------------------------------------------------------------------
vconv_d = c.Kpi*(ifref_d - ifd) + c.Kii*gamd + vcd - w*c.Xl1*ifq - c.Kad*(ifd - igd);
vconv_q = c.Kpi*(ifref_q - ifq) + c.Kii*gamq + vcq + w*c.Xl1*ifd - c.Kad*(ifq - igq);
vmid_d = vcd + c.Rd*(ifd - igd);
vmid_q = vcq + c.Rd*(ifq - igq);
difd = (wb/c.Xl1)*(vconv_d - vmid_d - c.r1*ifd) + w*wb*ifq;
difq = (wb/c.Xl1)*(vconv_q - vmid_q - c.r1*ifq) - w*wb*ifd;
dvcd = (wb/c.Bcf)*(ifd - igd) + w*wb*vcq;
dvcq = (wb/c.Bcf)*(ifq - igq) - w*wb*vcd;
digd = (wb/c.Xl2)*(vmid_d - vpd - c.r2*igd) + w*wb*igq;
digq = (wb/c.Xl2)*(vmid_q - vpq - c.r2*igq) - w*wb*igd;
fd = zeros(sp.nx, 1);
fd(k.theta) = (w - 1)*wb;
fd(k.w)  = (sp.ref.Pstar - Pm - c.Dv*(w - 1)) / (2*c.Hv);
fd(k.Pm) = c.wc*(P - Pm);
fd(k.Qm) = c.wc*(Q - Qm);
if sat
    fd(k.xiv_d) = 0;                 % conditional integration (anti-windup)
    fd(k.xiv_q) = 0;
else
    fd(k.xiv_d) = vstar_d - vcd;
    fd(k.xiv_q) = vstar_q - vcq;
end
fd(k.gam_d) = ifref_d - ifd;
fd(k.gam_q) = ifref_q - ifq;
fd(k.ifd) = difd; fd(k.ifq) = difq;
fd(k.vcd) = dvcd; fd(k.vcq) = dvcq;
fd(k.igd) = digd; fd(k.igq) = digq;
IDl = igd*ct - igq*st;  IQl = igd*st + igq*ct;
scl = c.Sn/c.Ssys;
ID = IDl*scl;  IQ = IQl*scl;
end

% ============================================================ diagnostics ====
function mg = iref_ratio(x, M)
%IREF_RATIO  |i_ref| magnitude (converter pu) per device at one state.
mg = zeros(numel(M.dev), 1);
for i = 1:numel(M.dev)
    d = M.dev(i);  sp = d.params;  k = sp.k;  c = sp.cvt;
    xd = x(d.ix);
    igd = xd(k.igd); igq = xd(k.igq);
    vcd = xd(k.vcd); vcq = xd(k.vcq);
    if strcmp(d.type, 'GFL')
        Pm = xd(k.Pm); Qm = xd(k.Qm);
        ifref_d =   c.Kpp*(sp.ref.Pstar - Pm) + c.Kip*xd(k.chi_d);
        ifref_q = -(c.Kpq*(sp.ref.Qstar - Qm) + c.Kiq*xd(k.chi_q));
    else
        w = xd(k.w); Qm = xd(k.Qm);
        Es = sp.ref.E0 + c.mq*(sp.ref.Qstar - Qm);
        vstar_d = Es - c.Rv*igd + c.Xv*igq;
        vstar_q =    - c.Rv*igq - c.Xv*igd;
        ifref_d = c.Kpv*(vstar_d - vcd) + c.Kiv*xd(k.xiv_d) ...
            + c.kff*(igd - w*c.Bcf*vcq);
        ifref_q = c.Kpv*(vstar_q - vcq) + c.Kiv*xd(k.xiv_q) ...
            + c.kff*(igq + w*c.Bcf*vcd);
    end
    mg(i) = sqrt(ifref_d^2 + ifref_q^2);
end
end

function [satfrac, irefmax, t1sat] = sat_diag(tt, XX, M, Imax)
%SAT_DIAG  Saturation statistics over the kept trace columns.
nc = size(XX, 2);
anysat = false(1, nc);
irefmax = 0;
for q = 1:nc
    mg = iref_ratio(XX(:, q), M);
    irefmax = max(irefmax, max(mg));
    anysat(q) = any(mg > Imax);
end
satfrac = mean(anysat);
i1 = find(anysat, 1);
if isempty(i1), t1sat = NaN; else, t1sat = tt(i1); end
end

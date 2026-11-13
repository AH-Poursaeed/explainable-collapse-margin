function PL = pencil_lossy(R, P, mpc, res, plant, opts)
%PENCIL_LOSSY  The reduced interconnection with the network losses kept.
%   The network's RESISTIVE parts and load conductances are RETAINED. It is
%   a lossy B-and-G reduced model. Its r -> 0, G -> 0 limit (opts.lossless)
%   is the lossless B x F form, the form of the susceptance index (build_B,
%   reduced.chain).
%
%   Construction:
%     1. Complex nodal admittance of the NETWORK ONLY at the solved
%        operating point: makeYbus (branch r, x, charging, taps, declared
%        parasitic BS) + constant-Z load admittances conj(S)/|V|^2.
%     2. Exact complex Kron reduction to the 10 converter buses:
%        Y_red = Ycc - Yci Yii^{-1} Yic (10x10 complex).
%     3. Dynamic realization on the reduced graph (rotating frame):
%        - pair branch (i, j): y = -Y_red(i, j) -> z = 1/y = r_s + j x_s,
%          realized as an RL branch, d i/dt = (wb/x_s)(V_i - V_j - r_s i)
%          - wb J2 i. A negative x_s is allowed: the equivalent is
%          mathematical, not hardware. A near-zero x_s becomes an algebraic
%          conductance instead.
%        - bus shunt: y_sh,i = sum_j Y_red(i, j). The capacitive part
%          b_cap = max(Im y_sh, Bc_par) becomes the bus capacitance (every
%          physical converter bus carries the declared parasitic C, so
%          b_cap > 0). The rest, y_sh - j b_cap (inductive/resistive),
%          becomes a series-RL shunt leg, or an algebraic conductance if it
%          is near-real.
%     4. Devices exactly as reduced.pencil_model: full GFL port state
%        spaces; the reduced GFM (inner Y_dq = -F/(w0 Leff),
%        2 states, + synchronization loop, 3 states). S_b at the interface.
%     5. Bus KCL is DYNAMIC (capacitor): pure ODE, plain eig, no descriptor.
%
%   Gate: the phasor admittance (s = 0, the fundamental), rebuilt from the
%   realization, equals Y_red within 1e-8 (asserted).
%
%   Returns eigenvalues, the sub-synchronous dominant mode and zeta_min by
%   the same conventions as full.modal.

if nargin < 6, opts = struct(); end
gfm_full = isfield(opts, 'gfm_full') && opts.gfm_full;   % full 14-state VSM ports
lossless = isfield(opts, 'lossless') && opts.lossless;   % r -> 0, load G -> 0

define_constants;
wb = P.wb;
J2 = [0 -1; 1 0];
isG = strcmp(R.plant_type, 'GFL');
cb = plant.bus(:);
nbusr = numel(cb);

% ---- full VSM ports on demand (attribution variant) --------------------------
ports = R.ports;
if gfm_full
    V = res.bus(:, VM).*exp(1j*res.bus(:, VA)*pi/180);
    Sg = zeros(nbusr, 1) + 0j;
    for gi = 1:size(res.gen, 1)
        pb = find(plant.bus == res.gen(gi, GEN_BUS), 1);
        if ~isempty(pb)
            Sg(pb) = Sg(pb) + (res.gen(gi, PG) + 1j*res.gen(gi, QG))/mpc.baseMVA;
        end
    end
    for i = 1:nbusr
        if ~isG(i)
            c = ports{i}.cvt;
            sp = struct('cvt', c, 'k', dev.state_map('VSM'), 'nx', c.nx, 'Ssys', c.Sn);
            Sconv = Sg(i)*mpc.baseMVA/plant.Sn(i);
            prtf = reduced.port('VSM', sp, V(plant.bus(i)), Sconv, false);
            prtf.cvt = c;
            ports{i} = prtf;
        end
    end
end

% ---- 1/2. complex Kron reduction of the (lossy) network ----------------------
mpcn = mpc;
if lossless, mpcn.branch(:, BR_R) = 0; end
Ybus = full(makeYbus(mpcn));
Vm = res.bus(:, VM);
Sload = (res.bus(:, PD) + 1j*res.bus(:, QD))/mpc.baseMVA;
yload = conj(Sload)./Vm.^2;
if lossless, yload = 1j*imag(yload); end
Yn = Ybus + diag(yload);
nb = size(Yn, 1);
intr = setdiff((1:nb)', cb);
Yr = Yn(cb, cb) - Yn(cb, intr)*(Yn(intr, intr)\Yn(intr, cb));

% ---- 3. dynamic realization ---------------------------------------------------
XTOL = 1e-7;                 % |x_s| below this: algebraic conductance
YTOL = 1e-9;                 % couplings below this: dropped (checked by the gate)
pairs = [];                  % [i, j, r_s, x_s]
galg = zeros(nbusr);         % algebraic conductance couplings (symmetric)
for i = 1:nbusr
    for j = i+1:nbusr
        y = -Yr(i, j);
        if abs(y) < YTOL, continue; end
        z = 1/y;
        if abs(imag(z)) > XTOL
            pairs(end+1, :) = [i, j, real(z), imag(z)];   %#ok<AGROW>
        else
            galg(i, j) = real(y);  galg(j, i) = real(y);
        end
    end
end
legs = [];                   % [i, r_s, x_s]
gsh = zeros(nbusr, 1);       % algebraic shunt conductances
bcap = zeros(nbusr, 1);
for i = 1:nbusr
    ysh = sum(Yr(i, :));
    bcap(i) = max(imag(ysh), P.Bc_par);
    yleg = ysh - 1j*bcap(i);
    if abs(yleg) < YTOL, continue; end
    z = 1/yleg;
    if abs(imag(z)) > XTOL
        legs(end+1, :) = [i, real(z), imag(z)];           %#ok<AGROW>
    else
        gsh(i) = gsh(i) + real(yleg);
    end
end

% ---- phasor-exactness gate -----------------------------------------------------
Ychk = zeros(nbusr) + 0j;
for q = 1:size(pairs, 1)
    i = pairs(q,1); j = pairs(q,2);
    y = 1/(pairs(q,3) + 1j*pairs(q,4));
    Ychk(i,i) = Ychk(i,i) + y;  Ychk(j,j) = Ychk(j,j) + y;
    Ychk(i,j) = Ychk(i,j) - y;  Ychk(j,i) = Ychk(j,i) - y;
end
Ychk = Ychk + galg.*(-1);                       % off-diagonal -g couplings
Ychk = Ychk + diag(sum(galg, 2));               % their diagonal counterparts
for q = 1:size(legs, 1)
    i = legs(q,1);
    Ychk(i,i) = Ychk(i,i) + 1/(legs(q,2) + 1j*legs(q,3));
end
Ychk = Ychk + diag(gsh + 1j*bcap);
PL.net_gate = max(abs(Ychk - Yr), [], 'all');
assert(PL.net_gate < 1e-8, 'pencil_lossy: phasor realisation gate failed (%.2e)', PL.net_gate);

% ---- 4/5. state assembly --------------------------------------------------------
ndev = zeros(nbusr, 1);
for i = 1:nbusr
    if isG(i) || gfm_full, ndev(i) = size(ports{i}.A, 1); else, ndev(i) = 5; end
end
dev0 = [0; cumsum(ndev)];
nxd = dev0(end);
np2 = size(pairs, 1);
nl2 = size(legs, 1);
ipair0 = nxd;
ileg0  = nxd + 2*np2;
ibus0  = nxd + 2*np2 + 2*nl2;
N = ibus0 + 2*nbusr;
A = zeros(N);
iV = @(b) ibus0 + (2*b-1:2*b);

for i = 1:nbusr
    ii = dev0(i) + (1:ndev(i));
    vv = iV(i);
    prt = ports{i};
    if isG(i) || gfm_full
        A(ii, ii) = prt.A;
        A(ii, vv) = prt.B;
    else
        c = prt.cvt;  Leff = R.Leff(i);
        v0 = prt.v0;  i0v = prt.i0;
        Jv = 2*c.Hv/wb;  Dv = c.Dv/wb;
        Rr = reduced.rot2(prt.th0);
        ixi = ii(1:2); iPf = ii(3); iom = ii(4); ith = ii(5);
        A(ixi, ixi) = -wb*J2;
        A(ixi, vv)  = (1/Leff)*Rr.';
        A(ixi, ith) = -(J2*v0)/Leff;
        A(iPf, ixi) = -(v0.')/c.TP * c.kP;
        A(iPf, vv)  = (c.kP/c.TP)*(i0v.'*Rr.');
        A(iPf, ith) = (c.kP/c.TP)*(v0.'*(J2*i0v));
        A(iPf, iPf) = -1/c.TP;
        A(iom, iPf) = -1/Jv;
        A(iom, iom) = -Dv/Jv;
        A(ith, iom) = 1;
    end
end
% pair branches
for q = 1:np2
    i = pairs(q,1); j = pairs(q,2); rs = pairs(q,3); xs = pairs(q,4);
    ib = ipair0 + (2*q-1:2*q);
    A(ib, ib) = -(wb*rs/xs)*eye(2) - wb*J2;
    A(ib, iV(i)) =  (wb/xs)*eye(2);
    A(ib, iV(j)) = -(wb/xs)*eye(2);
end
% shunt legs
for q = 1:nl2
    i = legs(q,1); rs = legs(q,2); xs = legs(q,3);
    ib = ileg0 + (2*q-1:2*q);
    A(ib, ib) = -(wb*rs/xs)*eye(2) - wb*J2;
    A(ib, iV(i)) = (wb/xs)*eye(2);
end
% bus capacitors: dV_i/dt = (wb/bcap_i) * I_in,i - wb J2 V_i
for i = 1:nbusr
    vv = iV(i);
    sc = wb/bcap(i);
    % device injection (system base)
    ii = dev0(i) + (1:ndev(i));
    prt = ports{i};
    if isG(i) || gfm_full
        A(vv, ii) = A(vv, ii) + sc*R.BB.Sb(i)*prt.C;
        A(vv, vv) = A(vv, vv) + sc*R.BB.Sb(i)*prt.D;
    else
        i0v = prt.i0;
        Rr = reduced.rot2(prt.th0);
        A(vv, dev0(i) + (1:2)) = A(vv, dev0(i) + (1:2)) - sc*R.BB.Sb(i)*Rr;
        A(vv, dev0(i) + 5)     = A(vv, dev0(i) + 5)     + sc*R.BB.Sb(i)*Rr*(J2*i0v);
    end
    % algebraic conductances
    A(vv, vv) = A(vv, vv) - sc*(gsh(i) + sum(galg(i, :)))*eye(2);
    for j = 1:nbusr
        if galg(i, j) ~= 0
            A(vv, iV(j)) = A(vv, iV(j)) + sc*galg(i, j)*eye(2);
        end
    end
    % rotating-frame capacitor cross term
    A(vv, vv) = A(vv, vv) - wb*J2;
end
% pair-branch currents into the buses
for q = 1:np2
    i = pairs(q,1); j = pairs(q,2);
    ib = ipair0 + (2*q-1:2*q);
    A(iV(i), ib) = A(iV(i), ib) - (wb/bcap(i))*eye(2);
    A(iV(j), ib) = A(iV(j), ib) + (wb/bcap(j))*eye(2);
end
% leg currents out of their buses
for q = 1:nl2
    i = legs(q,1);
    ib = ileg0 + (2*q-1:2*q);
    A(iV(i), ib) = A(iV(i), ib) - (wb/bcap(i))*eye(2);
end

% ---- eigen extraction (full.modal conventions) ----------------------------------
PL.A = A;                    % realization matrix (participation diagnostics)
lam = eig(A);
PL.lam = lam;
PL.N = N;
nearzero = abs(lam) < 1e-6;
PL.nzero = nnz(nearzero);
inb = ~nearzero & (abs(imag(lam)) <= 2*pi*55);
lamB = lam(inb);
[PL.sigma_dom, im] = max(real(lamB));
PL.lam_dom = lamB(im);
zet = -real(lamB)./max(abs(lamB), eps);
PL.zeta_min = min(zet);
PL.oob_maxRe = max([real(lam(~nearzero & ~inb)); -inf]);
PL.npairs = np2;  PL.nlegs = nl2;  PL.bcap = bcap;
end

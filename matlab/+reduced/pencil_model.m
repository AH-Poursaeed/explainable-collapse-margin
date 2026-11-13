function PM = pencil_model(R, P)
%PENCIL_MODEL  The reduced model as a descriptor pencil.
%
%   Assembles the reduced system as an explicit DAE and
%   returns its finite eigenvalues (QZ). This is the reduced spectrum, free
%   of the extra zeros that a raw scalar-determinant search lets in.
%
%   Blocks (all system-base at the interface, converter-base internally):
%     GFL i : the full port state space (A_i, B_i, C_i, D_i), local->common
%             frames already absorbed; injection scaled by S_b,i.
%     GFM j : the reduced device: inner Y_dq = -F/(w0 Leff_j)
%             (2 states) + synchronization loop (theta, omega,
%             power filter: 3 states), local frame, rotated to common.
%     Net   : Btilde_red x F(s) realized per bus pair:
%             d iL_b/dt = -w0 J2 iL_b + w0 V_b,  I_net = (Btilde_red (x) I2) iL.
%     KCL   : algebraic, sum_dev I_dev,b - I_net,b = 0 (V algebraic).
%
%   E dz/dt = A z, E = blkdiag(I_nx, 0_nV); eigenvalues from eig(A, E),
%   infinite ones discarded.

BB = R.BB;  wb = P.wb;
J2 = [0 -1; 1 0];
isG = strcmp(R.plant_type, 'GFL');
nbusr = 10;

% ---- index bookkeeping -------------------------------------------------------
nx = 0;
dev = struct('i0', {}, 'n', {}, 'type', {});
for i = 1:nbusr
    if isG(i), n_i = size(R.ports{i}.A, 1); else, n_i = 5; end
    dev(i) = struct('i0', nx, 'n', n_i, 'type', R.plant_type{i});
    nx = nx + n_i;
end
inet0 = nx;  nx = nx + 2*nbusr;          % network iL states
nV = 2*nbusr;                             % algebraic bus voltages
N = nx + nV;
iV = nx + (1:nV);

A = zeros(N);  E = zeros(N);
E(1:nx, 1:nx) = eye(nx);

for i = 1:nbusr
    ii = dev(i).i0 + (1:dev(i).n);
    vv = iV(2*i-1:2*i);
    if isG(i)
        prt = R.ports{i};
        A(ii, ii) = prt.A;
        A(ii, vv) = prt.B;
        % KCL rows get S_b * (C x + D V) later
    else
        prt = R.ports{i};
        c = prt.cvt;
        Leff = R.Leff(i);
        v0 = prt.v0;  i0v = prt.i0;
        Jv = 2*c.Hv/wb;  Dv = c.Dv/wb;
        Rr = reduced.rot2(prt.th0);
        % local states: xi(2), Pf(1), om(1), th(1)
        ixi = ii(1:2); iPf = ii(3); iom = ii(4); ith = ii(5);
        % dxi = -w0 J2 xi + (dV_loc - J2 v0 th)/Leff
        A(ixi, ixi) = -wb*J2;
        A(ixi, vv)  = (1/Leff)*Rr.';            % dV_loc = Rr' dV_common
        A(ixi, ith) = -(J2*v0)/Leff;
        % dI_loc = -xi + J2 i0 th ;  dPe = i0' dV_loc + v0' dI_loc
        % dPf = (kP dPe - Pf)/TP
        A(iPf, ixi) = -(v0.')/c.TP * c.kP;      % v0' * (-xi)
        A(iPf, vv)  = (c.kP/c.TP)*(i0v.'*Rr.');
        A(iPf, ith) = (c.kP/c.TP)*(v0.'*(J2*i0v));
        A(iPf, iPf) = -1/c.TP;
        % Jv dom = -Pf - Dv om   (Delta theta = -G dPe realization)
        A(iom, iPf) = -1/Jv;
        A(iom, iom) = -Dv/Jv;
        % dth = om
        A(ith, iom) = 1;
    end
end
% network iL: d iL = -w0 J2 iL + w0 V
for b = 1:nbusr
    ib = inet0 + (2*b-1:2*b);
    A(ib, ib) = -wb*J2;
    A(ib, iV(2*b-1:2*b)) = wb*eye(2);
end
% KCL rows (algebraic): sum_dev I_dev,b - I_net,b = 0
Bnet = kron(BB.Btilde_red, eye(2));
for b = 1:nbusr
    rows = iV(2*b-1:2*b);
    i = b;                                  % one device per reduced bus
    ii = dev(i).i0 + (1:dev(i).n);
    if isG(i)
        prt = R.ports{i};
        A(rows, ii) = BB.Sb(i)*prt.C;
        A(rows, rows) = A(rows, rows) + BB.Sb(i)*prt.D;
    else
        prt = R.ports{i};
        i0v = prt.i0;
        Rr = reduced.rot2(prt.th0);
        % I_common = Rr * ( -xi + J2 i0 th )
        A(rows, dev(i).i0 + (1:2)) = -BB.Sb(i)*Rr;
        A(rows, dev(i).i0 + 5)     =  BB.Sb(i)*Rr*(J2*i0v);
    end
    A(rows, inet0 + (1:2*nbusr)) = A(rows, inet0 + (1:2*nbusr)) - Bnet(2*b-1:2*b, :);
end

lam = eig(A, E);
% descriptor spectra carry near-infinite generalized eigenvalues from the
% algebraic rows. isfinite alone keeps O(1e20) artifacts, which would then
% spoil sigma_dom (the same |lam| < 1e5 filter as subsystem_modes)
lam = lam(isfinite(lam) & abs(lam) < 1e5);
PM.lam = lam;
PM.nx = nx;  PM.nV = nV;
% dominant over the sub-synchronous band, matching full.modal's convention
inb = abs(imag(lam)) <= 2*pi*55 & abs(lam) > 1e-7;
lamB = lam(inb);
[PM.sigma_dom, im] = max(real(lamB));
PM.lam_dom = lamB(im);
zet = -real(lamB)./abs(lamB);
PM.zeta_min = min(zet);
PM.nzero = nnz(abs(lam) <= 1e-7);
end

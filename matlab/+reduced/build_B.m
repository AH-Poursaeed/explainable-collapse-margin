function BB = build_B(mpc, res, plant, bin_conv)
%BUILD_B  Susceptance Laplacian, GFM stiffening, Schur reduction and stiffness matrix A.
%
%   B_lap = -imag(Ybus of the LOSSLESS case), with the loads absorbed as
%   their susceptance at the PF voltages. Branch b = 1/x (taps kept). Line
%   charging and capacitive shunts give negative diagonal grounding, and
%   inductive loads give positive diagonal grounding. Net inductive shunt
%   grounding is the lambda_min > 0 condition.
%
%   bin_conv : 10x1 converter-base stiffening susceptance 1/(w0 Leff) for
%              GFM plants (0 for GFL plants), from the port identification.
%
%   Returns BB with: Blap (39x39), Bin_sys (10x10 diag over plants), the
%   converter/interior partition, Btilde_red (the Schur complement),
%   Bkron (B_in = 0 Schur), A, lam (sorted eig), lam_min, phi_min, p, and
%   S_b (system pu).

define_constants;
% lossless copy: r -> 0, everything else as the scenario case
mpcl = mpc;
mpcl.branch(:, BR_R) = 0;
Ybl = makeYbus(mpcl.baseMVA, mpcl.bus, mpcl.branch);   % includes BS shunts
Blap = -imag(full(Ybl));

% loads absorbed as susceptance at the PF voltages (imag part only; the
% conductance part belongs to the lossy model, not the susceptance form)
Vm = res.bus(:, VM);
Qd = res.bus(:, QD)/mpc.baseMVA;
nb = size(mpc.bus, 1);
for b = 1:nb
    Blap(b, b) = Blap(b, b) + Qd(b)/Vm(b)^2;    % inductive +, capacitive -
end

% converter buses and ratings
cb = plant.bus(:);
Sb = plant.Sn(:)/mpc.baseMVA;                    % system pu
Bin_sys = diag(Sb .* bin_conv(:));               % (S_i/S_B) ||Y_i(0)|| as in (9): system base

% unreduced modified susceptance: add B_in at the converter diagonal
Btl = Blap;
Btl(sub2ind([nb nb], cb, cb)) = Btl(sub2ind([nb nb], cb, cb)) + diag(Bin_sys);

intr = setdiff((1:nb)', cb);
B4 = Btl(cb, cb);   B3 = Btl(cb, intr);
B2 = Btl(intr, cb); B1 = Btl(intr, intr);
BB.Btilde_red = B4 - B3*(B1\B2);                 % the Schur complement

% B_in = 0 counterpart (for the Weyl check and gSCR limit)
B4k = Blap(cb, cb);  B3k = Blap(cb, intr);
B2k = Blap(intr, cb); B1k = Blap(intr, intr);
BB.Bkron = B4k - B3k*(B1k\B2k);

% pass-through identity residual (Schur commutes with the diagonal add)
BB.passthrough_resid = max(abs(BB.Btilde_red - (BB.Bkron + Bin_sys)), [], 'all');

% ---- stiffness matrix ----------------------------------------------
symf = @(Mx) (Mx + Mx.')/2;
Sih = diag(1./sqrt(Sb));
BB.sym_noop_resid = max(abs(symf(BB.Btilde_red) - BB.Btilde_red), [], 'all');
A = Sih*symf(BB.Btilde_red)*Sih;
BB.A = A;
[Phi, Dg] = eig((A + A.')/2);
[lam, o] = sort(diag(Dg));
Phi = Phi(:, o);
BB.lam = lam;
BB.lam_min = lam(1);
BB.phi_min = Phi(:, 1);
BB.p = BB.phi_min.^2 / sum(BB.phi_min.^2);       % participations
BB.Phi = Phi;
BB.Sb = Sb;  BB.cb = cb;  BB.Bin_sys = Bin_sys;  BB.Blap = Blap;
% simplicity of lambda_min (an assumption of the first-order drift bound)
if numel(lam) > 1, BB.lam_gap = lam(2) - lam(1); else, BB.lam_gap = inf; end
end

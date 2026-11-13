function N = build_network(mpc, res, P)
%BUILD_NETWORK  Dynamic dq network structure from a solved MATPOWER case.
%
%   Every in-service branch becomes a series RL dq state pair (tap-aware,
%   from-side tap, no phase shift). Every bus becomes a shunt-C dq state pair
%   collecting line charging (from-side /t^2), capacitive load parts and a
%   parasitic floor P.Bc_par. Constant-impedance loads become a series RL
%   dq state pair or a conductance G (algebraic in the bus-C equation); the
%   load section below gives the rule.
%
%   Steady state reproduces the phasor equations exactly:
%     (r + jx) I = VF/t - VT ;  sum(I_in) = j Bc V ;  (j XL) I_L = V.

define_constants;
on = mpc.branch(:, BR_STATUS) == 1;
br = mpc.branch(on, :);
nb = size(mpc.bus, 1);

% internal bus numbering = row index (case39 is consecutive 1..39; assert)
assert(all(mpc.bus(:, BUS_I) == (1:nb)'), 'net:numbering', ...
    'bus numbering must be consecutive');

N.nbus = nb;
N.nbr  = size(br, 1);
N.f    = br(:, F_BUS);
N.t    = br(:, T_BUS);
N.r    = br(:, BR_R);
N.x    = br(:, BR_X);
tap    = br(:, TAP); tap(tap == 0) = 1;
N.tap  = tap;
assert(all(br(:, SHIFT) == 0), 'net:shift', 'phase shifters not supported');

% ---- bus shunt capacitance [pu susceptance] --------------------------------
Bc = zeros(nb, 1);
for e = 1:N.nbr
    Bc(N.f(e)) = Bc(N.f(e)) + br(e, BR_B)/2 / tap(e)^2;   % from-side charging
    Bc(N.t(e)) = Bc(N.t(e)) + br(e, BR_B)/2;
end
% MATPOWER bus shunts (case39 has none, keep general)
Bc = Bc + mpc.bus(:, BS)/mpc.baseMVA;
Gsh = mpc.bus(:, GS)/mpc.baseMVA;

% ---- constant-impedance loads at the PF voltages ---------------------------
% Inductive loads (P > 0, Q > 0): SERIES RL branch to ground,
%   Z_s = R_s + jX_s = |V|^2 (P + jQ)/(P^2 + Q^2)
% At the fundamental this is identical to the constant-Z load. Its
% stationary-frame DC mode is damped at sigma = -(P/Q) wb. A parallel R||L
% would not work: its L leg is undamped at DC and adds false +-j w0 modes
% to zeta_min.
% Capacitive-Q loads: conductance at the bus + capacitance into Bc.
% Resistive-only loads: conductance at the bus.
Vm = res.bus(:, VM);
Pd = res.bus(:, PD)/mpc.baseMVA;
Qd = res.bus(:, QD)/mpc.baseMVA;
Gbus = Gsh;
loadbus = []; Rs = []; Xs = [];
for b = 1:nb
    if Qd(b) > 1e-9 && Pd(b) > 1e-9
        S2 = Pd(b)^2 + Qd(b)^2;
        loadbus(end+1,1) = b;                          %#ok<AGROW>
        Rs(end+1,1) = Vm(b)^2*Pd(b)/S2;                %#ok<AGROW>
        Xs(end+1,1) = Vm(b)^2*Qd(b)/S2;                %#ok<AGROW>
    else
        Gbus(b) = Gbus(b) + Pd(b)/Vm(b)^2;
        if abs(Qd(b)) > 1e-9
            Bc(b) = Bc(b) - Qd(b)/Vm(b)^2;             % capacitive -> bus C
        end
    end
end
N.G = Gbus;

% ---- every bus voltage must be a state: the parasitic floor is declared in
% the CASE (build_case adds BS = Bc_par at every bus), so PF and dynamics see
% the same network. Here the code only checks that it is there.
assert(all(Bc >= 0), 'net:negC', 'negative bus capacitance: add inductive-shunt handling');
assert(all(Bc >= 0.9*P.Bc_par), 'net:floor', ...
    'bus with no capacitance: build_case must declare the parasitic shunt');
N.Bc = Bc;

% ---- CPL split -----------------------------------
% Optional P.cpl = struct('alpha', a, 'tau', tau_s): fraction a of every
% series-RL (P > 0, Q > 0) load is re-typed as a smoothed constant-power
% sink (the converter-interfaced load class). The POWER FLOW, and so the
% equilibrium, stays the same: the RL leg is rescaled to carry (1 - a) of
% the load, and the CPL leg draws a*(P + jQ) exactly at the PF voltage.
% Default (no P.cpl field): ncpl = 0 and the model is bit-identical to the
% published one.
N.ncpl = 0;  N.cplbus = [];  N.Pc = [];  N.Qc = [];  N.tauc = NaN;
if isfield(P, 'cpl') && ~isempty(P.cpl) && P.cpl.alpha > 0
    a = P.cpl.alpha;
    assert(a > 0 && a <= 1, 'net:cpl', 'cpl.alpha must be in (0, 1]');
    N.tauc = P.cpl.tau;
    assert(N.tauc > 0, 'net:cpl', 'cpl.tau must be positive');
    N.cplbus = loadbus;
    N.Pc = a*Pd(loadbus);
    N.Qc = a*Qd(loadbus);
    N.ncpl = numel(N.cplbus);
    if a >= 1 - 1e-12
        loadbus = [];  Rs = [];  Xs = [];        % all-CPL: no RL legs
    else
        Rs = Rs/(1 - a);  Xs = Xs/(1 - a);       % RL carries (1 - a)(P + jQ)
    end
end

N.loadbus = loadbus;
N.Rs = Rs;
N.Xs = Xs;
N.nload = numel(loadbus);
N.wb = P.wb;
end

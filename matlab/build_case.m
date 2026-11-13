function [mpc, plant] = build_case(cfg, P)
%BUILD_CASE  Modified IEEE 39-bus 100% IBR MATPOWER case for one scenario.
%
%   cfg fields (all optional; defaults = base case):
%     .devtype  1x10 cellstr {'GFL'|'VSM'} for plants at buses 30..39
%     .kL       load scale (default 1)
%     .k        line-impedance scale (default 1); scales r and x of ALL branches
%     .kdisp    1x10 per-plant dispatch multipliers (default ones; renormalized
%               so total scheduled PG tracks kL * base total)
%     .outage   branch row index (into the 46-branch table) to remove ([] none)
%     .tune     per-plant tuning struct array (see params.converter)
%   P = params.sys_const()
%
%   Returns the MATPOWER case (slack = largest-rating GFM plant) and a plant
%   struct (bus, Sn, Nunit, type, PGbase, slack). The plant ratings are
%     Sn_i = 50 MVA * ceil(1.25 * PG_base_i / 50 MVA), N_units = Sn/5.

define_constants;
mpc0 = loadcase('case39');
if nargin < 1, cfg = struct(); end

dv = def(cfg, 'devtype', [repmat({'GFL'},1,10)]);
kL = def(cfg, 'kL', 1.0);
k  = def(cfg, 'k', 1.0);
kd = def(cfg, 'kdisp', ones(1,10));
outage = def(cfg, 'outage', []);

mpc = mpc0;

% ---- plant ratings from the BASE dispatch (fixed across scenarios) --------
PGb = mpc0.gen(:, PG);                        % base dispatch [MW]
Sn  = 50*ceil(1.25*PGb/50);                   % plant MVA rule
plant.bus   = mpc0.gen(:, GEN_BUS);
plant.Sn    = Sn;                             % [MVA]
plant.Nunit = Sn/5;                           % 5 MVA aggregated units
plant.type  = dv(:);
plant.PGbase= PGb;

% ---- scenario scaling ------------------------------------------------------
mpc.bus(:, [PD QD]) = kL * mpc0.bus(:, [PD QD]);
mpc.branch(:, [BR_R BR_X]) = k * mpc0.branch(:, [BR_R BR_X]);
% Declared minimum series resistance floor: it changes NINE branches of
% case39 (the four zero-r transformers plus five low-r lines at 2e-4..4e-4
% pu). Winding and conductor resistance is physical. r = 0 is a data
% idealization that leaves stationary-frame DC loop modes undamped in the
% dynamic network model. The declared 5e-4 pu is on the LOW side: real
% values are higher, so the modeled system has less damping than the real
% one. Sensitivity: the floor has a negligible effect at the base case.
% Near the marginal shell, zeta scales with the floor (a class-2 row flips
% to class 3 at 2x). The paper mentions the floor in Section II-B.
rmin = 5e-4 * k;
zr = mpc.branch(:, BR_R) < rmin & mpc.branch(:, BR_X) > 0;
mpc.branch(zr, BR_R) = rmin;
% (charging B is a property of the lines; it is NOT scaled by k. k scales the
%  series impedance, the grid-strength axis of the study.)

% Declared parasitic bus capacitance: every bus carries a small shunt
% capacitor. So every bus voltage is a differential state, AND the power
% flow sees exactly the network that the dynamic model integrates. 0.005 pu
% susceptance = 0.5 MVAr at 1 pu voltage per bus: negligible as hardware,
% but the dynamics need it. The C-insensitivity check varies this value.
mpc.bus(:, BS) = mpc.bus(:, BS) + P.Bc_par*mpc.baseMVA;

% dispatch: per-plant multipliers, renormalized to kL * base total; then the
% slack absorbs the mismatch and the losses in the PF.
PGs = PGb .* kd(:);
PGs = PGs * (kL*sum(PGb)/sum(PGs));
mpc.gen(:, PG) = PGs;

% wide Q capability of the plants (converter class +-0.6 Sn). The slack
% plant carries 0.75 Sn, so that fewer stressed cases lose the reference
% bus to a Q-limit switch.
mpc.gen(:, QMAX) =  0.6*Sn;
mpc.gen(:, QMIN) = -0.6*Sn;
mpc.gen(:, PMAX) = Sn;
mpc.gen(:, PMIN) = 0;
mpc.gen(:, MBASE) = Sn;

% ---- slack assignment: largest-rating GFM plant ----------------------------
isGFM = strcmp(dv(:), 'VSM');
if ~any(isGFM)
    slackPlant = 2;   % all-GFL design-study case: keep bus 31 slack (PF only)
else
    gfmIdx = find(isGFM);
    [~, kmax] = max(Sn(gfmIdx));
    slackPlant = gfmIdx(kmax);
end
mpc.bus(:, BUS_TYPE) = 1;                                  % all PQ
mpc.bus(ismember(mpc.bus(:,BUS_I), plant.bus), BUS_TYPE) = 2;  % plants PV
mpc.bus(mpc.bus(:,BUS_I) == plant.bus(slackPlant), BUS_TYPE) = 3;
mpc.gen(slackPlant, QMAX) =  0.75*Sn(slackPlant);
mpc.gen(slackPlant, QMIN) = -0.75*Sn(slackPlant);
plant.slack = slackPlant;

% ---- N-1 outage -------------------------------------------------------------
if ~isempty(outage)
    mpc.branch(outage, BR_STATUS) = 0;
end
end

function v = def(s, f, v)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); end
end

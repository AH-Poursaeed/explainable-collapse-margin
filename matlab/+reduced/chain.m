function R = chain(mpc, res, plant, P, tune)
%CHAIN  The reduced-model index chain for one scenario.
%
%   Ports -> Y_eff/L_eff -> B_lap + B_in -> Schur complement -> A ->
%   lambda_min, phi_min, p, H/Hbar, PWSGSI ->
%   analytic threshold (realness rule) -> C_PWSGSI.
%   Every quantity comes back with its check residuals.

define_constants;
% tune: without it, every converter has its design default (right for the
% design and verification studies). BANK rows MUST pass the sampled tune:
% the interior threshold depends on f_pll/Hv/Dv/kbw, while the s = 0 block
% provably does not depend on the tune.
if nargin < 5 || isempty(tune), tune = repmat(struct(), 10, 1); end
nplant = numel(plant.bus);
V = res.bus(:, VM).*exp(1j*res.bus(:, VA)*pi/180);

% generator injections per plant (system pu)
Sg = zeros(nplant, 1) + 0j;
for gi = 1:size(res.gen, 1)
    pb = find(plant.bus == res.gen(gi, GEN_BUS), 1);
    if ~isempty(pb)
        Sg(pb) = Sg(pb) + (res.gen(gi, PG) + 1j*res.gen(gi, QG))/mpc.baseMVA;
    end
end

bin_conv = zeros(nplant, 1);
Yeff = zeros(nplant, 1);
ports = cell(nplant, 1);
for i = 1:nplant
    c = params.converter(plant.type{i}, plant.Sn(i), P, tune(min(i, numel(tune))));
    c.Ssys = c.Sn;                                  % converter-base port
    sp = struct('cvt', c, 'k', dev.state_map(plant.type{i}), 'nx', c.nx, 'Ssys', c.Sn);
    Sconv = Sg(i) * mpc.baseMVA / plant.Sn(i);      % converter-base injection
    % Evaluation point: the paper evaluates each GFM port at s = 0 with its
    % synchronization frozen (Y_i(0) in (9)). F(0) = -J2 has sigma_max = 1,
    % so sigma_max(-F(0)/(w0 Leff)) = 1/(w0 Leff) exactly.
    % Below, a GFL port keeps its synchronization (freeze_sync = false) and
    % gives Yeff only; a VSM port is frozen (freeze_sync = true), and its
    % Yeff = 1/(w0 Leff) is also its stiffening susceptance bin_conv.
    % GFL plants add zero to B_in.
    switch plant.type{i}
        case 'GFL'
            prt = reduced.port('GFL', sp, V(plant.bus(i)), Sconv, false);
            Yeff(i) = max(svd(prt.Y(0)));           % full terminal admittance
        case 'VSM'
            prt = reduced.port('VSM', sp, V(plant.bus(i)), Sconv, true);   % sync frozen
            Yeff(i) = max(svd(prt.Y(0)));           % = 1/(w0 Leff), the GFM Yeff
            bin_conv(i) = Yeff(i);
    end
    prt.cvt = c;
    ports{i} = prt;
end
R.Yeff = Yeff;
R.Leff = 1./(P.wb*max(bin_conv, eps));              % GFM only (GFL rows meaningless)
R.port_resid = max(cellfun(@(q) q.residual, ports));

% ---- susceptance chain -----------------------------------------------------
BB = reduced.build_B(mpc, res, plant, bin_conv);
R.BB = BB;
R.lam_min = BB.lam_min;
R.p = BB.p;
Nc = nplant;
pp = max(BB.p, 1e-300);
R.H = -sum(pp.*log(pp));                            % participation entropy
if Nc > 1, R.Hbar = R.H/log(Nc); else, R.Hbar = 1; end   % Nc = 1 convention
R.PWSGSI = R.lam_min * R.Hbar;                      % GRIP
R.lam_gap = BB.lam_gap;

% classical indices for the comparison: gSCR (B_in = 0) and the per-bus
% SCR from the driving-point susceptance of the reduced network
Sih = diag(1./sqrt(BB.Sb));
R.gSCR = min(eig(Sih*(BB.Bkron + BB.Bkron.')/2*Sih));
bth = 1./diag(inv(BB.Bkron));                       % driving-point susceptance
R.SCR_bus = bth ./ BB.Sb;
R.SCR_min = min(R.SCR_bus);

% ---- analytic threshold -------------------------------------------
% Y_agg(jw) = sum_GFL p_i Y_GFL,i(jw) + sum_GFM p_j Y_delta,j(jw), converter-base
% pu, common frame. GFL ports are in the common frame by construction; Y_delta is
% built in the local frame and rotated into the common frame by rot2(th0).
isG = strcmp(plant.type, 'GFL');
R.Yagg_s = @(s) yagg_s(s, ports, BB.p, isG, R.Leff, P.wb);
R.Yagg = @(w) R.Yagg_s(1j*w);
T = reduced.threshold(R.Yagg, P.wb);
R.C_SGSI = T.C_SGSI;  R.w_c = T.w_c;  R.im_resid = T.im_resid;
R.thr_rule = T.rule;  R.C_interior = T.C_interior;  R.C_dc = T.C_dc;
R.thr_found = T.found;
R.C_PWSGSI = R.C_SGSI * R.Hbar;    % participation-weighted threshold at this OP
R.ports = ports;
R.plant_type = plant.type;
end

% ---------------------------------------------------------------------------
function Y = yagg_s(s, ports, p, isG, Leff, wb)
Y = zeros(2, 2);
for i = 1:numel(ports)
    if isG(i)
        Y = Y + p(i)*ports{i}.Y(s);
    else
        prt = ports{i};
        Yd_loc = reduced.Ydelta(s, Leff(i), prt.v0, prt.i0, prt.cvt, wb);
        Rth = reduced.rot2(prt.th0);
        Y = Y + p(i)*(Rth*Yd_loc*Rth.');            % load absorption
    end
end
end

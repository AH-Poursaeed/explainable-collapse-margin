function M = build_model(mpc, res, plant, P, tune)
%BUILD_MODEL  Assemble the full-order pure-ODE model (devices + dynamic net).
%
%   State layout: [dev(1..10) 14 each | branch iDQ 2*nbr | bus vDQ 2*nbus |
%                  load iDQ 2*nload]
%
%   mpc/res : MATPOWER case and solved power flow
%   plant   : plant table from build_case
%   P       : params.sys_const()
%   tune    : 1x10 struct array of per-plant tuning overrides (optional)

if nargin < 5 || isempty(tune), tune = repmat(struct(), 10, 1); end
define_constants;

N = net.build_network(mpc, res, P);
M.N = N;  M.P = P;  M.plant = plant;
M.nbus = N.nbus;

% ---- devices ----------------------------------------------------------------
ndev = numel(plant.bus);
ix0 = 0;
for i = 1:ndev
    d.type = plant.type{i};
    d.bus  = plant.bus(i);
    c = params.converter(d.type, plant.Sn(i), P, tune(min(i, numel(tune))));
    sp.cvt = c;  sp.k = dev.state_map(d.type);  sp.nx = c.nx;  sp.Ssys = P.Ssys;
    sp.ref = struct();          % filled by init_state
    d.params = sp;
    d.ix = ix0 + (1:c.nx);
    ix0 = ix0 + c.nx;
    M.dev(i) = d;
end
M.nxdev = ix0;

% ---- network state indices ---------------------------------------------------
M.ix_br   = M.nxdev            + (1:2*N.nbr);     % [id;iq] interleaved per branch
M.ix_bus  = M.ix_br(end)       + (1:2*N.nbus);
lastix = M.ix_bus(end);
M.ix_load = lastix + (1:2*N.nload);
if N.nload > 0, lastix = M.ix_load(end); end
% CPL legs: 2 filtered-current states per constant-power load
if isfield(N, 'ncpl') && N.ncpl > 0
    M.ix_cpl = lastix + (1:2*N.ncpl);
    lastix = M.ix_cpl(end);
else
    M.ix_cpl = [];
end
M.nx = lastix;

% device bus incidence (for the RHS injection scatter)
M.devbus = [M.dev.bus];

% ---- per-row residual scales -----------------------------------------------
% Dividing rhs by these turns every row into a dimensionless pu mismatch of
% its defining algebraic relation (voltage/current/power error). This removes
% the wb/L-style stiffness amplification from the residual metric.
rs = ones(M.nx, 1);
for i = 1:ndev
    d = M.dev(i); c = d.params.cvt; k = d.params.k;
    rs(d.ix(k.theta)) = P.wb;
    if strcmp(d.type, 'VSM'), rs(d.ix(k.w)) = 1/(2*c.Hv); end
    rs(d.ix(k.Pm)) = c.wc;  rs(d.ix(k.Qm)) = c.wc;
    rs(d.ix(k.ifd)) = P.wb/c.Xl1;  rs(d.ix(k.ifq)) = P.wb/c.Xl1;
    rs(d.ix(k.vcd)) = P.wb/c.Bcf;  rs(d.ix(k.vcq)) = P.wb/c.Bcf;
    rs(d.ix(k.igd)) = P.wb/c.Xl2;  rs(d.ix(k.igq)) = P.wb/c.Xl2;
end
for e = 1:N.nbr
    rs(M.ix_br([2*e-1, 2*e])) = P.wb/N.x(e);
end
for b = 1:N.nbus
    rs(M.ix_bus([2*b-1, 2*b])) = P.wb/N.Bc(b);
end
for l = 1:N.nload
    rs(M.ix_load([2*l-1, 2*l])) = P.wb/N.Xs(l);
end
if isfield(N, 'ncpl') && N.ncpl > 0
    rs(M.ix_cpl) = 1/N.tauc;                      % filtered-current rows
end
M.rowscale = rs;
end

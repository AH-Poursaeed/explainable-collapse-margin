function out = run_scenario(cfg, opts)
%RUN_SCENARIO  Run one configuration end-to-end.
%   The steps: case -> PF -> full model -> init (residual gate) ->
%   linearize -> modal. The design study, the verification sweeps and the
%   scenario bank all use this one pipeline.
%
%   cfg  : see build_case (plus cfg.tune = 1x10 struct array of overrides)
%   opts : .linearize (default true), .fdcheck (default false),
%          .quiet (default true)

if nargin < 1, cfg = struct(); end
if nargin < 2, opts = struct(); end
lin  = getdef(opts, 'linearize', true);
fdck = getdef(opts, 'fdcheck', false);

P = params.sys_const();
% optional constant-power-load split, cfg.cpl = struct('alpha', a,
% 'tau', tau_s). Power flow and equilibrium are unchanged; only the dynamic
% load realization changes. Without the field, the model is exactly the
% published one.
if isfield(cfg, 'cpl') && ~isempty(cfg.cpl)
    P.cpl = cfg.cpl;
end
[mpc, plant] = build_case(cfg, P);

mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1, ...
                 'pf.tol', 1e-12);
res = runpf(mpc, mpopt);
out.pf_ok = res.success == 1;
if ~out.pf_ok
    out.fail = 'powerflow';
    return
end
out.res = res;  out.plant = plant;  out.mpc = mpc;

tune = getdef(cfg, 'tune', repmat(struct(), 10, 1));
M = full.build_model(mpc, res, plant, P, tune);
[x0, M, resid0] = full.init_state(M, mpc, res);
out.residual_analytic = resid0;

% Newton polish on the exact RHS: the analytic init is the seed. The
% bus-voltage error left by the PF tolerance (amplified by wb/Bc at low-C
% buses) is removed by 1-3 quasi-Newton steps with the CS Jacobian. Both
% the analytic and the polished residuals are kept and reported.
L = full.linearize(M, x0, 'cs');
resid = resid0;  it = 0;
while resid > 1e-13 && it < 4
    % minimum-norm step: Asys carries the structural zero mode (global
    % rotation), whose null-space component must not enter the correction
    dx = - lsqminnorm(L.Asys, full.rhs(x0, M));
    x0 = x0 + dx;
    resid = norm(full.rhs(x0, M), inf);
    it = it + 1;
end
if it > 0
    L = full.linearize(M, x0, 'cs');   % linearize again at the polished point
end
out.M = M;  out.x0 = x0;  out.residual = resid;
out.residual_pu = norm(full.rhs(x0, M)./M.rowscale, inf);   % pu mismatch form
out.resid_ok = resid < P.tol.resid && out.residual_pu < 1e-12;
if ~out.resid_ok
    out.fail = 'init_residual';
    return
end

if lin
    out.Asys = L.Asys;
    if fdck
        Lfd = full.linearize(M, x0, 'fd');
        out.jacRelErr = max(abs(L.Asys(:) - Lfd.Asys(:)))/max(abs(L.Asys(:)));
    end
    out.S = full.modal(L.Asys, P);
end
out.fail = '';
end

function v = getdef(s, f, v)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); end
end

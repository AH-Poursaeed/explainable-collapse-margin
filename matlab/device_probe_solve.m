function s = device_probe_solve(r, dPc, dQc, P, dev)
%DEVICE_PROBE_SOLVE  Solve one probe, with an optional device-parameter move.
%   A copy of solve_probe (lines 154-213) and blank (lines 143-152) of
%   e2_sens_production.m, statement for statement, with ONE addition:
%   the optional DEV moves one device parameter of the full-order model
%   before the build (the power flow never reads it: build_case and runpf
%   take no tune). DEV = [] (or absent) is the injection probe exactly.
%
%   dev.name  : 'Hv' | 'Dv' (VSM slot), 'f_pll' (GFL slot), 'kbw' (common
%               to the ten plants, as bank.row2cfg builds it)
%   dev.slot  : plant slot 1..10 (ignored for 'kbw')
%   dev.delta : additive change in the parameter's own unit
%   Added output field (diagnostic only, not in the stored export):
%   resid = norm(full.rhs(x0, M), inf) after the single polish step.
if nargin < 5, dev = []; end
s = blank(r.id);
try
    cfg = bank.row2cfg(r);
    if ~isempty(dev)
        cfg.tune = move_device(cfg.tune, cfg.devtype, dev);
    end
    [mpc, plant] = build_case(cfg, P);
    define_constants;
    for i = 1:10
        if i == plant.slack, continue; end
        mpc.gen(i, PG) = mpc.gen(i, PG) + dPc(i)*plant.Sn(i);
    end
    mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1, 'pf.tol', 1e-10);
    if any(dQc)
        res0 = runpf(mpc, mpopt);
        if res0.success ~= 1, return; end
        for i = 1:10
            if i == plant.slack, continue; end
            mpc.bus(mpc.bus(:, BUS_I) == plant.bus(i), BUS_TYPE) = 1;   % PQ
            mpc.gen(i, QG) = res0.gen(i, QG) + dQc(i)*plant.Sn(i);
            mpc.gen(i, QMIN) = mpc.gen(i, QG);  mpc.gen(i, QMAX) = mpc.gen(i, QG);
        end
    end
    res = runpf(mpc, mpopt);
    if res.success ~= 1, return; end
    tune = cfg.tune;
    M = full.build_model(mpc, res, plant, P, tune);
    [x0, M, ~] = full.init_state(M, mpc, res);
    L = full.linearize(M, x0, 'cs');
    dx = - lsqminnorm(L.Asys, full.rhs(x0, M));
    x0 = x0 + dx;
    L = full.linearize(M, x0, 'cs');
    S = full.modal(L.Asys, P);
    R = reduced.chain(mpc, res, plant, P, tune);

    s.ok = true;
    s.sigma_dom = S.sigma_dom;
    z = S.zeta_min;
    if ~S.oob_stable, z = min(z, -1e-6); end
    s.zeta_min = z;
    s.lam_min = R.lam_min;  s.Hbar = R.Hbar;  s.PWSGSI = R.PWSGSI;

    Pg = res.gen(:, PG);  Qg = res.gen(:, QG);
    Vm = res.bus(ismember(res.bus(:, BUS_I), plant.bus), VM);
    Pconv = Pg ./ plant.Sn;  Qconv = Qg ./ plant.Sn;
    Sapp = sqrt(Pg.^2 + Qg.^2) ./ plant.Sn;
    Yht = Sapp(:)./Vm(:).^2;
    for i = 1:10
        s.(sprintf('Pinv_%d', i)) = Pconv(i);
        s.(sprintf('Qinv_%d', i)) = Qconv(i);
        s.(sprintf('Vbus_%d', i)) = Vm(i);
        s.(sprintf('Yeff_%d', i)) = R.Yeff(i);
        s.(sprintf('Yhat_%d', i)) = Yht(i);
    end
    s.Ptot = sum(Pg)/100;  s.Qtot = sum(Qg)/100;
    s.resid = norm(full.rhs(x0, M), inf);          % the one addition (diagnostic)
catch
    % leave blank (ok = false)
end
end

function tune = move_device(tune, devtype, dev)
switch dev.name
    case 'kbw'
        for i = 1:numel(tune), tune(i).kbw = tune(i).kbw + dev.delta; end
    case {'Hv', 'Dv'}
        assert(strcmp(devtype{dev.slot}, 'VSM'), 'device_probe:slot', ...
            '%s probe on a non-VSM slot %d', dev.name, dev.slot);
        tune(dev.slot).(dev.name) = tune(dev.slot).(dev.name) + dev.delta;
    case 'f_pll'
        assert(strcmp(devtype{dev.slot}, 'GFL'), 'device_probe:slot', ...
            'f_pll probe on a non-GFL slot %d', dev.slot);
        tune(dev.slot).f_pll = tune(dev.slot).f_pll + dev.delta;
    otherwise
        error('device_probe:dev', 'unknown device parameter %s', dev.name);
end
end

function s = blank(id)
s = struct('id', id, 'probe', "", 'ok', false, 'sigma_dom', NaN, ...
    'zeta_min', NaN, 'lam_min', NaN, 'Hbar', NaN, 'PWSGSI', NaN, ...
    'Ptot', NaN, 'Qtot', NaN);
for i = 1:10
    s.(sprintf('Pinv_%d', i)) = NaN;  s.(sprintf('Qinv_%d', i)) = NaN;
    s.(sprintf('Vbus_%d', i)) = NaN;  s.(sprintf('Yeff_%d', i)) = NaN;
    s.(sprintf('Yhat_%d', i)) = NaN;
end
s.resid = NaN;
end

function device_probes_selfcheck(root, outdir, which)
%DEVICE_PROBES_SELFCHECK  The first stage of the device-data probes.
%   It stops with an error (nonzero exit) on any failure. No device probe
%   runs before it passes.
%   (i)  injection mode: device_probe_solve with dev = [] re-solves all 41
%        probes (base, Pp/Pm/Qp/Qm of plants 1..10, +-0.02 p.u.) of three of
%        the 236 scenarios (ids 933, 751, 463, the WORKED ids of
%        r22_faith_slack.py). They must reproduce the stored export
%        root/results/final/e2_probe_features.csv:
%          ok flags identical;
%          sigma_dom: |d| <= 1e-6 * max(1, |stored|) s^-1 (the reported
%          rel_dsigma column is |d| / max(1e-3, |stored|));
%          gamma_r = (sigma(+) - sigma(-))/(2*0.02): |d| <= max(1e-6, 1e-6*|stored|);
%          the M block (Pinv, Qinv, Vbus, Ptot, Qtot): |d| <= 1e-8.
%   (ii) (which = "i+ii") a device sensitivity by a second, independent step.
%        On the first id, all its device probes run at s = 0.02 of the
%        sampled range. The probe with the largest |gamma| is recomputed at
%        s = 0.01. It passes when |g(0.02) - g(0.01)| <= 0.05 * |g(0.01)|.
%        Its s = 0.10 value is reported. Also asserted: the M block of every
%        device solve equals the base solve's to 1e-9 (the power flow is
%        untouched).
%   Writes outdir/device_probes_selfcheck_i.csv, device_probes_selfcheck_ii.csv
%   and device_probes_selfcheck.txt.
if nargin < 3, which = "i"; end
if ~exist(outdir, 'dir'), mkdir(outdir); end
P = params.sys_const();
T = readtable(fullfile(root, 'data', 'dataset.csv'));
E = readtable(fullfile(root, 'results', 'final', 'e2_probe_features.csv'), 'TextType', 'string');
ids = [933 751 463];
DP = 0.02;
Mcols = [arrayfun(@(i) sprintf('Pinv_%d', i), 1:10, 'UniformOutput', false), ...
         arrayfun(@(i) sprintf('Qinv_%d', i), 1:10, 'UniformOutput', false), ...
         arrayfun(@(i) sprintf('Vbus_%d', i), 1:10, 'UniformOutput', false), {'Ptot', 'Qtot'}];
probes = mkprobes();
rows = {};  msg = {};  ok = true;
t0 = tic;
for id = ids
    r = table2struct(T(T.id == id, :));
    assert(numel(r) == 1, 'id %d not in the dataset', id);
    sig = containers.Map();  sst = containers.Map();
    for p = 1:size(probes, 1)
        ip = probes{p, 2};  iq = probes{p, 3};
        dPc = zeros(1, 10);  dQc = zeros(1, 10);
        if ip ~= 0, dPc(abs(ip)) = sign(ip)*DP; end
        if iq ~= 0, dQc(abs(iq)) = sign(iq)*DP; end
        s = device_probe_solve(r, dPc, dQc, P, []);
        e = E(E.id == id & E.probe == probes{p, 1}, :);
        assert(height(e) == 1, 'stored probe %s of id %d not found', probes{p, 1}, id);
        dsg = abs(s.sigma_dom - e.sigma_dom) / max(1e-3, abs(e.sigma_dom));
        dM = 0;
        for c = 1:numel(Mcols), dM = max(dM, abs(s.(Mcols{c}) - e.(Mcols{c}))); end
        okf = (s.ok == logical(e.ok));
        rows(end+1, :) = {id, probes{p, 1}, double(s.ok), double(e.ok), s.sigma_dom, ...
            e.sigma_dom, dsg, dM};                                          %#ok<AGROW>
        if s.ok && e.ok
            good = okf && abs(s.sigma_dom - e.sigma_dom) <= 1e-6*max(1, abs(e.sigma_dom)) && dM <= 1e-8;
        else
            good = okf;
        end
        ok = ok && good;
        sig(probes{p, 1}) = s.sigma_dom;  sst(probes{p, 1}) = e.sigma_dom;
    end
    % gamma_r, both channels, plants 1..10
    worstg = 0;
    for kind = ['P', 'Q']
        for i = 1:10
            pp = sprintf('%sp_%d', kind, i);  pm = sprintf('%sm_%d', kind, i);
            gn = (sig(pp) - sig(pm)) / (2*DP);  gs = (sst(pp) - sst(pm)) / (2*DP);
            if isfinite(gs) || isfinite(gn)
                dg = abs(gn - gs);
                if ~(dg <= max(1e-6, 1e-6*abs(gs))), ok = false; end
                worstg = max(worstg, dg);
            end
        end
    end
    msg{end+1} = sprintf('(i) id %d: 41 probes re-solved, worst |gamma diff| %.3g', id, worstg); %#ok<AGROW>
end
C = cell2table(rows, 'VariableNames', {'id', 'probe', 'ok_new', 'ok_stored', ...
    'sigma_new', 'sigma_stored', 'rel_dsigma', 'max_abs_dM'});
writetable(C, fullfile(outdir, 'device_probes_selfcheck_i.csv'));
msg{end+1} = sprintf('(i) worst rel sigma diff %.3g, worst M-block diff %.3g, %.0f s', ...
    max(C.rel_dsigma(C.ok_new == 1 & C.ok_stored == 1)), ...
    max(C.max_abs_dM(C.ok_new == 1 & C.ok_stored == 1)), toc(t0));
msg{end+1} = sprintf('(i) verdict %s', tern(ok, 'PASS', 'FAIL'));
write_msg(outdir, msg);
if ~ok, error('device_probes:selfcheck', 'DEVICE-PROBE SELF-CHECK (i) FAIL (see device_probes_selfcheck.txt)'); end

if which ~= "i+ii", fprintf('%s\n', msg{:}); return; end
% ---- (ii) a device sensitivity by a second, independent step ---------------
r = table2struct(T(T.id == ids(1), :));
Q = device_probe_set(r);
b0 = device_probe_solve(r, zeros(1, 10), zeros(1, 10), P, []);
assert(b0.ok, 'base solve of id %d failed', ids(1));
g = nan(numel(Q), 1);  dMmax = 0;
for q = 1:numel(Q)
    [g(q), dMq] = dev_gamma(r, P, Q(q), 0.02, b0, Mcols);
    dMmax = max(dMmax, dMq);
end
[~, qb] = max(abs(g));
[g01, ~] = dev_gamma(r, P, Q(qb), 0.01, b0, Mcols);
[g10, ~] = dev_gamma(r, P, Q(qb), 0.10, b0, Mcols);
pass2 = abs(g(qb) - g01) <= 0.05*abs(g01) && dMmax <= 1e-9;
D = table(string({Q.feat})', [Q.W]', [Q.base]', g, 'VariableNames', {'feat', 'W', 'base', 'gamma_s002'});
writetable(D, fullfile(outdir, 'device_probes_selfcheck_ii.csv'));
msg{end+1} = sprintf('(ii) id %d: %d device probes; largest |gamma| %s: g(0.02) = %.6g, g(0.01) = %.6g, g(0.10) = %.6g; max M-block move %.3g', ...
    ids(1), numel(Q), Q(qb).feat, g(qb), g01, g10, dMmax);
msg{end+1} = sprintf('(ii) verdict %s', tern(pass2, 'PASS', 'FAIL'));
write_msg(outdir, msg);
fprintf('%s\n', msg{:});
if ~pass2, error('device_probes:selfcheck', 'DEVICE-PROBE SELF-CHECK (ii) FAIL (see device_probes_selfcheck.txt)'); end
end

function [g, dM] = dev_gamma(r, P, q, sc, b0, Mcols)
sp = device_probe_solve(r, zeros(1, 10), zeros(1, 10), P, struct('name', q.name, 'slot', q.slot, 'delta', +sc*q.W));
sm = device_probe_solve(r, zeros(1, 10), zeros(1, 10), P, struct('name', q.name, 'slot', q.slot, 'delta', -sc*q.W));
g = (sp.sigma_dom - sm.sigma_dom) / (2*sc);       % per fraction of the sampled range
dM = 0;
for c = 1:numel(Mcols)
    dM = max([dM, abs(sp.(Mcols{c}) - b0.(Mcols{c})), abs(sm.(Mcols{c}) - b0.(Mcols{c}))]);
end
end

function probes = mkprobes()
% the same as mkprobes in e2_sens_production.m (lines 120-128)
probes = {'base', 0, 0};
for i = 1:10
    probes(end+1, :) = {sprintf('Pp_%d', i),  i, 0};           %#ok<AGROW>
    probes(end+1, :) = {sprintf('Pm_%d', i), -i, 0};           %#ok<AGROW>
    probes(end+1, :) = {sprintf('Qp_%d', i), 0,  i};           %#ok<AGROW>
    probes(end+1, :) = {sprintf('Qm_%d', i), 0, -i};           %#ok<AGROW>
end
end

function write_msg(outdir, msg)
fid = fopen(fullfile(outdir, 'device_probes_selfcheck.txt'), 'w');
fprintf(fid, 'device-probe self-check %s\n', char(datetime('now', 'TimeZone', 'UTC')));
fprintf(fid, '%s\n', msg{:});
fclose(fid);
end

function v = tern(c, a, b)
if c, v = a; else, v = b; end
end

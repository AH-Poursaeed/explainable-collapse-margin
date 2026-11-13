function device_probes_shard(root, outdir, k, nshard)
%DEVICE_PROBES_SHARD  One shard of the device-data probes on 236 scenarios.
%   The scenarios are those of root/results/tables/r17/e2r17_faith_rows_fine_md.csv
%   (file order; scenario j goes to shard mod(j-1, NSHARD)+1). Per scenario:
%   one base solve, then every probe of device_probe_set at the two scales
%   s = 0.02 and 0.10 of the parameter's sampled range W, both signs, so
%   theta = base +- s*W. device_probe_solve re-solves each one (power flow,
%   full model, one polish, re-linearization, modal: the same path as the
%   injection probes). Checkpoint: one file per scenario,
%   outdir/scenarios/scenario_<id>.csv; a scenario whose file exists is
%   skipped. Columns: id, probe (feature name or "base"), the solve's export
%   block (ok, sigma_dom, ..., resid), then scale, sgn, delta (parameter
%   unit) and base_value.
%   root: the repository root; outdir: results/device_probes.
maxNumCompThreads(1);
P = params.sys_const();
SCALES = [0.02 0.10];
T = readtable(fullfile(root, 'data', 'dataset.csv'));
F = readtable(fullfile(root, 'results', 'tables', 'r17', 'e2r17_faith_rows_fine_md.csv'));
assert(height(F) == 236, 'the scenario list has %d rows (expected 236)', height(F));
ddir = fullfile(outdir, 'scenarios');
if ~exist(ddir, 'dir'), mkdir(ddir); end
mine = k:nshard:height(F);
t0 = tic;  nsolve = 0;
for j = mine
    id = F.id(j);
    fn = fullfile(ddir, sprintf('scenario_%d.csv', id));
    if isfile(fn), continue; end
    r = table2struct(T(T.id == id, :));
    assert(numel(r) == 1, 'id %d not in the dataset', id);
    Q = device_probe_set(r);
    L = {};
    s = device_probe_solve(r, zeros(1, 10), zeros(1, 10), P, []);
    L{end+1} = tag(s, "base", 0, 0, 0, NaN);                                %#ok<AGROW>
    for q = 1:numel(Q)
        for sc = SCALES
            for sg = [+1 -1]
                d = struct('name', Q(q).name, 'slot', Q(q).slot, 'delta', sg*sc*Q(q).W);
                s = device_probe_solve(r, zeros(1, 10), zeros(1, 10), P, d);
                L{end+1} = tag(s, string(Q(q).feat), sc, sg, d.delta, Q(q).base); %#ok<AGROW>
            end
        end
    end
    nsolve = nsolve + numel(L);
    writetable(struct2table([L{:}]), [fn '.part'], 'FileType', 'text');   % '.part' is no known extension
    movefile([fn '.part'], fn);                     % a killed write never looks done
    fprintf('shard %d: id %d done (%d solves, %.0f s so far)\n', k, id, numel(L), toc(t0));
end
fprintf('shard %d complete: %d solves in %.0f s\n', k, nsolve, toc(t0));
end

function s = tag(s, probe, sc, sg, delta, base)
s.probe = probe;
s.scale = sc;  s.sgn = sg;  s.delta = delta;  s.base_value = base;
end

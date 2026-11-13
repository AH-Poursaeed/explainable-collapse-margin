function cp60_relabel_check(root, outdir, nshard)
%CP60_RELABEL_CHECK  The gate before the 30% re-label.
%   The 60% re-label of the whole bank by cp_relabel_shard must reproduce
%   the stored result root/results/tables/r16/r16_cp_relabel.csv (7324 ok,
%   0 fail, 3310 binary flips, 45.1939%).
%
%   PASS needs ALL of:
%     G1  7,324 rows, ids = the dataset ids, each once; 7,324 ok (finite sigma_cp)
%     G2  3,310 binary flips (flip_bin == 1)
%     G3  every flip goes to unstable (sigma_cz < 0 and sigma_cp >= 0)
%     G4  the flipped ids are exactly the reference's flipped ids
%     G5  per row |sigma_cp - sigma_cp_ref| / max(1e-3, |sigma_cp_ref|) <= 1e-6
%         (the same rule as the r16 gate and audit_rows)
%   Reported, not gated: rows whose class_cp differs, worst zeta_cp difference.
%   Writes outdir/cp_relabel_gate_060.txt and outdir/cp60_relabel_check.csv;
%   error() (nonzero exit) on FAIL.

R = merge_shards(outdir, 60, nshard);
T = readtable(fullfile(root, 'data', 'dataset.csv'));
F = readtable(fullfile(root, 'results', 'tables', 'r16', 'r16_cp_relabel.csv'));
msg = {};  ok = true;

g1 = height(R) == 7324 && numel(unique(R.id)) == 7324 && ...
     isempty(setxor(R.id, T.id)) && all(isfinite(R.sigma_cp));
msg{end+1} = sprintf('G1 rows %d, unique ids %d, ids = dataset ids %d, ok rows %d', ...
    height(R), numel(unique(R.id)), isempty(setxor(R.id, T.id)), sum(isfinite(R.sigma_cp)));
ok = ok && g1;

[tf, loc] = ismember(R.id, F.id);
assert(all(tf), 'an id of the 60-percent re-label is not in the reference');
Fr = F(loc, :);
nflip = sum(R.flip_bin == 1);
g2 = nflip == 3310;
msg{end+1} = sprintf('G2 binary flips %d (reference 3310)', nflip);
ok = ok && g2;

fl = R.flip_bin == 1;
nup = sum(fl & R.sigma_cz < 0 & R.sigma_cp >= 0);
g3 = nup == nflip;
msg{end+1} = sprintf('G3 flips to unstable %d of %d', nup, nflip);
ok = ok && g3;

g4 = isequal(sort(R.id(fl)), sort(F.id(F.flip_bin == 1)));
msg{end+1} = sprintf('G4 flipped ids identical to the reference: %d', g4);
ok = ok && g4;

d = abs(R.sigma_cp - Fr.sigma_cp) ./ max(1e-3, abs(Fr.sigma_cp));
[wd, wi] = max(d);
g5 = all(isfinite(d)) && wd <= 1e-6;
msg{end+1} = sprintf('G5 worst rel sigma_cp difference %.3g (id %d), tol 1e-6', wd, R.id(wi));
ok = ok && g5;

ncls = sum(R.class_cp ~= Fr.class_cp);
dz = max(abs(R.zeta_cp - Fr.zeta_cp));
msg{end+1} = sprintf('report: class_cp differs on %d rows; worst |zeta_cp diff| %.3g', ncls, dz);

verdict = 'FAIL';  if ok, verdict = 'PASS'; end
fid = fopen(fullfile(outdir, 'cp_relabel_gate_060.txt'), 'w');
fprintf(fid, 'cp relabel 60%% reproduction gate %s %s\n', verdict, char(datetime('now', 'TimeZone', 'UTC')));
fprintf(fid, '%s\n', msg{:});
fclose(fid);
fprintf('%s\n', msg{:});
writetable(R, fullfile(outdir, 'cp60_relabel_check.csv'));
if ~ok
    error('cp_relabel:gate', 'GATE FAIL: the 60%% re-label does not reproduce r16_cp_relabel.csv (see cp_relabel_gate_060.txt)');
end
fprintf('GATE PASS\n');
end

function R = merge_shards(outdir, a100, nshard)
parts = cell(nshard, 1);
for k = 1:nshard
    f = fullfile(outdir, sprintf('cp_relabel_a%03d_shard%02d.csv', a100, k));
    assert(isfile(f), 'missing shard file %s', f);
    parts{k} = readtable(f, 'Delimiter', ',', 'TextType', 'char');
    % an all-empty fail column reads as double NaN; make it text in every part
    if ~iscell(parts{k}.fail), parts{k}.fail = repmat({''}, height(parts{k}), 1); end
end
R = vertcat(parts{:});
R = sortrows(R, 'id');
end

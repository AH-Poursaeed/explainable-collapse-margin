function cp30_relabel_merge(root, outdir, nshard)
%CP30_RELABEL_MERGE  Merge the alpha = 0.30 shards of cp_relabel_shard.
%   It writes the result:
%   outdir/cp30_relabel.csv          per row, the r16_cp_relabel.csv columns
%       (sigma_cz = stored constant-impedance decay rate, sigma_cp = the 30%
%       constant-power one; the margin m of the paper is min(sigma, +1))
%   outdir/cp30_relabel_summary.csv  the r16_cp_summary.csv columns plus the
%       flip directions and the nesting in the 60% flips
%   outdir/cp30_relabel_summary.txt  the same, readable
%   Runs only after cp60_relabel_check has passed (its gate file must say
%   PASS).

g = fileread(fullfile(outdir, 'cp_relabel_gate_060.txt'));
assert(startsWith(g, 'cp relabel 60% reproduction gate PASS'), 'the 60%% gate has not passed');
R = merge30(outdir, nshard);
T = readtable(fullfile(root, 'data', 'dataset.csv'));
assert(height(R) == 7324 && numel(unique(R.id)) == 7324 && isempty(setxor(R.id, T.id)), ...
    'incomplete 30%% re-label: %d rows', height(R));
writetable(R, fullfile(outdir, 'cp30_relabel.csv'));

okrow = isfinite(R.sigma_cp);
fl = okrow & R.flip_bin == 1;
up = fl & R.sigma_cz < 0 & R.sigma_cp >= 0;
down = fl & R.sigma_cz >= 0 & R.sigma_cp < 0;
dm = -(R.sigma_cp(okrow) - R.sigma_cz(okrow));      % as in the r16 files: m = -sigma shift
mcz = min(R.sigma_cz, 1);  mcp = min(R.sigma_cp, 1);  % capped margins

R60 = readtable(fullfile(outdir, 'cp60_relabel_check.csv'));
[~, l60] = ismember(R.id, R60.id);
fl60 = R60.flip_bin(l60) == 1;
n_in60 = sum(fl & fl60);  n_not60 = sum(fl & ~fl60);
n_cz_unst = sum(R.sigma_cz >= 0);

S = struct();
S.n_rows = height(R);  S.n_ok = sum(okrow);  S.n_fail = height(R) - sum(okrow);
S.n_flip_bin = sum(fl);  S.pct_flip_bin = 100 * sum(fl) / height(R);
S.n_to_unstable = sum(up);  S.n_to_stable = sum(down);
S.n_class_change = sum(R.dclass(okrow) ~= 0);
S.median_dm = median(dm);  S.q05_dm = quantile(dm, 0.05);  S.q95_dm = quantile(dm, 0.95);
S.mean_dm = mean(dm);  S.max_abs_dm = max(abs(dm));
S.n_unstable_cz = n_cz_unst;  S.n_unstable_cp = sum(R.sigma_cp >= 0);
S.n_flips_also_flip_at_060 = n_in60;  S.n_flips_not_at_060 = n_not60;
S.median_capped_shift = median(mcp(okrow) - mcz(okrow));
S.n_trust_cp1 = sum(R.trust_cp == 1);
writetable(struct2table(S), fullfile(outdir, 'cp30_relabel_summary.csv'));

fid = fopen(fullfile(outdir, 'cp30_relabel_summary.txt'), 'w');
fprintf(fid, '30%% constant-power re-label of the 7,324-row bank (tau 0.020), %s\n', ...
    char(datetime('now', 'TimeZone', 'UTC')));
fn = fieldnames(S);
for i = 1:numel(fn), fprintf(fid, '%s = %.10g\n', fn{i}, S.(fn{i})); end
fclose(fid);
type(fullfile(outdir, 'cp30_relabel_summary.txt'));
end

function R = merge30(outdir, nshard)
parts = cell(nshard, 1);
for k = 1:nshard
    f = fullfile(outdir, sprintf('cp_relabel_a%03d_shard%02d.csv', 30, k));
    assert(isfile(f), 'missing shard file %s', f);
    parts{k} = readtable(f, 'Delimiter', ',', 'TextType', 'char');
    if ~iscell(parts{k}.fail), parts{k}.fail = repmat({''}, height(parts{k}), 1); end
end
R = sortrows(vertcat(parts{:}), 'id');
end

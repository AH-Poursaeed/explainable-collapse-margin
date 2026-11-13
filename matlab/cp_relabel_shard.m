function cp_relabel_shard(alpha, k, nshard, root, outdir)
%CP_RELABEL_SHARD  One shard of the constant-power re-label of the bank.
%   The bank has 7,324 rows. ALPHA is the load fraction: that share of every
%   inductive load becomes smoothed constant power, tau = 20 ms.
%
%   The per-row computation is r16_cp_relabel.m lines 84-125, statement for
%   statement (same TAU = 0.02, same EDGES, same columns and printf
%   formats). Differences, none of them a model parameter:
%     - ALPHA is an argument (0.60 gives the labels of r16_cp_relabel.csv,
%       0.30 those of results/cp30/cp30_relabel.csv);
%     - rows are split into NSHARD interleaved shards (row i goes to shard
%       mod(i-1, NSHARD)+1) so that 16 single-thread MATLAB processes share
%       one node; each shard writes its own file;
%     - maxNumCompThreads(1) (r16_cp_relabel.m runs maxNumCompThreads(2));
%     - the 50-row alpha = 0 gate and the metrics block of r16_cp_relabel.m
%       are not here: the 60% re-label of the whole bank, checked by
%       cp60_relabel_check.m, is the gate.
%   Row-checkpointed: append per row, skip-if-exists by id (resumable).
%
%   root   : the repository root (root/data/dataset.csv)
%   outdir : where the shard file goes (results/cp30)
%
%   cp_relabel_shard(0.30, 3, 16, '..', '../results/cp30')

maxNumCompThreads(1);
assert(alpha > 0 && alpha < 1, 'alpha must be in (0, 1)');
TAU = 0.02;
EDGES = [0 0.01 0.03 0.05 0.10];
if ~exist(outdir, 'dir'), mkdir(outdir); end
fout = fullfile(outdir, sprintf('cp_relabel_a%03d_shard%02d.csv', round(100*alpha), k));

T = readtable(fullfile(root, 'data', 'dataset.csv'));
assert(height(T) == 7324, 'dataset drift: %d rows (expected 7324)', height(T));
S = table2struct(T);                       % never row-slice a table in a hot loop
N = numel(S);
mine = k:nshard:N;

done = containers.Map('KeyType', 'double', 'ValueType', 'logical');
if isfile(fout)
    T0 = readtable(fout);
    for i = 1:height(T0), done(T0.id(i)) = true; end
    fprintf('resume: %d rows already done in shard %d\n', done.Count, k);
else
    fid = fopen(fout, 'w');
    fprintf(fid, ['id,sigma_cz,zeta_cz,class_cz,sigma_cp,zeta_cp,fdom_cp,', ...
        'class_cp,oob_cp,trust_cp,resid_pu,flip_bin,dclass,fail\n']);
    fclose(fid);
end

t0 = tic;  n0 = done.Count;
for i = mine
    r = S(i);
    if isKey(done, r.id), continue; end
    % ---- r16_cp_relabel.m lines 87-118, ALPHA -> alpha ----------------------
    cfg = bank.row2cfg(r);
    cfg.cpl = struct('alpha', alpha, 'tau', TAU);
    sig = NaN; zeta = NaN; fdom = NaN; cls = NaN; oobf = NaN; trust = NaN;
    resid = NaN; flip = NaN; dcls = NaN; failmsg = '';
    try
        o = run_scenario(cfg);
        if ~isempty(o.fail)
            failmsg = o.fail;
        elseif o.S.nzero ~= 1
            failmsg = 'zero_mode_count';
        else
            zeta = o.S.zeta_min;  sig = o.S.sigma_dom;  fdom = o.S.f_dom;
            trust = o.S.trust_dom;  resid = o.residual_pu;
            if ~o.S.oob_stable
                zeta = min(zeta, -1e-6);
                sig = max(sig, o.S.oob_maxRe);
                oobf = 1;
            else
                oobf = 0;
            end
            cls = sum(zeta >= EDGES);  if zeta <= 0, cls = 0; end
            flip = double((sig >= 0) ~= (r.sigma_dom >= 0));
            dcls = cls - r.class;
        end
    catch e
        failmsg = strrep(['error: ' e.message], ',', ';');
    end
    fid = fopen(fout, 'a');
    fprintf(fid, '%d,%.10g,%.10g,%d,%.10g,%.10g,%.10g,%g,%g,%g,%.3g,%g,%g,%s\n', ...
        r.id, r.sigma_dom, r.zeta_min, r.class, sig, zeta, fdom, cls, oobf, ...
        trust, resid, flip, dcls, failmsg);
    fclose(fid);
    % ---- end of the copied block -------------------------------------------
    done(r.id) = true;
    if mod(done.Count, 50) == 0
        el = toc(t0);  rate = (done.Count - n0) / max(el, 1);
        fprintf('shard %d: %d/%d rows (%.3f rows/s, ETA %.0f min)\n', k, ...
            done.Count, numel(mine), rate, (numel(mine) - done.Count) / rate / 60);
    end
end
fprintf('shard %d (alpha %.2f) complete: %d rows in %.0f s\n', k, alpha, ...
    done.Count, toc(t0));
end

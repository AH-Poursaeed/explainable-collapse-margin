function T = assemble_bank(outname)
%ASSEMBLE_BANK  Collect the bank chunks into the flat dataset (CSV + MAT).
%   The dataset follows the schema in data/schema.md. The function runs the
%   class-0 integrity audit (class 0 <=> sigma_dom >= 0) sample by sample
%   and refuses to export on any mismatch.

if nargin < 1, outname = 'bank'; end
here = fileparts(mfilename('fullpath'));
dirs = {fullfile(here, '..', 'results', outname), ...
        fullfile(here, '..', 'results', [outname '_topup'])};
S = [];
for dd2 = 1:numel(dirs)
    dd = dir(fullfile(dirs{dd2}, 'chunk_*.mat'));
    for i = 1:numel(dd)
        d = load(fullfile(dirs{dd2}, dd(i).name));
        Snew = d.S;
        % the ONLY field allowed to differ across rounds is 'round'
        if ~isfield(Snew, 'round'), [Snew.round] = deal(1); end
        if ~isempty(S)
            if ~isfield(S, 'round'), [S.round] = deal(1); end
            assert(isempty(setxor(fieldnames(S), fieldnames(Snew))), ...
                'assemble_bank: field mismatch between rounds');
            Snew = orderfields(Snew, S);
        end
        S = [S, Snew];                                        %#ok<AGROW>
    end
end
assert(~isempty(S), 'no chunks found');
fprintf('collected %d rows from %d chunks; ok = %d, failed = %d\n', ...
    numel(S), numel(dd), nnz([S.ok]), nnz(~[S.ok]));
fails = {S(~[S.ok]).fail};
if ~isempty(fails)
    [uf, ~, ic] = unique(fails);
    for u = 1:numel(uf)
        fprintf('  fail[%d]: %s\n', nnz(ic == u), uf{u});
    end
end

ok = S([S.ok]);

% ---- class-0 integrity audit ------------------------
% the label: class 0 <=> zeta_min <= 0, and zeta = -Re/|lambda| makes
% zeta_min <= 0 <=> sigma_dom >= 0 (tolerance for the exact-marginal edge).
cls = [ok.class];  sig = [ok.sigma_dom];  zet = [ok.zeta_min];
bad1 = find((cls == 0) ~= (zet <= 0));
% the identity is EXACT by construction (same eigenvalue set on both sides;
% a tolerance here widens the mismatch set instead of narrowing it)
bad2 = find((zet <= 0) ~= (sig >= 0));
fprintf('class-0 audit: label-vs-zeta mismatches %d, zeta-vs-sigma mismatches %d (of %d rows)\n', ...
    numel(bad1), numel(bad2), numel(ok));
assert(isempty(bad1) && isempty(bad2), 'CLASS-0 AUDIT FAILED: %s | %s', ...
    mat2str(bad1(1:min(10,end))), mat2str(bad2(1:min(10,end))));

% ---- flat table ---------------------------------------------------------------
scal = {'id','round','outage','kL','k','residual_pu','oob_flag','zeta_min','sigma_dom', ...
    'f_dom','class','abstain','lam_min','H','Hbar','PWSGSI','C_SGSI','C_interior', ...
    'C_dc','im_resid','w_c','gSCR','SCR_min','lam_gap','port_resid','Ptot','Qtot', ...
    'kbw','nGFM','oob_stable','nzero','trust_dom'};   % the last three are exported as well
T = table();
for f = scal
    T.(f{1}) = [ok.(f{1})]';
end
T.thr_rule = string({ok.thr_rule}');
T.cfgbuses = string({ok.cfgbuses}');
vec = {'Pinv','Qinv','Yeff','Vbus','Yhat','p_part','fpll','Hv','Dv','kdisp'};
for f = vec
    M = vertcat(ok.(f{1}));
    for j = 1:size(M, 2)
        T.(sprintf('%s_%d', f{1}, j)) = M(:, j);
    end
end

datadir = fullfile(here, '..', 'data');
if ~exist(datadir, 'dir'), mkdir(datadir); end
writetable(T, fullfile(datadir, 'dataset.csv'));
save(fullfile(datadir, 'dataset.mat'), 'T');
fprintf('exported %d rows x %d cols -> data/dataset.csv\n', height(T), width(T));

% class support
fprintf('class support: ');
for c = 0:5, fprintf('%d:%d  ', c, nnz(T.class == c)); end
fprintf('| abstain flagged: %d\n', nnz(T.abstain));
end

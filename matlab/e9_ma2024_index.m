function e9_ma2024_index(outdir, r_first, r_last, peer_csv)
%E9_MA2024_INDEX  The grid-strength index of Ma et al. (2024) on every bank row.
%   The script computes the in-domain index on every row of the scenario bank,
%   in three network forms. The third form is the network as published,
%   without the shunts.
%
%   Source: F. Ma, H. Xin, D. Wu, Y. Liu, X. Chen, "Assessing small-signal
%   grid strength of 100% inverter-based power systems", IEEE Trans. Power
%   Del., vol. 39, no. 5, 2024 (preprint doi 10.36227/techrxiv.21120517.v8,
%   eqs. 22-24, 33 and 39).
%
%   Per bank row:
%   1. Rebuild the scenario (bank.row2cfg, build_case, runpf) and the
%      stiffness part of reduced.chain. The GFM ports identified at s = 0
%      give B_in = 1/(w0 Leff) (reduced.port, synchronization frozen).
%      reduced.build_B gives the Kron-reduced susceptance B_kron, the
%      stiffening Bin_sys and the plant ratings S_b on the system base. The
%      GFL ports and the threshold search of the chain are not needed here.
%   2. Check the row against the bank: trace_ok = 1 when lambda_min(A) and
%      the classical gSCR (B_in = 0) of the stored row are both reproduced
%      to 1e-6.
%   3. Take S = sym(B + Bin_sys) and split the plants into GFL (G) and GFM
%      (F). Form Y = S_bG^(-1/2) (S_GG - S_GF S_FF^-1 S_FG) S_bG^(-1/2). It
%      is the symmetric form of Y_eq of eq. 33, with the same eigenvalues.
%      Then compute eq. 39 in both of its forms:
%        form 1   gSCR = lambda_min(Y)
%        form 2   gSCR = 1 / lambda_max(Z S_bG),  Z = (S_GG - S_GF S_FF^-1 S_FG)^-1
%      The two forms are equal when Y is positive definite.
%   The networks B:
%     loads absorbed   B_kron as the chain builds it: the loads enter B_lap
%                      as their susceptance at the power-flow voltages (the
%                      load model of this paper)
%     as published     the same reduction with the load susceptance taken
%                      out of B_lap: the source states that its loads play
%                      no role in the small-signal admittance model
%                      (printed p. 4)
%   ASSUMPTION (as-published network): the line charging and the declared
%   bus shunt capacitance stay in B_lap. The pages of the source that were
%   read for this script (printed pp. 4-8) do not settle how the source
%   treats line charging and other shunts.
%     as published, no shunts
%                      the as-published reduction with the line charging
%                      (BR_B) and the declared bus shunt capacitance (BS)
%                      taken out as well. makeYbus rebuilds B_lap from the
%                      lossless case with BR_B = 0 and BS = 0, loads out. The
%                      GFM stiffening and the transformer taps stay. Per row
%                      the script records the identity
%                      B_noshunt - B_published = diag(s). Here s is the shunt
%                      susceptance of each bus, read from the case data in
%                      makeYbus's convention: BS / baseMVA, and per
%                      in-service branch b/2 at its to bus and (b/2)/|tap|^2
%                      at its from bus. The gated stage
%                      (python/e9_ma2024_index.py) holds it to 1e-9.
%   NOT COMPUTED here: the threshold CgSCR of the source (eq. 40).
%   e9_ma2024_threshold.m computes it.
%
%   Columns of the rows file (one line per bank row):
%     id, nGFL, nGFM             bank id; GFL and GFM plant counts
%     lam_min_row, lam_min_re    stored and rebuilt lambda_min(A)
%     gSCR_row, gSCR_re          stored and rebuilt classical gSCR
%     trace_ok                   1 when both are reproduced to 1e-6
%     gscr_ma_load               eq. 39 form 1, loads absorbed
%     gscr_ma_load_eq39          eq. 39 form 2, loads absorbed
%     gscr_ma_noload             eq. 39 form 1, as published
%     gscr_ma_noload_eq39        eq. 39 form 2, as published
%     lam_min_noload             smallest eigenvalue of the S_b-scaled
%                                stiffened matrix over all plants, as
%                                published (the loads-out counterpart of
%                                lambda_min(A))
%     eigmin_gfm_noload          smallest eigenvalue of S_FF, as published
%     gscr_ma_noshunt            eq. 39 form 1, as published, no shunts
%     gscr_ma_noshunt_eq39       eq. 39 form 2, as published, no shunts
%     lam_min_noshunt            lam_min_noload's counterpart, no shunts
%     eigmin_gfm_noshunt         smallest eigenvalue of S_FF, no shunts
%     shunt_offdiag_max          max |off-diagonal of B_noshunt - B_published|
%     shunt_diag_resid           max_i |(B_noshunt - B_published)_ii - s_i|
%     shunt_s_min, shunt_s_sum   the smallest and the sum of the s_i (pu)
%     eigmin_int_noload          smallest eigenvalue of the interior block of
%                                B_published, the premise of the Schur
%                                monotonicity that the gated stage checks
%     code_id                    first 16 hex digits of the SHA-256 of this
%                                file with carriage returns removed
%   The two inertia columns support the sign count of the as-published
%   index. When S_FF is positive definite, the inertia of S is that of S_FF
%   plus that of its Schur complement (Haynsworth). So gscr_ma_noload <= 0
%   exactly when lam_min_noload <= 0.
%
%   Usage
%     e9_ma2024_index(outdir, r_first, r_last)
%     e9_ma2024_index(outdir, r_first, r_last, peer_csv)
%   Runs bank rows r_first..r_last in file order. r_first > r_last runs
%   downward, so two workers can start at the two ends of the bank. The
%   optional peer_csv is the other worker's rows file. Rows found there are
%   skipped, and the script reads the file again at most every 5 s. So the
%   two workers meet wherever their speeds take them. Writes only
%   outdir/e9_rows_<r_first>_<r_last>.csv. Per row, the script appends one
%   line and closes the file. A rerun resumes. It drops a last line cut by a
%   kill (no trailing newline). It refuses a file with another header, a
%   malformed line, or rows written by another version of this script.
%   python/e9_ma2024_index.py --merge joins the workers' rows files into
%   results/final/tables/e9_ma2024_index_rows.csv.

here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..');
Tb = readtable(fullfile(root, 'data', 'dataset.csv'));
Sbank = table2struct(Tb);   % row access on the wide table is slow
nrow = numel(Sbank);
if nargin < 1 || isempty(outdir), outdir = fullfile(tempdir, 'e9_ma2024_index'); end
if nargin < 2 || isempty(r_first), r_first = 1; end
if nargin < 3 || isempty(r_last), r_last = nrow; end
if nargin < 4, peer_csv = ''; end
assert(all([r_first r_last] >= 1 & [r_first r_last] <= nrow) && ...
    r_first == round(r_first) && r_last == round(r_last), 'e9:range', ...
    'row range %g..%g is outside 1..%d', r_first, r_last, nrow);
if ~isfolder(outdir), mkdir(outdir); end

maxNumCompThreads(2);
P = params.sys_const();
code_id = code_hash([mfilename('fullpath') '.m']);
ncol = numel(columns());
outcsv = fullfile(outdir, sprintf('e9_rows_%d_%d.csv', r_first, r_last));
ids = [Sbank.id];
assert(all(ids >= 1 & ids == round(ids)), 'e9:ids', 'bank ids must be positive integers');
doneOwn = false(max(ids), 1);
doneOwn(resume_own(outcsv, code_id, ncol)) = true;
donePeer = false(max(ids), 1);
fprintf('e9_ma2024_index: bank rows %d, range %d..%d, code_id %s, MATLAB %s\n', ...
    nrow, r_first, r_last, code_id, version);
fprintf('rows file %s (%d rows already in it)\n', outcsv, nnz(doneOwn));
if ~isempty(peer_csv), fprintf('peer file %s\n', peer_csv); end

mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1, 'pf.tol', 1e-12);
define_constants;
FMT = ['%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%d,%.12g,%.12g,%.12g,%.12g,', ...
    '%.12g,%.12g,', repmat('%.12g,', 1, 9), '%s\n'];
step = 1;
if r_last < r_first, step = -1; end
order = r_first:step:r_last;
PEER_REFRESH_S = 5;
tPeer = -inf;
[ncomp, nown, npeer, nfail, ntrace] = deal(0);
t0 = tic;
for k = 1:numel(order)
    row = Sbank(order(k));
    if doneOwn(row.id)
        nown = nown + 1;
    else
        if ~isempty(peer_csv) && ~donePeer(row.id) && toc(t0) - tPeer > PEER_REFRESH_S
            donePeer(peer_ids(peer_csv, code_id, ncol)) = true;
            tPeer = toc(t0);
        end
        if donePeer(row.id)
            npeer = npeer + 1;
        else
            [line, ok] = one_row(row, P, mpopt, code_id, FMT, VM, QD);
            if isempty(line)
                nfail = nfail + 1;
            else
                append_line(outcsv, line);   % an I/O failure stops the worker
                doneOwn(row.id) = true;
                ncomp = ncomp + 1;
                ntrace = ntrace + ~ok;
            end
        end
    end
    if mod(k, 200) == 0
        fprintf(['position %d of %d (bank row %d), %.0f s: computed %d, ', ...
            'skipped own %d, skipped peer %d, failed %d, trace misses %d\n'], ...
            k, numel(order), order(k), toc(t0), ncomp, nown, npeer, nfail, ntrace);
    end
end
fprintf(['FINISHED range %d..%d: computed %d, skipped own %d, skipped peer %d, ', ...
    'failed %d, trace misses %d, %.0f s\n'], r_first, r_last, ncomp, nown, npeer, ...
    nfail, ntrace, toc(t0));
end

% ---------------------------------------------------------------------------
function [line, ok] = one_row(row, P, mpopt, code_id, FMT, VM, QD)
% One bank row; an empty line reports a failed row (printed, then skipped).
ok = false;
try
    cfg = bank.row2cfg(row);
    [mpc, plant] = build_case(cfg, P);
    res = runpf(mpc, mpopt);
    assert(res.success == 1, 'power flow failed');
    R = stiffness_part(mpc, res, plant, P, cfg.tune);
    BB = R.BB;
    ok = abs(R.lam_min - row.lam_min) < 1e-6 && abs(R.gSCR - row.gSCR) < 1e-6;

    isF = strcmp(plant.type(:), 'VSM');
    nGFL = sum(strcmp(plant.type(:), 'GFL'));
    assert(any(isF) && any(~isF), 'e9:partition', 'the row has no GFL or no GFM plant');
    L = index_parts(BB.Bkron + BB.Bin_sys, BB.Sb, isF);

    % the same reduction with the load susceptance taken out of B_lap
    nb = size(mpc.bus, 1);
    noload = BB.Blap;
    Vm = res.bus(:, VM);
    Qd = res.bus(:, QD) / mpc.baseMVA;
    noload(sub2ind([nb nb], 1:nb, 1:nb)) = diag(BB.Blap) - Qd ./ Vm.^2;
    cb = BB.cb;
    intr = setdiff((1:nb)', cb);
    Bk = noload(cb, cb) - noload(cb, intr) * (noload(intr, intr) \ noload(intr, cb));
    N = index_parts(Bk + BB.Bin_sys, BB.Sb, isF);

    % as published, no shunts: the line charging and the declared bus shunt
    % capacitance taken out as well
    [Bns, sh] = noshunt_lap(mpc);
    Dsh = Bns - noload;
    offd = Dsh;
    offd(sub2ind([nb nb], 1:nb, 1:nb)) = 0;
    Bk3 = Bns(cb, cb) - Bns(cb, intr) * (Bns(intr, intr) \ Bns(intr, cb));
    W = index_parts(Bk3 + BB.Bin_sys, BB.Sb, isF);
    Bii = noload(intr, intr);
    eint = min(eig((Bii + Bii.') / 2));

    line = sprintf(FMT, row.id, nGFL, sum(isF), row.lam_min, R.lam_min, row.gSCR, ...
        R.gSCR, ok, L.lam1, L.lam1_eq39, N.lam1, N.lam1_eq39, N.lam_all, ...
        N.eigmin_FF, W.lam1, W.lam1_eq39, W.lam_all, W.eigmin_FF, max(abs(offd(:))), ...
        max(abs(diag(Dsh) - sh)), min(sh), sum(sh), eint, code_id);
catch err
    fprintf('row id %d failed: %s\n', row.id, err.message);
    line = '';
end
end

function R = stiffness_part(mpc, res, plant, P, tune)
% The stiffness part of reduced.chain (its lines 19-52, 61 and 73), without
% the GFL ports and the threshold search. The stored lam_min and gSCR
% columns gate it on every row.
define_constants;
nplant = numel(plant.bus);
V = res.bus(:, VM) .* exp(1j * res.bus(:, VA) * pi / 180);
Sg = zeros(nplant, 1) + 0j;
for gi = 1:size(res.gen, 1)
    pb = find(plant.bus == res.gen(gi, GEN_BUS), 1);
    if ~isempty(pb)
        Sg(pb) = Sg(pb) + (res.gen(gi, PG) + 1j * res.gen(gi, QG)) / mpc.baseMVA;
    end
end
bin_conv = zeros(nplant, 1);
for i = 1:nplant
    if strcmp(plant.type{i}, 'VSM')
        c = params.converter(plant.type{i}, plant.Sn(i), P, tune(min(i, numel(tune))));
        c.Ssys = c.Sn;
        sp = struct('cvt', c, 'k', dev.state_map(plant.type{i}), 'nx', c.nx, 'Ssys', c.Sn);
        Sconv = Sg(i) * mpc.baseMVA / plant.Sn(i);
        prt = reduced.port('VSM', sp, V(plant.bus(i)), Sconv, true);
        bin_conv(i) = max(svd(prt.Y(0)));
    end
end
BB = reduced.build_B(mpc, res, plant, bin_conv);
Sih = diag(1 ./ sqrt(BB.Sb));
R = struct('BB', BB, 'lam_min', BB.lam_min, ...
    'gSCR', min(eig(Sih * (BB.Bkron + BB.Bkron.') / 2 * Sih)));
end

function out = index_parts(B, Sb, isF)
% Y_eq of eq. 33 on the partition GFL (G) / GFM (F) and eq. 39 in both
% forms, with the two inertia diagnostics of the sign count.
S = (B + B.') / 2;
isG = ~isF;
SGG = S(isG, isG);
SGF = S(isG, isF);
SFF = S(isF, isF);
Sch = SGG - SGF * (SFF \ SGF.');
SbG = Sb(isG);
Sih = diag(1 ./ sqrt(SbG));
out.lam1 = min(eig(Sih * ((Sch + Sch.') / 2) * Sih));
% eq. 39 writes Z as an explicit inverse; it is kept so that form 2 is the
% source's expression as written
Z = inv(Sch);
out.lam1_eq39 = 1 / max(real(eig(Z * diag(SbG)))); %#ok<MINV>
out.eigmin_FF = min(eig(SFF));
Sa = diag(1 ./ sqrt(Sb));
A = Sa * S * Sa;
out.lam_all = min(eig((A + A.') / 2));
end

function [B0, sh] = noshunt_lap(mpc)
% B0 is the B_lap of reduced.build_B before the loads enter (-imag(Ybus) of
% the lossless case), with the line charging (BR_B) and the bus shunts (BS)
% set to zero. sh is the susceptance that those shunts put on each bus
% diagonal. It is read from the case data in makeYbus's convention:
% BS / baseMVA at its bus, and per in-service branch b/2 at its to bus and
% (b/2)/|tap|^2 at its from bus. A tap of 0 is read as 1, and the phase shift
% leaves |tap| unchanged. The as-published B_lap then equals B0 - diag(sh).
% The caller records that identity.
define_constants;
nb = size(mpc.bus, 1);
assert(isequal(mpc.bus(:, BUS_I), (1:nb)'), 'e9:buses', 'the buses are not numbered 1..%d', nb);
mpcl = mpc;
mpcl.branch(:, BR_R) = 0;
mpcl.branch(:, BR_B) = 0;
mpcl.bus(:, BS) = 0;
B0 = -imag(full(makeYbus(mpcl.baseMVA, mpcl.bus, mpcl.branch)));
b = mpc.branch(:, BR_STATUS) .* mpc.branch(:, BR_B);
tap = mpc.branch(:, TAP);
tap(tap == 0) = 1;
sh = mpc.bus(:, BS) / mpc.baseMVA ...
    + accumarray(mpc.branch(:, F_BUS), b / 2 ./ abs(tap).^2, [nb 1]) ...
    + accumarray(mpc.branch(:, T_BUS), b / 2, [nb 1]);
end

% ---------------------------------------------------------------------------
function c = columns()
c = {'id', 'nGFL', 'nGFM', 'lam_min_row', 'lam_min_re', 'gSCR_row', 'gSCR_re', ...
    'trace_ok', 'gscr_ma_load', 'gscr_ma_load_eq39', 'gscr_ma_noload', ...
    'gscr_ma_noload_eq39', 'lam_min_noload', 'eigmin_gfm_noload', ...
    'gscr_ma_noshunt', 'gscr_ma_noshunt_eq39', 'lam_min_noshunt', 'eigmin_gfm_noshunt', ...
    'shunt_offdiag_max', 'shunt_diag_resid', 'shunt_s_min', 'shunt_s_sum', ...
    'eigmin_int_noload', 'code_id'};
end

function ids = resume_own(outcsv, code_id, ncol)
% Prepare the worker's own rows file and return the ids already in it. A
% missing file is created with the header. A last line cut by a kill (no
% trailing newline) is dropped through a temporary file and a rename. A
% foreign header, any other malformed line or a row of another code version
% is an error, and the file is not appended to.
header = strjoin(columns(), ',');
ids = zeros(0, 1);
if ~isfile(outcsv)
    write_text(outcsv, sprintf('%s\n', header));
    return
end
txt = fileread(outcsv);
cut = find(txt == newline, 1, 'last');
if isempty(cut)
    error('e9:corrupt', '%s has no complete header line', outcsv);
end
if cut < numel(txt)
    fprintf('dropping the cut last line of %s\n', outcsv);
    tmp = [outcsv '.tmp'];
    write_text(tmp, txt(1:cut));
    movefile(tmp, outcsv, 'f');
    txt = txt(1:cut);
end
lines = strsplit(txt(1:end-1), newline, 'CollapseDelimiters', false);
if ~strcmp(lines{1}, header)
    error('e9:header', '%s has another header; refusing to append', outcsv);
end
[ids, codes, okfmt] = parse_lines(lines(2:end), ncol);
if ~all(okfmt)
    error('e9:corrupt', '%s holds %d malformed lines', outcsv, nnz(~okfmt));
end
if ~all(strcmp(codes, code_id))
    error('e9:identity', '%s holds rows of another code version', outcsv);
end
end

function ids = peer_ids(peer_csv, code_id, ncol)
% Ids of the complete rows of the peer's file written by this code version.
% The peer's file is only read, never changed; a read that fails (the file
% is absent or held) returns no ids and is retried at the next refresh.
ids = zeros(0, 1);
if ~isfile(peer_csv), return; end
try
    txt = fileread(peer_csv);
catch
    return
end
cut = find(txt == newline, 1, 'last');
if isempty(cut), return; end
lines = strsplit(txt(1:cut-1), newline, 'CollapseDelimiters', false);
[pid, codes, okfmt] = parse_lines(lines(2:end), ncol);
ids = pid(okfmt & strcmp(codes, code_id));
end

function [ids, codes, okfmt] = parse_lines(body, ncol)
% Split complete data lines; a line is well formed when it has ncol fields
% and a positive integer id.
n = numel(body);
ids = nan(n, 1);
codes = repmat({''}, n, 1);
okfmt = false(n, 1);
for k = 1:n
    f = strsplit(body{k}, ',', 'CollapseDelimiters', false);
    if numel(f) == ncol
        v = str2double(f{1});
        if isfinite(v) && v >= 1 && v == round(v)
            ids(k) = v;
            codes{k} = f{end};
            okfmt(k) = true;
        end
    end
end
ids = ids(:);
end

function append_line(outcsv, line)
% Append one complete line and close the file, so every finished row is on
% disk before the next one starts.
for attempt = 1:10
    fid = fopen(outcsv, 'a');
    if fid >= 0
        nbytes = fprintf(fid, '%s', line);
        status = fclose(fid);
        if nbytes ~= numel(line) || status ~= 0
            error('e9:io', 'incomplete write to %s', outcsv);
        end
        return
    end
    pause(0.5);
end
error('e9:io', 'cannot open %s for appending', outcsv);
end

function write_text(f, txt)
fid = fopen(f, 'w');
if fid < 0, error('e9:io', 'cannot write %s', f); end
fprintf(fid, '%s', txt);
fclose(fid);
end

function h = code_hash(f)
% First 16 hex digits of the SHA-256 of a text file with carriage returns
% removed, so the identity does not depend on the line-ending convention.
fid = fopen(f, 'r');
b = fread(fid, inf, '*uint8');
fclose(fid);
b(b == 13) = [];
md = java.security.MessageDigest.getInstance('SHA-256');
d = typecast(md.digest(typecast(b, 'int8')), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
h = h(1:16);
end

function r15_jvsm(nmax)
%r15_jvsm  The JVSM grid-strength screen of [13] on the held-out test set.
%   The screen of [13] (arXiv:2607.09249) is measured on the held-out test
%   set of the released bank (1,308 rows). It is a ranking screen that takes
%   the minimum over the converter buses. It sits beside the same-population
%   metrics of r15_same_population_metrics.csv.
%
%   The code follows their (1)-(3), (19)-(21) and Section VI-B. The bus
%   JVSM comes from the WHOLE-SYSTEM driving-point admittance at the bus
%   ("dynamic frequency scan ... from the whole-system EMT model", all
%   devices connected), swept from fmin = 5 Hz to fmax = 100 Hz. J(s) comes
%   from Y(s) by their (3). JVSM_V = int |Q(s)/V(s)| log10(f) df / int
%   log10(f) df, and JVSM_theta is the same on |P(s)/theta(s)|. The
%   combined JVSM = sqrt((JVSM_theta^2 + JVSM_V^2)/2).
%
%   Per-unit reading of their (3): the dq frame is power-invariant
%   (P = vd id + vq iq), so every (3/2)Vp factor maps to V0 in pu. At a
%   scan port that carries no power before the disturbance (a fictitious
%   port with no device, P0 = Q0 = 0), the two transfers used reduce to
%       P(s)/theta(s) = -V0^2 * Ydq(s),   Q(s)/V(s) = V0 * Yqd(s).
%   Y(s) is the driving-point admittance in the orientation of their (2):
%   current INTO the characterized subsystem (here, the whole grid) is
%   positive. Y(s) comes from the state space of the closed system, by dq
%   current injection at the bus C states: Z(s) = Csel (sI - A)^-1 B and
%   Y = inv(Z) (2x2 per bus).
%
%   GATES (they run first, and the bank loop does not start if any fails):
%     G1a transform: the pu transform applied to the analytic I-VSBI
%         admittance of their (2) must reproduce the closed forms
%         (4a)/(4b) to 1e-10 across the sweep grid.
%     G1b quadrature: the NF-point log-grid trapz of |Q/V| must match
%         adaptive quadrature to 1 percent on the I-VSBI. This is the
%         least damped case, with the sharpest resonance this integral sees.
%     G2  extraction: on a hand-built pu state space of exactly the
%         full.rhs form (ideal source behind series R-L + terminal bus C),
%         the Y found by injection must match the analytic branch + shunt
%         admittance to 1e-9 across the sweep.
%   Per-row gate: the recomputed sigma_dom must match the stored dataset
%   row to 1e-6. The scenario is rebuilt exactly from its row, and 1e-6 is
%   the standard tolerance for rebuilt rows.
%
%   Output: results/tables/r15/r15_jvsm_test.csv (checkpointed by id, safe
%   to relaunch) and r15_jvsm_gates.txt (the gate record). The AUC against
%   the unstable label is computed later, on the Python side.
%

if nargin < 1, nmax = inf; end
maxNumCompThreads(2);
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..');
outdir = fullfile(root, 'results', 'tables', 'r15');
fout = fullfile(outdir, 'r15_jvsm_test.csv');
fgate = fullfile(outdir, 'r15_jvsm_gates.txt');

NF = 150;
fgrid = logspace(log10(5), log10(100), NF);
wlg = log10(fgrid);
Wnorm = trapz(fgrid, wlg);

% ---------------- Gate G1: transform + quadrature on the I-VSBI ---------------
R = 0.033; X = 0.33; wb0 = 2*pi*60; Lp = X/wb0;
V0g = 1.02; Q0g = 0.08;
Yiv = @(s) [R + s*Lp, X; -X, R + s*Lp] / ((R + s*Lp)^2 + X^2);
g1a = 0;
for f = fgrid
    s = 1i*2*pi*f;
    Y = Yiv(s);
    Pth = -Q0g - V0g^2 * Y(1,2);              % pu transform, port carries Q0
    QV  =  Q0g/V0g + V0g * Y(2,1);
    Pth_cf = -Q0g - V0g^2 * X / ((R + s*Lp)^2 + X^2);   % their (4a) in pu
    QV_cf  =  Q0g/V0g - V0g * X / ((R + s*Lp)^2 + X^2); % their (4b) in pu
    g1a = max(g1a, abs(Pth - Pth_cf) + abs(QV - QV_cf));
end
QVm = @(f) abs(Q0g/V0g - V0g*X ./ ((R + 1i*2*pi*f*Lp).^2 + X^2));
jv_tr = trapz(fgrid, QVm(fgrid).*wlg) / Wnorm;
jv_qd = integral(@(f) QVm(f).*log10(f), 5, 100, 'AbsTol', 1e-12, ...
    'RelTol', 1e-10) / integral(@(f) log10(f), 5, 100);
g1b = abs(jv_tr - jv_qd)/abs(jv_qd);

% ---------------- Gate G2: state-space extraction, full.rhs form --------------
Bc = 0.02; wb = wb0;
Am = [-wb*R/X,  wb,     -wb/X,  0;
      -wb,     -wb*R/X,  0,    -wb/X;
       wb/Bc,   0,       0,     wb;
       0,       wb/Bc,  -wb,    0];
Bm = [0 0; 0 0; wb/Bc 0; 0 wb/Bc];
Cm = [0 0 1 0; 0 0 0 1];
g2 = 0;
for f = fgrid
    s = 1i*2*pi*f;
    Z = Cm * ((s*eye(4) - Am) \ Bm);
    Yx = inv(Z);
    Ybr = inv([R + s*X/wb, -X; X, R + s*X/wb]);
    Ysh = [s*Bc/wb, -Bc; Bc, s*Bc/wb];
    Yt = Ybr + Ysh;
    g2 = max(g2, norm(Yx - Yt, 'fro')/norm(Yt, 'fro'));
end

fid = fopen(fgate, 'w');
fprintf(fid, 'G1a transform vs (4a)/(4b): max abs err = %.3e (gate < 1e-10)\n', g1a);
fprintf(fid, 'G1b trapz(NF=%d) vs quadrature: rel err = %.3e (gate < 1e-2)\n', NF, g1b);
fprintf(fid, 'G2 state-space extraction vs analytic: max rel err = %.3e (gate < 1e-9)\n', g2);
ok = g1a < 1e-10 && g1b < 1e-2 && g2 < 1e-9;
if ok, fprintf(fid, 'VERDICT: PASS\n'); else, fprintf(fid, 'VERDICT: FAIL\n'); end
fclose(fid);
fprintf('gates: G1a=%.3e G1b=%.3e G2=%.3e -> %s\n', g1a, g1b, g2, ...
    ternary(ok, 'PASS', 'FAIL'));
assert(ok, 'JVSM implementation gate FAILED; bank loop refused.');

% ---------------- The test-set sweep ------------------------------------------
Tid = readtable(fullfile(outdir, 'r15_test_ids.csv'));
ids = Tid.id;
Tb = readtable(fullfile(root, 'data', 'dataset.csv'));
Tb = Tb(ismember(Tb.id, ids), :);
Tb = sortrows(Tb, 'id');
S = table2struct(Tb);
fprintf('test rows: %d\n', numel(S));

done = [];
if isfile(fout)
    Tdone = readtable(fout);
    done = Tdone.id;
    fid = fopen(fout, 'a');
else
    fid = fopen(fout, 'w');
    hdr = 'id,sigma_row,sigma_re,jvsm_min,bus_min,jvsmV_min,jvsmTh_min';
    for j = 1:10, hdr = [hdr sprintf(',jvsm_%d', j)]; end          %#ok<AGROW>
    for j = 1:10, hdr = [hdr sprintf(',jvsmV_%d', j)]; end         %#ok<AGROW>
    fprintf(fid, '%s\n', hdr);
end

t0 = tic; ndone = 0;
for i = 1:numel(S)
    row = S(i);
    if ismember(row.id, done), continue; end
    if ndone >= nmax, break; end
    try
        cfg = bank.row2cfg(row);
        o = run_scenario(cfg);
        assert(isempty(o.fail), o.fail);
        dsig = abs(o.S.sigma_dom - row.sigma_dom);
        assert(dsig < 1e-6, sprintf('sigma gate %.2e', dsig));
        A = o.Asys; M = o.M; nx = size(A, 1);
        pb = [M.dev.bus];
        B = zeros(nx, 20); Cs = zeros(20, nx); V0 = zeros(1, 10);
        for j = 1:10
            b = pb(j);
            B(M.ix_bus(2*b-1), 2*j-1) = M.N.wb / M.N.Bc(b);
            B(M.ix_bus(2*b),   2*j)   = M.N.wb / M.N.Bc(b);
            Cs(2*j-1, M.ix_bus(2*b-1)) = 1;
            Cs(2*j,   M.ix_bus(2*b))   = 1;
            V0(j) = hypot(o.x0(M.ix_bus(2*b-1)), o.x0(M.ix_bus(2*b)));
        end
        aYqd = zeros(10, NF); aYdq = zeros(10, NF);
        In = eye(nx);
        for q = 1:NF
            s = 1i*2*pi*fgrid(q);
            Xs = (s*In - A) \ B;
            for j = 1:10
                Z = Cs(2*j-1:2*j, :) * Xs(:, 2*j-1:2*j);
                Y = inv(Z);                                        %#ok<MINV>
                aYdq(j, q) = abs(Y(1, 2));
                aYqd(j, q) = abs(Y(2, 1));
            end
        end
        jvV = V0 .* (trapz(fgrid, aYqd .* wlg, 2).') / Wnorm;
        jvT = (V0.^2) .* (trapz(fgrid, aYdq .* wlg, 2).') / Wnorm;
        jv = sqrt((jvT.^2 + jvV.^2)/2);
        [jmin, bmin] = min(jv);
        [jvVmin, ~] = min(jvV);
        [jvTmin, ~] = min(jvT);
        fprintf(fid, '%d,%.8g,%.8g,%.8g,%d,%.8g,%.8g', row.id, ...
            row.sigma_dom, o.S.sigma_dom, jmin, pb(bmin), jvVmin, jvTmin);
        fprintf(fid, ',%.8g', jv);
        fprintf(fid, ',%.8g', jvV);
        fprintf(fid, '\n');
        ndone = ndone + 1;
        if mod(ndone, 25) == 0
            fprintf('[%d done this run / %d total] id=%d jvsm_min=%.3f (%.1f min)\n', ...
                ndone, numel(done) + ndone, row.id, jmin, toc(t0)/60);
        end
    catch ME
        fprintf('id=%d FAILED: %s\n', row.id, ME.message);
    end
end
fclose(fid);
fprintf('r15_jvsm complete: %d rows this run (%.1f min).\n', ndone, toc(t0)/60);
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end

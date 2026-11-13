function T = threshold(Yagg_fun, wb, wgrid)
%THRESHOLD  Analytic critical threshold by the realness rule.
%
%   The code tracks the two eigenvalue branches of F^{-1}(jw) Y_agg(jw)
%   over the sub-synchronous band. A threshold candidate is a crossing where
%   an eigenvalue branch becomes purely real with positive real part. The
%   threshold is the largest real root over ALL realness candidates: the
%   interior crossings of the sweep AND the w_c = 0 static/aperiodic root.
%   The static family is computed EXACTLY:
%   F(0) = -J2 gives F(0)^{-1} = J2, so the candidates at the phasor point
%   are eig(J2 Y_agg(0)). Y_agg(0) is real, because the port DC maps and the
%   Y_delta limit -m w^T/c are real. A real positive eigenvalue there is the
%   static/aperiodic marginal condition. For N_c = 1 it is exact: it matches
%   the full-model marginal point found by bisection. For mixed GFM/GFL
%   populations, the static root of the participation-weighted aggregation
%   is conservative. The empirical per-family calibration measures by how
%   much.
%
%   |Imag| at a refined interior crossing is a gate on the tracking quality.
%   A true crossing has |Im| -> 0 under the bisection by construction, so
%   candidates with |Im| above IM_GATE are branch-swap artifacts and are
%   REJECTED (counted in T.n_reject). NO full-model eigenvalue enters the
%   analytic path.
%
%   Yagg_fun : @(w) 2x2 aggregated converter admittance at s = jw; must
%              return the exact s = 0 limit at w = 0 (Ydelta implements it)
%   wgrid    : sweep grid [rad/s]; default 2 : 0.05 : 2*pi*55, the SAME
%              sub-synchronous band edge as the label band of the full
%              model (a narrower sweep would leave an asymmetric dead band)

if nargin < 3, wgrid = 2:0.05:2*pi*55; end
IM_GATE = 1e-6;

pen = @(w) eig(reduced.Fmat(1j*w, wb) \ Yagg_fun(w));

% ---- static root (w_c = 0), exact algebra --------------------------------
C_dc = NaN; im_dc = NaN;
Y0 = Yagg_fun(0);
if all(isfinite(Y0(:)))
    e0 = eig([0 -1; 1 0]*real(Y0));
    for b = 1:2
        if abs(imag(e0(b))) < 1e-9 && real(e0(b)) > 0 && real(e0(b)) > max(C_dc, 0)
            C_dc = real(e0(b));  im_dc = abs(imag(e0(b)));
        end
    end
end

% ---- interior crossings: branch-tracked sweep + bisection refinement ------
cands = [];
n_reject = 0;
e_prev = sortpair(pen(wgrid(1)), []);
for kk = 2:numel(wgrid)
    w = wgrid(kk);
    e = sortpair(pen(w), e_prev);
    for b = 1:2
        if sign(imag(e(b))) ~= sign(imag(e_prev(b))) && abs(imag(e(b))) < 0.5
            % refine by bisection on Imag of the tracked branch
            [wc, lc] = refine(pen, wgrid(kk-1), w, b, e_prev);
            if real(lc) > 0
                if abs(imag(lc)) < IM_GATE
                    cands(end+1, :) = [real(lc), wc, abs(imag(lc))]; %#ok<AGROW>
                else
                    n_reject = n_reject + 1;   % branch-swap artifact
                end
            end
        end
    end
    e_prev = e;
end

% ---- assemble: largest real root over BOTH families -----------------------
T.C_dc = C_dc;  T.im_dc = im_dc;  T.n_reject = n_reject;
if ~isempty(cands)
    [~, ib] = max(cands(:, 1));
    T.C_interior = cands(ib, 1);
    T.w_interior = cands(ib, 2);
    T.im_interior = cands(ib, 3);
    T.all = cands;
else
    T.C_interior = NaN; T.w_interior = NaN; T.im_interior = NaN;
    T.all = zeros(0, 3);
end
if ~isnan(C_dc) && (isnan(T.C_interior) || C_dc >= T.C_interior)
    T.C_SGSI = C_dc;  T.w_c = 0;  T.im_resid = im_dc;  T.rule = 'dc';
elseif ~isnan(T.C_interior)
    T.C_SGSI = T.C_interior;  T.w_c = T.w_interior;  T.im_resid = T.im_interior;
    T.rule = 'interior';
else
    T.C_SGSI = NaN;  T.w_c = NaN;  T.im_resid = NaN;  T.rule = 'none';
end
T.found = ~isnan(T.C_SGSI);
end

% -------------------------------------------------------------------------
function e = sortpair(e, eprev)
if isempty(eprev), e = sort_by_imag(e); return; end
% match branches by proximity to the previous pair
if abs(e(1)-eprev(1)) + abs(e(2)-eprev(2)) > abs(e(1)-eprev(2)) + abs(e(2)-eprev(1))
    e = e([2 1]);
end
end

function e = sort_by_imag(e)
[~, o] = sort(imag(e));
e = e(o);
end

function [wc, lc] = refine(pen, wlo, whi, b, e_lo_pair)
elo = track(pen(wlo), e_lo_pair);
slo = sign(imag(elo(b)));
for it = 1:60
    wm = (wlo + whi)/2;
    em = track(pen(wm), elo);
    if sign(imag(em(b))) == slo
        wlo = wm;  elo = em;
    else
        whi = wm;
    end
end
wc = (wlo + whi)/2;
ef = track(pen(wc), elo);
lc = ef(b);
end

function e = track(e, eprev)
if abs(e(1)-eprev(1)) + abs(e(2)-eprev(2)) > abs(e(1)-eprev(2)) + abs(e(2)-eprev(1))
    e = e([2 1]);
end
end

function out = dominant_root(R, P, seeds)
%DOMINANT_ROOT  Roots of the reduced characteristic equation.
%   The equation is det{ -Y_Gdelta(s) + Btilde_red x F(s) } = 0. The roots
%   come from Newton iteration with a numeric derivative, polished from the
%   given seeds. Returns the rightmost converged root (the reduced
%   counterpart of sigma_dom) and the full converged set.
%
%   R     : reduced.chain output (ports, Leff, BB, p, ...)
%   seeds : column of complex seeds (e.g. the full model's dominant
%           eigenvalue for the correspondence root, plus a coarse grid for
%           the independent search)

BB = R.BB;  wb = P.wb;
isG = strcmp(R.plant_type, 'GFL');

    function D = detfun(s)
        Fs = reduced.Fmat(s, wb);
        YGd = zeros(20);
        for i = 1:10
            ii = 2*i-1:2*i;
            if isG(i)
                YGd(ii, ii) = BB.Sb(i)*R.ports{i}.Y(s);
            else
                prt = R.ports{i};
                Rr = reduced.rot2(prt.th0);
                Yd = Rr*reduced.Ydelta(s, R.Leff(i), prt.v0, prt.i0, prt.cvt, wb)*Rr.';
                YGd(ii, ii) = BB.Sb(i)*Yd;
            end
        end
        D = det(-YGd + kron(BB.Btilde_red, Fs));
    end

    function [smin, smax] = sigmin(s)
        Fs = reduced.Fmat(s, wb);
        YGd = zeros(20);
        for i = 1:10
            ii = 2*i-1:2*i;
            if isG(i)
                YGd(ii, ii) = BB.Sb(i)*R.ports{i}.Y(s);
            else
                prt = R.ports{i};
                Rr = reduced.rot2(prt.th0);
                Yd = Rr*reduced.Ydelta(s, R.Leff(i), prt.v0, prt.i0, prt.cvt, wb)*Rr.';
                YGd(ii, ii) = BB.Sb(i)*Yd;
            end
        end
        sv = svd(-YGd + kron(BB.Btilde_red, Fs));
        smin = sv(end);  smax = sv(1);
    end

roots_found = [];
for k = 1:numel(seeds)
    z = seeds(k);
    ok = false;
    for it = 1:80
        h = 1e-5*(1 + abs(z));
        d0 = detfun(z);
        dp = (detfun(z + h) - detfun(z - h))/(2*h);
        if abs(dp) < 1e-300 || ~isfinite(d0) || ~isfinite(dp), break; end
        step = d0/dp;
        % damp large steps (raw-det Newton can jump wildly)
        if abs(step) > 20, step = 20*step/abs(step); end
        z = z - step;
        if abs(step) < 1e-10*(1 + abs(z)), ok = true; break; end
    end
    if ~ok, continue; end
    % VALIDATE: a true root has sigma_min ~ 0 relative to sigma_max. The test
    % does not depend on the scale of det, and it rejects false Newton
    % convergence.
    [smin, smax] = sigmin(z);
    if smin/smax > 1e-9, continue; end
    if abs(imag(z)) < 0.9*wb && real(z) > -200
        if isempty(roots_found) || all(abs(roots_found - z) > 1e-4*(1 + abs(z)))
            roots_found(end+1, 1) = z;                     %#ok<AGROW>
        end
    end
end
out.roots = roots_found;
if isempty(roots_found)
    out.dom = NaN;
else
    [~, ir] = max(real(roots_found));
    out.dom = roots_found(ir);
end
end

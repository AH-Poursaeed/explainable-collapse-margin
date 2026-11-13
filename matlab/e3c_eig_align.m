function e3c_eig_align
%E3C_EIG_ALIGN  Eigenvalue-level alignment of the full model and the pencils.
%   Compares the full-order model, the full-port pencil and the reduced-port
%   pencil, per k, on the three verification families of Section II-C. It
%   shows the reduced-vs-full alignment at the eigenvalue level, not just in
%   the crossing values.
%
%   For every (family, k), the code computes the sub-synchronous in-band
%   spectrum of the three variants. It writes the NTOP least-damped in-band
%   modes (upper half-plane members of each conjugate pair) and a
%   dominant-mode summary with frequency/damping and full-vs-pencil errors.
%
%   Variants (identical lossy Kron-reduced network realization):
%     full     : full-order model, S = full.modal(Asys)  (348 states)
%     fullport : reduced.pencil_lossy with opts.gfm_full = true
%     redport  : reduced.pencil_lossy as-is (5-state GFM ports)
%
%   Outputs (results/final/tables/):
%     e3c_eig_align.csv    long format: family,kL,k,variant,rank,sigma,
%                          omega,f_Hz,zeta
%     e3c_dom_summary.csv  per (family,k): dominant sigma/f/zeta per variant
%                          + absolute and relative sigma errors vs full
%
%   Runtime ~5 min serial. Same setup as e3b_gfmfull_design.m.

maxNumCompThreads(2);
NTOP = 8;
W_BAND = 2*pi*55;
P = params.sys_const();
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'final', 'tables');
if ~exist(outdir, 'dir'), mkdir(outdir); end

dv2 = repmat({'GFL'}, 1, 10);  dv2{9} = 'VSM';  dv2{10} = 'VSM';
dv1 = repmat({'GFL'}, 1, 10);  dv1{10} = 'VSM';

% families as e3b (full-model k* from the fs3/T3 record: 1.9730 / 2.3926 /
% 1.3926); the k grids are dense toward the collapse point, plus one
% beyond-fold point on the design ray
fams = {
 struct('label', "2-GFM kL=1.0", 'dv', {dv2}, 'kL', 1.0, ...
        'ks', [0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.4 1.6 1.8 1.9 1.95 1.97 2.0]);
 struct('label', "2-GFM kL=0.8", 'dv', {dv2}, 'kL', 0.8, ...
        'ks', [0.5 0.8 1.1 1.4 1.7 2.0 2.2 2.3 2.39]);
 struct('label', "1-GFM kL=1.0", 'dv', {dv1}, 'kL', 1.0, ...
        'ks', [0.5 0.7 0.9 1.0 1.1 1.2 1.3 1.39]);
};

long = {};  summ = {};
for q = 1:numel(fams)
    F = fams{q};
    for k = F.ks
        o = run_scenario(struct('devtype', {F.dv}, 'kL', F.kL, 'k', k));
        if ~isempty(o.fail)
            fprintf('[%s k=%.3f] scenario failed: %s\n', F.label, k, o.fail);
            continue
        end
        R = reduced.chain(o.mpc, o.res, o.plant, P);
        PLr = reduced.pencil_lossy(R, P, o.mpc, o.res, o.plant);
        PLg = reduced.pencil_lossy(R, P, o.mpc, o.res, o.plant, ...
                                   struct('gfm_full', true));
        V = {"full", o.S.lam(~ (abs(o.S.lam) < P.tol.zeroMode)); ...
             "fullport", PLg.lam(abs(PLg.lam) >= 1e-6); ...
             "redport",  PLr.lam(abs(PLr.lam) >= 1e-6)};
        s = struct('family', F.label, 'kL', F.kL, 'k', k);
        for v = 1:size(V, 1)
            lam = V{v, 2};
            lam = lam(abs(imag(lam)) <= W_BAND);      % in-band
            lam = lam(imag(lam) >= -1e-9);            % one per conj pair
            [~, ix] = sort(real(lam), 'descend');
            lam = lam(ix);
            ntake = min(NTOP, numel(lam));
            for r = 1:ntake
                long{end+1} = struct('family', F.label, 'kL', F.kL, ...
                    'k', k, 'variant', V{v, 1}, 'rank', r, ...
                    'sigma', real(lam(r)), 'omega', imag(lam(r)), ...
                    'f_Hz', imag(lam(r))/(2*pi), ...
                    'zeta', -real(lam(r))/max(abs(lam(r)), eps)); %#ok<AGROW>
            end
            dom = lam(1);
            s.(sprintf('sigma_%s', V{v, 1})) = real(dom);
            s.(sprintf('f_%s', V{v, 1})) = abs(imag(dom))/(2*pi);
            s.(sprintf('zeta_%s', V{v, 1})) = -real(dom)/max(abs(dom), eps);
        end
        s.zeta_min_full = o.S.zeta_min;
        s.err_abs_fullport = abs(s.sigma_fullport - s.sigma_full);
        s.err_abs_redport = abs(s.sigma_redport - s.sigma_full);
        s.err_rel_fullport = s.err_abs_fullport/max(abs(s.sigma_full), eps);
        s.err_rel_redport = s.err_abs_redport/max(abs(s.sigma_full), eps);
        summ{end+1} = s;                                        %#ok<AGROW>
        fprintf(['[%s k=%.3f] full %+.4f (f %.2f, zeta %.3f) | fullport ' ...
                 '%+.4f | redport %+.4f\n'], F.label, k, s.sigma_full, ...
                 s.f_full, s.zeta_full, s.sigma_fullport, s.sigma_redport);
    end
end
Tl = struct2table([long{:}]);
Ts = struct2table([summ{:}]);
writetable(Tl, fullfile(outdir, 'e3c_eig_align.csv'));
writetable(Ts, fullfile(outdir, 'e3c_dom_summary.csv'));
fprintf('e3c_eig_align.csv (%d rows) + e3c_dom_summary.csv (%d rows) written\n', ...
    height(Tl), height(Ts));
end

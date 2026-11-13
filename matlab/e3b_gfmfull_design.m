function e3b_gfmfull_design
%E3B_GFMFULL_DESIGN  Full-VSM-port pencil on the design-family boundary.
%   The full-VSM-port pencil variant corrects the 1-GFM family. This script
%   tests whether it also tightens the boundary prediction of the design
%   families. It bisects the pencil crossing (reduced ports vs full GFM
%   ports) on the 2-GFM rays kL in {0.8, 1.0}, and locates the full-port
%   crossing on the 1-GFM family. The full-model k* are the recorded values
%   (fs3: 2.3926 / 1.9730; T3: 1.3926).
%   Output: results/final/tables/e3b_gfmfull.csv

P = params.sys_const();
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'results', 'final', 'tables');
dv2 = repmat({'GFL'}, 1, 10);  dv2{9} = 'VSM';  dv2{10} = 'VSM';
dv1 = repmat({'GFL'}, 1, 10);  dv1{10} = 'VSM';

rows = {};
runs = {
 % devtype  kL    kstar_full  label
 {dv2},     0.8,  2.3926,    '2-GFM kL=0.8';
 {dv2},     1.0,  1.9730,    '2-GFM kL=1.0';
 {dv1},     1.0,  1.3926,    '1-GFM kL=1.0';
};
for q = 1:size(runs, 1)
    dv = runs{q, 1}{1};  kL = runs{q, 2};  kf = runs{q, 3};  lab = runs{q, 4};
    for arm = ["reduced", "gfmfull"]
        opts = struct('gfm_full', arm == "gfmfull");
        kc = pencil_cross(dv, kL, P, opts, kf);
        err = 100*abs(kc - kf)/kf;
        rows{end+1} = struct('family', string(lab), 'ports', arm, ...
            'kstar_full', kf, 'kstar_pencil', kc, 'err_pct', err); %#ok<AGROW>
        fprintf('[%s %s] pencil k* = %.4f (err %.1f%% vs %.4f)\n', ...
            lab, arm, kc, err, kf);
    end
end
T = struct2table([rows{:}]);
writetable(T, fullfile(outdir, 'e3b_gfmfull.csv'));
fprintf('e3b_gfmfull.csv written\n');
end

function kc = pencil_cross(dv, kL, P, opts, kf)
%PENCIL_CROSS scan up from 0.5 then bisect the pencil sigma crossing.
sig = @(k) pget(struct('devtype', {dv}, 'kL', kL, 'k', k), P, opts);
klo = 0.5;  khi = NaN;
slo = sig(klo);
assert(isfinite(slo) && slo < 0, 'start not stable');
for k = 0.6:0.1:(kf + 1.0)
    s = sig(k);
    if ~isfinite(s), break; end
    if s >= 0, khi = k; break; end
    klo = k;
end
if isnan(khi), kc = NaN; return; end
while khi - klo > 1e-3
    km = 0.5*(klo + khi);
    s = sig(km);
    if ~isfinite(s), khi = km; continue; end
    if s >= 0, khi = km; else, klo = km; end
end
kc = 0.5*(klo + khi);
end

function s = pget(cfg, P, opts)
try
    o = run_scenario(cfg, struct('linearize', false));
    if ~isempty(o.fail), s = NaN; return; end
    R = reduced.chain(o.mpc, o.res, o.plant, P);
    PL = reduced.pencil_lossy(R, P, o.mpc, o.res, o.plant, opts);
    s = PL.sigma_dom;
catch
    s = NaN;
end
end

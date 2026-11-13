function cfg = row2cfg(r)
%ROW2CFG  Reconstruct a scenario configuration from a dataset table row.
%   The sensitivity runs, the recourse check and the cold row audit all
%   call this one function, so they rebuild a row in the same way.
gfmb = str2num(char(r.cfgbuses));                              %#ok<ST2NM>
devtype = repmat({'GFL'}, 1, 10);
tune = repmat(struct('f_pll', [], 'Hv', [], 'Dv', [], 'kbw', r.kbw), 10, 1);
buses = 30:39;
for b = 1:10
    if ismember(buses(b), gfmb)
        devtype{b} = 'VSM';
        tune(b).Hv = r.(sprintf('Hv_%d', b));
        tune(b).Dv = r.(sprintf('Dv_%d', b));
    else
        tune(b).f_pll = r.(sprintf('fpll_%d', b));
    end
end
kdisp = zeros(1, 10);
for b = 1:10, kdisp(b) = r.(sprintf('kdisp_%d', b)); end
cfg = struct('devtype', {devtype}, 'kL', r.kL, 'k', r.k, 'kdisp', kdisp, ...
    'tune', tune);
if r.outage > 0, cfg.outage = r.outage; end
end

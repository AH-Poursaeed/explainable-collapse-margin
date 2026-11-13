function Q = device_probe_set(r)
%DEVICE_PROBE_SET  The device-data probes of one bank row.
%   The probes are the M+D device features that are parameters of the
%   full-order model of that scenario. VSM slot b: Hv_b, Dv_b; GFL slot b:
%   fpll_b; all rows: kbw.
%   nGFM is a configuration count, not a model parameter: never probed.
%   W = the width of the parameter's sampled range in build_bank.m (lines
%   58-68 and 141): Hv 2 + 8u -> 8 s; Dv 40 + 160u (main round; top-up 40 +
%   60u) -> 160; f_pll = (8 + 27u)(0.8 + 0.4u'), clip [6, 45] not binding ->
%   [6.4, 42] -> 35.6 Hz; kbw 0.7 + 0.7u -> 0.7.
%   Q: struct array with fields name (tune field), slot, feat (dataset and
%   M+D column), W, base (the row's value).
gfmb = str2num(char(r.cfgbuses));                              %#ok<ST2NM>
buses = 30:39;
Q = struct('name', {}, 'slot', {}, 'feat', {}, 'W', {}, 'base', {});
for b = 1:10
    if ismember(buses(b), gfmb)
        Q(end+1) = struct('name', 'Hv', 'slot', b, 'feat', sprintf('Hv_%d', b), ...
            'W', 8, 'base', r.(sprintf('Hv_%d', b)));                    %#ok<AGROW>
        Q(end+1) = struct('name', 'Dv', 'slot', b, 'feat', sprintf('Dv_%d', b), ...
            'W', 160, 'base', r.(sprintf('Dv_%d', b)));                  %#ok<AGROW>
    else
        Q(end+1) = struct('name', 'f_pll', 'slot', b, 'feat', sprintf('fpll_%d', b), ...
            'W', 35.6, 'base', r.(sprintf('fpll_%d', b)));               %#ok<AGROW>
    end
end
Q(end+1) = struct('name', 'kbw', 'slot', 0, 'feat', 'kbw', 'W', 0.7, 'base', r.kbw);
end

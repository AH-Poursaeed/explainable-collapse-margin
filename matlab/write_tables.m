function write_tables()
%WRITE_TABLES  Write the plant table of the base case.
%   Output: results/tables/T7_plants.csv (and T7_plants.tex) with the bus,
%   type, rating Sn, number of 5 MVA units and dispatch of the ten plant
%   slots (eight GFL, two GFM at buses 38 and 39).

here = fileparts(mfilename('fullpath'));
res = fullfile(here, '..', 'results');
outdir = fullfile(res, 'tables');
if ~exist(outdir, 'dir'), mkdir(outdir); end

% ---------------- T7: plants ------------------------------------------------------
P = params.sys_const();
cfg = struct('devtype', {[repmat({'GFL'},1,8), {'VSM','VSM'}]});
o = run_scenario(cfg, struct('linearize', false));
define_constants;
T7 = table(o.plant.bus, string(o.plant.type), o.plant.Sn, o.plant.Nunit, ...
    o.res.gen(:, PG), o.res.gen(:, QG), ...
    'VariableNames', {'bus', 'type', 'Sn_MVA', 'units_5MVA', 'PG_MW', 'QG_MVAr'});
writetable(T7, fullfile(outdir, 'T7_plants.csv'));
tex_table(T7, fullfile(outdir, 'T7_plants.tex'), 'Plant table (base case)');

fprintf('T7_plants.csv written to results/tables\n');
end

function tex_table(T, fname, caption)
fid = fopen(fname, 'w');
vn = T.Properties.VariableNames;
fprintf(fid, '%% %s\n\\begin{tabular}{%s}\n\\hline\n', caption, repmat('l', 1, numel(vn)));
fprintf(fid, '%s \\\\\n\\hline\n', strjoin(strrep(vn, '_', '\_'), ' & '));
for i = 1:height(T)
    cells = cell(1, numel(vn));
    for j = 1:numel(vn)
        v = T.(vn{j})(i);
        if isnumeric(v)
            if abs(v - round(v)) < 1e-9 && abs(v) < 1e6
                cells{j} = sprintf('%d', round(v));
            else
                cells{j} = sprintf('%.4g', v);
            end
        else
            cells{j} = strrep(char(string(v)), '_', '\_');
        end
    end
    fprintf(fid, '%s \\\\\n', strjoin(cells, ' & '));
end
fprintf(fid, '\\hline\n\\end{tabular}\n');
fclose(fid);
end

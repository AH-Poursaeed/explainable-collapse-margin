function k = state_map(type)
%STATE_MAP  Per-device state index map (14 states each).
switch upper(type)
    case 'GFL'
        names = {'theta','eps_pll','Pm','Qm','chi_d','chi_q','gam_d','gam_q', ...
                 'ifd','ifq','vcd','vcq','igd','igq'};
    case 'VSM'
        names = {'theta','w','Pm','Qm','xiv_d','xiv_q','gam_d','gam_q', ...
                 'ifd','ifq','vcd','vcq','igd','igq'};
    otherwise
        error('dev:state_map', 'unknown type %s', type);
end
for i = 1:numel(names), k.(names{i}) = i; end
k.names = names;
end

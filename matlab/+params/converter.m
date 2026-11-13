function c = converter(type, Sn, P, tune)
%CONVERTER  Parameter factory for one aggregated plant converter.
%   c = params.converter('GFL'|'VSM', Sn_MVA, params.sys_const(), tune)
%
%   Converter-internal per-unit is on the converter base Sn (RMS-phasor-
%   invariant dq: P = vd*id + vq*iq, no 3/2 factor). The only base crossing
%   is the current injected into the network, which the device RHS scales
%   by Sn/Ssys.
%
%   tune (optional struct) overrides scenario-axis parameters:
%     .f_pll  [Hz]   GFL PLL bandwidth            (default 20)
%     .Hv     [s]    GFM virtual inertia          (default 5)
%     .Dv     [pu]   GFM virtual damping          (default 100)
%     .kbw    [-]    current-loop bandwidth scale (default 1)

if nargin < 4, tune = struct(); end
def = @(f, v) getfield_or(tune, f, v);

wb = P.wb;
c.type = upper(type);
c.Sn   = Sn;                 % plant rating [MVA]
c.Ssys = P.Ssys;
c.wb   = wb;
c.nx   = 14;

% ---------------- LCL filter (converter base) -------------------------------
c.Xl1 = 0.06;  c.r1 = 0.005;    % converter-side inductor
c.Xl2 = 0.06;  c.r2 = 0.005;    % grid-side inductor (collector/step-up role)
c.Bcf = 0.05;                   % filter capacitor susceptance
c.fsw = 5000;                   % switching class [Hz] (transmission scale)
c.wres = wb*sqrt((c.Xl1 + c.Xl2)/(c.Xl1*c.Xl2*c.Bcf));
c.fres = c.wres/(2*pi);
c.Rd   = wb/(3*c.wres*c.Bcf);   % passive damping (Reznik rule, pu form)
c.fresOK = (c.fres >= 10*P.f0) && (c.fres <= 0.5*c.fsw);

% ---------------- inner current loop (IMC) ---------------------------------
% 800 Hz nominal: fsw/6 for the 5 kHz class, under f_res/1.9, and TWO
% decades above the 40 Hz voltage loop. At 400 Hz the GFM voltage/current
% cascade goes unstable at 6.4 Hz: a single decade of separation is too
% little.
c.fbw_i = 800*def('kbw',1);
ai = 2*pi*c.fbw_i;
c.Kpi = ai*c.Xl1/wb;
c.Kii = ai*c.r1;
c.Kad = 0.0;                    % capacitor-current active damping (off; Rd passive)
c.kff = 1.0;                    % grid-current/cap-current feedforward (GFM V-loop)

% ---------------- power measurement LPF ------------------------------------
c.wc = 2*pi*5;                  % ~5 Hz (droop/VSM measurement band)

switch c.type
    case 'GFL'
        zeta = 0.7; wn = 2*pi*def('f_pll', 20);
        c.pll_Kp = 2*zeta*wn/wb;       % pu-omega convention (dtheta = (w-1)*wb)
        c.pll_Ki = wn*wn/wb;
        c.pll_wn = wn; c.pll_zeta = zeta;
        c.Kpp = 0.2; c.Kip = 5.0;      % outer P loop -> id ref
        c.Kpq = 0.2; c.Kiq = 5.0;      % outer Q loop -> iq ref
    case 'VSM'
        c.Hv = def('Hv', 5.0);         % virtual inertia [s]
        c.Dv = def('Dv', 100.0);       % virtual damping [pu]
        % Sync-loop identification for the reduced chain (matches the device):
        % dtheta = -F_P(s)/(J_v s^2 + D_v s) dPe  with  F_P = kP/(1 + TP s),
        % kP = 1, TP = 1/wc (the Pm measurement LPF), J_v = 2 Hv/wb and
        % D_v = Dv/wb. This is (1) of the paper divided by w0: the paper keeps
        % w0 in the numerator and uses Dv itself.
        c.kP = 1.0;  c.TP = 1/c.wc;
        c.mq = 0.05;                   % Q-V droop
        c.Rv = 0.2;  c.Xv = 0.0;       % virtual output impedance
        fv = 40;  av = 2*pi*fv;        % voltage loop (IMC on Cf)
        c.Kpv = av*c.Bcf/wb;
        c.Kiv = av*av*c.Bcf/wb/8;      % symmetric-optimum-style integral
    otherwise
        error('params:converter:type', 'unknown type %s', type);
end

% ---------------- hardware-drift hook -----------------
% Optional tune.kX scales the LCL reactances AFTER the loop gains are set:
% the plant drifts while the controller keeps its nameplate tune. The
% default 1 has no effect; the bank and the feasibility runs never set it.
% The drifted plant goes through the port identification again, so the
% realized drift of the stiffening susceptance is measured, not assumed.
kX = def('kX', 1);
if kX ~= 1
    c.Xl1 = c.Xl1*kX;
    c.Xl2 = c.Xl2*kX;
    c.wres = wb*sqrt((c.Xl1 + c.Xl2)/(c.Xl1*c.Xl2*c.Bcf));
    c.fres = c.wres/(2*pi);
end
end

function v = getfield_or(s, f, v)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); end
end

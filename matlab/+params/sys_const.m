function P = sys_const()
%SYS_CONST  Global constants for the 100% IBR IEEE 39-bus build.
%   The one place that sets the bases, tolerances and numerical floors.

P.f0    = 60;                 % system frequency [Hz] (New England, 60 Hz)
P.wb    = 2*pi*P.f0;          % base/system angular frequency [rad/s]
P.Ssys  = 100;                % system base [MVA] (MATPOWER case39 base)

% numerical floors and tolerances
P.Bc_par      = 0.005;        % parasitic bus-shunt susceptance floor [pu]
P.tol.resid   = 1e-9;         % init residual gate (target ~1e-12 typical)
P.tol.zeroMode= 1e-6;         % |lambda| below which a mode is the reference mode
P.tol.jacRel  = 1e-6;         % complex-step vs finite-difference agreement

% conditioning gate (engineering choices, stated here)
P.gate.condA_huge  = 1e10;
P.gate.condeig_big = 1e3;

% master seed
P.seed = 42;
end

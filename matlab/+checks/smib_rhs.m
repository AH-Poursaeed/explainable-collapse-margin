function f = smib_rhs(x, prm)
%SMIB_RHS  RHS for the single-converter infinite-bus harness.
%   States: [dev(14); iline(2); vpcc(2)]; the infinite bus is a fixed phasor.
%   All quantities converter-base pu (Ssys = Sn in the device params).
wb = prm.wb;
xd  = x(1:14);
il  = x(15:16);
vp  = x(17:18);

switch prm.type
    case 'GFL'
        [fd, ID, IQ] = dev.gfl_rhs(xd, vp(1), vp(2), prm.sp);
    case 'VSM'
        [fd, ID, IQ] = dev.gfm_vsm_rhs(xd, vp(1), vp(2), prm.sp);
end

% line PCC -> infinite bus: L di/dt = vpcc - vinf - R i - wb L J2 i
dil = [ (wb/prm.Xl)*(vp(1) - prm.Vinf - prm.Rl*il(1)) + wb*il(2)
        (wb/prm.Xl)*(vp(2) - 0        - prm.Rl*il(2)) - wb*il(1) ];

% PCC capacitor: C dv/dt = I_dev - i_line - wb C J2 v
dvp = [ (wb/prm.Bc)*(ID - il(1)) + wb*vp(2)
        (wb/prm.Bc)*(IQ - il(2)) - wb*vp(1) ];

f = [fd; dil; dvp];
end

function [fd, ID, IQ, aux] = gfm_vsm_rhs(xd, VDi, VQi, sp)
%GFM_VSM_RHS  Averaged dq model of a VSM grid-forming converter (14 states).
%   Virtual swing (Hv, Dv) + Q-V droop + virtual impedance + outer voltage
%   loop + inner current loop + LCL. Converter-base pu. The injection
%   (ID, IQ) is in the network frame, on the SYSTEM base. Complex-step safe.
%
%   The swing equation, linearized: dtheta = -F_P(s)/(J_v s^2+D_v s) dPe
%   with F_P = 1/(1 + s/wc) (the Pm measurement LPF). This is (1) of the
%   paper divided by w0: J_v = 2 Hv/wb and D_v = Dv/wb here; the paper keeps
%   w0 in the numerator and uses Dv itself. params.converter pins kP = 1,
%   TP = 1/wc, so the reduced chain's G_j(s) describes exactly this device.

k = sp.k; c = sp.cvt; wb = c.wb;
th = xd(k.theta); w = xd(k.w); Pm = xd(k.Pm); Qm = xd(k.Qm);
xivd = xd(k.xiv_d); xivq = xd(k.xiv_q); gamd = xd(k.gam_d); gamq = xd(k.gam_q);
ifd = xd(k.ifd); ifq = xd(k.ifq); vcd = xd(k.vcd); vcq = xd(k.vcq);
igd = xd(k.igd); igq = xd(k.igq);

ct = cos(th); st = sin(th);
vpd =  VDi*ct + VQi*st;          % PCC voltage in the GFM local frame
vpq = -VDi*st + VQi*ct;

% power measured at the filter capacitor (GFM-internal measurement point)
P = vcd*igd + vcq*igq;
Q = vcq*igd - vcd*igq;

% Q-V droop -> voltage magnitude reference; virtual output impedance
Es = sp.ref.E0 + c.mq*(sp.ref.Qstar - Qm);
vstar_d = Es - c.Rv*igd + c.Xv*igq;
vstar_q =    - c.Rv*igq - c.Xv*igd;

% outer voltage loop -> current reference (with grid-current/cap feedforward)
ifref_d = c.Kpv*(vstar_d - vcd) + c.Kiv*xivd + c.kff*(igd - w*c.Bcf*vcq);
ifref_q = c.Kpv*(vstar_q - vcq) + c.Kiv*xivq + c.kff*(igq + w*c.Bcf*vcd);

% inner current loop -> converter voltage
vconv_d = c.Kpi*(ifref_d - ifd) + c.Kii*gamd + vcd - w*c.Xl1*ifq - c.Kad*(ifd - igd);
vconv_q = c.Kpi*(ifref_q - ifq) + c.Kii*gamq + vcq + w*c.Xl1*ifd - c.Kad*(ifq - igq);

% LCL (frame speed = VSM speed), passive damping Rd
vmid_d = vcd + c.Rd*(ifd - igd);
vmid_q = vcq + c.Rd*(ifq - igq);
difd = (wb/c.Xl1)*(vconv_d - vmid_d - c.r1*ifd) + w*wb*ifq;
difq = (wb/c.Xl1)*(vconv_q - vmid_q - c.r1*ifq) - w*wb*ifd;
dvcd = (wb/c.Bcf)*(ifd - igd) + w*wb*vcq;
dvcq = (wb/c.Bcf)*(ifq - igq) - w*wb*vcd;
digd = (wb/c.Xl2)*(vmid_d - vpd - c.r2*igd) + w*wb*igq;
digq = (wb/c.Xl2)*(vmid_q - vpq - c.r2*igq) - w*wb*igd;

fd = zeros(sp.nx, 1);
fd(k.theta) = (w - 1)*wb;
fd(k.w)  = (sp.ref.Pstar - Pm - c.Dv*(w - 1)) / (2*c.Hv);   % virtual swing
fd(k.Pm) = c.wc*(P - Pm);
fd(k.Qm) = c.wc*(Q - Qm);
fd(k.xiv_d) = vstar_d - vcd;  fd(k.xiv_q) = vstar_q - vcq;
fd(k.gam_d) = ifref_d - ifd;  fd(k.gam_q) = ifref_q - ifq;
fd(k.ifd) = difd; fd(k.ifq) = difq;
fd(k.vcd) = dvcd; fd(k.vcq) = dvcq;
fd(k.igd) = digd; fd(k.igq) = digq;

IDl = igd*ct - igq*st;  IQl = igd*st + igq*ct;
scl = c.Sn/c.Ssys;
ID = IDl*scl;  IQ = IQl*scl;
if nargout > 3, aux = struct('P',P,'Q',Q,'w',w); end
end

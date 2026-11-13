function [fd, ID, IQ, aux] = gfl_rhs(xd, VDi, VQi, sp)
%GFL_RHS  Averaged dq model of a grid-following converter (14 states).
%   SRF-PLL (d-axis on PCC voltage) + outer P/Q current-reference loops +
%   inner current loop + LCL. Converter-base pu, RMS-phasor-invariant dq
%   (P = vd id + vq iq). The injection (ID, IQ) is in the network frame, on
%   the SYSTEM base. Analytic real arithmetic (complex-step safe).

k = sp.k; c = sp.cvt; wb = c.wb;
th = xd(k.theta); eps = xd(k.eps_pll); Pm = xd(k.Pm); Qm = xd(k.Qm);
chid = xd(k.chi_d); chiq = xd(k.chi_q); gamd = xd(k.gam_d); gamq = xd(k.gam_q);
ifd = xd(k.ifd); ifq = xd(k.ifq); vcd = xd(k.vcd); vcq = xd(k.vcq);
igd = xd(k.igd); igq = xd(k.igq);

ct = cos(th); st = sin(th);
vpd =  VDi*ct + VQi*st;          % PCC voltage in the PLL local frame
vpq = -VDi*st + VQi*ct;          % PLL drives this to zero

% SRF-PLL (pu speed)
wpll = 1 + c.pll_Kp*vpq + c.pll_Ki*eps;

% PCC power (converter pu) and measurement LPFs
Ppcc = vpd*igd + vpq*igq;
Qpcc = vpq*igd - vpd*igq;

% outer P/Q loops -> current references (d on voltage: P ~ vd id, Q ~ -vd iq)
ifref_d =   c.Kpp*(sp.ref.Pstar - Pm) + c.Kip*chid;
ifref_q = -(c.Kpq*(sp.ref.Qstar - Qm) + c.Kiq*chiq);

% inner current loop -> converter voltage (decoupling at PLL freq, vc feedfwd)
vconv_d = c.Kpi*(ifref_d - ifd) + c.Kii*gamd + vcd - wpll*c.Xl1*ifq - c.Kad*(ifd - igd);
vconv_q = c.Kpi*(ifref_q - ifq) + c.Kii*gamq + vcq + wpll*c.Xl1*ifd - c.Kad*(ifq - igq);

% LCL (frame speed = PLL frequency), passive damping Rd on the capacitor leg
vmid_d = vcd + c.Rd*(ifd - igd);
vmid_q = vcq + c.Rd*(ifq - igq);
difd = (wb/c.Xl1)*(vconv_d - vmid_d - c.r1*ifd) + wpll*wb*ifq;
difq = (wb/c.Xl1)*(vconv_q - vmid_q - c.r1*ifq) - wpll*wb*ifd;
dvcd = (wb/c.Bcf)*(ifd - igd) + wpll*wb*vcq;
dvcq = (wb/c.Bcf)*(ifq - igq) - wpll*wb*vcd;
digd = (wb/c.Xl2)*(vmid_d - vpd - c.r2*igd) + wpll*wb*igq;
digq = (wb/c.Xl2)*(vmid_q - vpq - c.r2*igq) - wpll*wb*igd;

fd = zeros(sp.nx, 1);            % auto-promotes to complex under complex-step
fd(k.theta)   = (wpll - 1)*wb;
fd(k.eps_pll) = vpq;
fd(k.Pm)      = c.wc*(Ppcc - Pm);
fd(k.Qm)      = c.wc*(Qpcc - Qm);
fd(k.chi_d)   = sp.ref.Pstar - Pm;
fd(k.chi_q)   = sp.ref.Qstar - Qm;
fd(k.gam_d)   = ifref_d - ifd;
fd(k.gam_q)   = ifref_q - ifq;
fd(k.ifd) = difd; fd(k.ifq) = difq;
fd(k.vcd) = dvcd; fd(k.vcq) = dvcq;
fd(k.igd) = digd; fd(k.igq) = digq;

% injection into the network (rotate to common frame, scale to system base)
IDl = igd*ct - igq*st;  IQl = igd*st + igq*ct;
scl = c.Sn/c.Ssys;
ID = IDl*scl;  IQ = IQl*scl;
if nargout > 3, aux = struct('P',Ppcc,'Q',Qpcc,'wpll',wpll,'vpq',vpq); end
end

function [xd, ref, s] = init_device(type, sp, Vpcc, Ssys_inj)
%INIT_DEVICE  Analytic back-calculation of one converter's states.
%   The states follow from the power flow.
%
%   type     : 'GFL' | 'VSM'
%   sp       : device param struct (.cvt, .k, .nx, .Ssys)
%   Vpcc     : PCC voltage phasor, network frame, system pu
%   Ssys_inj : complex power injected at the bus, system-base pu
%
%   Returns the state vector xd, the reference setpoints ref, and the
%   back-calculated phasor record s (for audits).

c = sp.cvt; k = sp.k;
switch upper(type)
    case 'GFL',  align = 'pcc';
    case 'VSM',  align = 'emf';
end
s = lcl_backcalc(c, Vpcc, Ssys_inj, align);

xd = zeros(sp.nx, 1);
switch upper(type)
    case 'GFL'
        ref.Pstar = s.Ppcc;  ref.Qstar = s.Qpcc;
        % outer P/Q integrators (wpll = 1, vpq = 0 at equilibrium)
        chid =  s.ifd / c.Kip;          % ifref_d = Kip*chid = ifd
        chiq = -s.ifq / c.Kiq;          % ifref_q = -Kiq*chiq = ifq
        gamd = (s.vconvd - s.vcd + c.Xl1*s.ifq + c.Kad*(s.ifd - s.igd))/c.Kii;
        gamq = (s.vconvq - s.vcq - c.Xl1*s.ifd + c.Kad*(s.ifq - s.igq))/c.Kii;
        xd(k.theta) = s.th;  xd(k.eps_pll) = 0;
        xd(k.Pm) = s.Ppcc;   xd(k.Qm) = s.Qpcc;
        xd(k.chi_d) = chid;  xd(k.chi_q) = chiq;
        xd(k.gam_d) = gamd;  xd(k.gam_q) = gamq;
    case 'VSM'
        ref.Pstar = s.Pcap;  ref.Qstar = s.Qcap;  ref.E0 = s.Es;
        % voltage-loop integrators (v* = vc at equilibrium, no Kpv error term)
        xivd = (s.ifd - c.kff*(s.igd - c.Bcf*s.vcq))/c.Kiv;
        xivq = (s.ifq - c.kff*(s.igq + c.Bcf*s.vcd))/c.Kiv;
        gamd = (s.vconvd - s.vcd + c.Xl1*s.ifq + c.Kad*(s.ifd - s.igd))/c.Kii;
        gamq = (s.vconvq - s.vcq - c.Xl1*s.ifd + c.Kad*(s.ifq - s.igq))/c.Kii;
        xd(k.theta) = s.th;  xd(k.w) = 1;
        xd(k.Pm) = s.Pcap;   xd(k.Qm) = s.Qcap;
        xd(k.xiv_d) = xivd;  xd(k.xiv_q) = xivq;
        xd(k.gam_d) = gamd;  xd(k.gam_q) = gamq;
end
xd(k.ifd) = s.ifd; xd(k.ifq) = s.ifq;
xd(k.vcd) = s.vcd; xd(k.vcq) = s.vcq;
xd(k.igd) = s.igd; xd(k.igq) = s.igq;
end

% --------------------------------------------------------------------------
function s = lcl_backcalc(c, Vpcc, Ssys_inj, align)
%LCL_BACKCALC  Steady-state phasors for a converter + LCL from the PF terminal.
%   Voltage pu carries 1:1 across bases; current scales by Ssys/Sn.
scl = c.Sn / c.Ssys;
Iinj_sys = conj(Ssys_inj / Vpcc);            % system-pu current into the bus
ig = Iinj_sys / scl;                         % grid-side current, converter pu
vc = (Vpcc + (c.r2 + 1j*c.Xl2)*ig) / (1 + 1j*c.Rd*c.Bcf);
iff = ig + 1j*c.Bcf*vc;                      % converter-side inductor current
vmid = vc + c.Rd*(iff - ig);
vconv = vmid + (c.r1 + 1j*c.Xl1)*iff;
switch align
    case 'emf'   % GFM: d-axis on the internal EMF behind the virtual impedance
        Eint = vc + (c.Rv + 1j*c.Xv)*ig;
        th = angle(Eint);
    case 'pcc'   % GFL: d-axis on the PCC voltage (PLL lock)
        Eint = vc;
        th = angle(Vpcc);
end
r = exp(-1j*th);
vcL = vc*r; igL = ig*r; ifL = iff*r; vconvL = vconv*r; vpL = Vpcc*r;
s.th = th; s.Es = abs(Eint);
s.vcd = real(vcL); s.vcq = imag(vcL);
s.igd = real(igL); s.igq = imag(igL);
s.ifd = real(ifL); s.ifq = imag(ifL);
s.vconvd = real(vconvL); s.vconvq = imag(vconvL);
s.vpd = real(vpL); s.vpq = imag(vpL);
s.Pcap = s.vcd*s.igd + s.vcq*s.igq;
s.Qcap = s.vcq*s.igd - s.vcd*s.igq;
s.Ppcc = s.vpd*s.igd + s.vpq*s.igq;
s.Qpcc = s.vpq*s.igd - s.vpd*s.igq;
end

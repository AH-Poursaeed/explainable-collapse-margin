function f = rhs(x, M)
%RHS  Full-order ODE right-hand side, common rotating frame at w0.
%   Devices (converter pu, local frames) + branch RL + bus C + load L states.
%   Complex-step safe (analytic arithmetic only).

N = M.N;  wb = N.wb;
nbr = N.nbr; nb = N.nbus; nld = N.nload;

ibr = reshape(x(M.ix_br),  2, nbr);      % [id; iq] per branch
vb  = reshape(x(M.ix_bus), 2, nb);       % [vd; vq] per bus
if nld > 0
    ild = reshape(x(M.ix_load), 2, nld);
else
    ild = zeros(2, 0);
end

f = zeros(size(x));
Iinj = zeros(2, nb);                     % accumulated bus current injections
if ~isreal(x), Iinj = complex(Iinj); end

% ---- devices ----------------------------------------------------------------
for i = 1:numel(M.dev)
    d = M.dev(i); b = d.bus;
    switch d.type
        case 'GFL'
            [fd, ID, IQ] = dev.gfl_rhs(x(d.ix), vb(1,b), vb(2,b), d.params);
        case 'VSM'
            [fd, ID, IQ] = dev.gfm_vsm_rhs(x(d.ix), vb(1,b), vb(2,b), d.params);
    end
    f(d.ix) = fd;
    Iinj(1,b) = Iinj(1,b) + ID;
    Iinj(2,b) = Iinj(2,b) + IQ;
end

% ---- branches: L di/dt = vF/t - vT - r i - wb L J2 i -------------------------
% pu form: di/dt = (wb/x)(vF/t - vT - r i) + wb [iq; -id]
fF = N.f; tT = N.t; tap = N.tap;
for e = 1:nbr
    id = ibr(1,e); iq = ibr(2,e);
    vFd = vb(1,fF(e))/tap(e); vFq = vb(2,fF(e))/tap(e);
    vTd = vb(1,tT(e));        vTq = vb(2,tT(e));
    did = (wb/N.x(e))*(vFd - vTd - N.r(e)*id) + wb*iq;
    diq = (wb/N.x(e))*(vFq - vTq - N.r(e)*iq) - wb*id;
    f(M.ix_br(2*e-1)) = did;
    f(M.ix_br(2*e))   = diq;
    % KCL contributions: leaves the from bus (through the tap), enters the to bus
    Iinj(1,fF(e)) = Iinj(1,fF(e)) - id/tap(e);
    Iinj(2,fF(e)) = Iinj(2,fF(e)) - iq/tap(e);
    Iinj(1,tT(e)) = Iinj(1,tT(e)) + id;
    Iinj(2,tT(e)) = Iinj(2,tT(e)) + iq;
end

% ---- series-RL loads: L di/dt = v - Rs i - wb L J2 i --------------------------
% pu form: di/dt = (wb/Xs)(v - Rs i) + wb [iq; -id]
for l = 1:nld
    b = N.loadbus(l);
    id = ild(1,l); iq = ild(2,l);
    f(M.ix_load(2*l-1)) = (wb/N.Xs(l))*(vb(1,b) - N.Rs(l)*id) + wb*iq;
    f(M.ix_load(2*l))   = (wb/N.Xs(l))*(vb(2,b) - N.Rs(l)*iq) - wb*id;
    Iinj(1,b) = Iinj(1,b) - id;
    Iinj(2,b) = Iinj(2,b) - iq;
end

% ---- CPL legs: tau di/dt = i_ref(v) - i; bus draws i -----------------
% i_ref = [P v + Q J2' v]/|v|^2 draws exactly P + jQ at any v (smoothed
% constant-power sink). Analytic in (vd, vq): complex-step safe.
if isfield(N, 'ncpl') && N.ncpl > 0
    icp = reshape(x(M.ix_cpl), 2, N.ncpl);
    for l = 1:N.ncpl
        b = N.cplbus(l);
        vd = vb(1,b); vq = vb(2,b);
        v2 = vd*vd + vq*vq;
        ird = (N.Pc(l)*vd + N.Qc(l)*vq)/v2;
        irq = (N.Pc(l)*vq - N.Qc(l)*vd)/v2;
        f(M.ix_cpl(2*l-1)) = (ird - icp(1,l))/N.tauc;
        f(M.ix_cpl(2*l))   = (irq - icp(2,l))/N.tauc;
        Iinj(1,b) = Iinj(1,b) - icp(1,l);
        Iinj(2,b) = Iinj(2,b) - icp(2,l);
    end
end

% ---- bus capacitors: C dv/dt = sum(I) - G v - wb C J2 v -----------------------
% pu form: dv/dt = (wb/Bc)(Iinj - G v) + wb [vq; -vd]
for b = 1:nb
    vd = vb(1,b); vq = vb(2,b);
    f(M.ix_bus(2*b-1)) = (wb/N.Bc(b))*(Iinj(1,b) - N.G(b)*vd) + wb*vq;
    f(M.ix_bus(2*b))   = (wb/N.Bc(b))*(Iinj(2,b) - N.G(b)*vq) - wb*vd;
end
end

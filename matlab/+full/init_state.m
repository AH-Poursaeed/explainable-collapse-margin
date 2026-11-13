function [x0, M, resid] = init_state(M, mpc, res)
%INIT_STATE  Analytic back-calculation of the full state vector.
%   It starts from the converged power flow, fills the device reference
%   setpoints and returns the initialization residual ||rhs(x0)||_inf. This
%   residual detects sign and wiring errors; expect a value of order 1e-12.

define_constants;
N = M.N;
V = res.bus(:, VM) .* exp(1j*res.bus(:, VA)*pi/180);   % bus phasors

x0 = zeros(M.nx, 1);

% ---- bus voltages ------------------------------------------------------------
for b = 1:N.nbus
    x0(M.ix_bus(2*b-1)) = real(V(b));
    x0(M.ix_bus(2*b))   = imag(V(b));
end

% ---- branch currents: (r + jx) I = VF/t - VT ---------------------------------
for e = 1:N.nbr
    I = (V(N.f(e))/N.tap(e) - V(N.t(e))) / (N.r(e) + 1j*N.x(e));
    x0(M.ix_br(2*e-1)) = real(I);
    x0(M.ix_br(2*e))   = imag(I);
end

% ---- series-RL load currents: (Rs + jXs) I = V --------------------------------
for l = 1:N.nload
    I = V(N.loadbus(l)) / (N.Rs(l) + 1j*N.Xs(l));
    x0(M.ix_load(2*l-1)) = real(I);
    x0(M.ix_load(2*l))   = imag(I);
end

% ---- CPL leg currents: I = (Pc - jQc) V / |V|^2 ----------------------
if isfield(N, 'ncpl') && N.ncpl > 0
    for l = 1:N.ncpl
        b = N.cplbus(l);
        I = (N.Pc(l) - 1j*N.Qc(l)) * V(b) / abs(V(b))^2;
        x0(M.ix_cpl(2*l-1)) = real(I);
        x0(M.ix_cpl(2*l))   = imag(I);
    end
end

% ---- device bus net injections required by the DYNAMIC network ----------------
% The device must supply the branch KCL surplus at its bus:
% I_dev(b) = jBc_eff V(b) + G(b) V(b) + I_load(b) + sum_branch_out(b).
% This equals the PF injection EXCEPT for the parasitic-C floor delta at the
% device bus (and any rounding). The code computes it exactly from the
% network, so the equilibrium is exact by construction.
Idev = zeros(N.nbus, 1) + 0j;
for b = 1:N.nbus
    Idev(b) = (N.G(b) + 1j*N.Bc(b)) * V(b);
end
for l = 1:N.nload
    b = N.loadbus(l);
    Idev(b) = Idev(b) + V(b)/(N.Rs(l) + 1j*N.Xs(l));
end
if isfield(N, 'ncpl') && N.ncpl > 0
    for l = 1:N.ncpl
        b = N.cplbus(l);
        Idev(b) = Idev(b) + (N.Pc(l) - 1j*N.Qc(l)) * V(b) / abs(V(b))^2;
    end
end
for e = 1:N.nbr
    I = (V(N.f(e))/N.tap(e) - V(N.t(e))) / (N.r(e) + 1j*N.x(e));
    Idev(N.f(e)) = Idev(N.f(e)) + I/N.tap(e);
    Idev(N.t(e)) = Idev(N.t(e)) - I;
end

% ---- devices -------------------------------------------------------------------
for i = 1:numel(M.dev)
    d = M.dev(i);  b = d.bus;
    Sinj = V(b) * conj(Idev(b));          % exact system-pu injection required
    [xd, ref] = dev.init_device(d.type, d.params, V(b), Sinj);
    x0(d.ix) = xd;
    M.dev(i).params.ref = ref;
end

f0 = full.rhs(x0, M);
resid = norm(f0, inf);
end

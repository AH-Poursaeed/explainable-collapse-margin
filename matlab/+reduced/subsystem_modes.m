function out = subsystem_modes(R, P, lam_i)
%SUBSYSTEM_MODES  Modes of the 2x2 modal subsystem as a DESCRIPTOR eigenproblem.
%   The subsystem is the equivalent single-infeed system of network
%   direction lam_i: the participation-weighted aggregate device behind
%   the modal inductive tie lam_i F(s).
%
%     devices : sum_i p_i Y_i(s), realized as blkdiag of the GFL port state
%               spaces and the reduced GFM devices (gfm_red_ss)
%     network : d iL/dt = -w0 J2 iL + w0 V,  I_net = lam_i * iL
%     KCL     : sum_i p_i I_dev,i - I_net = 0   (V algebraic)
%
%   QZ eigenvalues, |lam| < 1e5 retained; sigma_dom over the sub-synchronous
%   band |Im| <= 0.9 w0 (out-of-band LCL/filter modes reported separately).
%   This gives the subsystem's sub-synchronous dominant root without the
%   extra zeros that a scalar-determinant root search lets in for detailed
%   rational ports.

if nargin < 3, lam_i = R.lam_min; end
wb = P.wb;
J2 = [0 -1; 1 0];
isG = strcmp(R.plant_type, 'GFL');

blocks = {};
for i = 1:10
    if isG(i)
        prt = R.ports{i};
        blocks{end+1} = struct('A', prt.A, 'B', prt.B, 'C', R.p(i)*prt.C, ...
                               'D', R.p(i)*prt.D);                  %#ok<AGROW>
    else
        G = reduced.gfm_red_ss(R.ports{i}, R.Leff(i), wb);
        blocks{end+1} = struct('A', G.A, 'B', G.B, 'C', R.p(i)*G.C, ...
                               'D', R.p(i)*G.D);                    %#ok<AGROW>
    end
end

nx = sum(cellfun(@(b) size(b.A, 1), blocks));
N = nx + 2 + 2;                      % + iL (2) + algebraic V (2)
A = zeros(N);  E = zeros(N);
E(1:nx+2, 1:nx+2) = eye(nx+2);
iV = N-1:N;
i0 = 0;
Crow = zeros(2, nx);  Dsum = zeros(2);
for q = 1:numel(blocks)
    b = blocks{q};
    n = size(b.A, 1);
    A(i0+(1:n), i0+(1:n)) = b.A;
    A(i0+(1:n), iV) = b.B;
    Crow(:, i0+(1:n)) = b.C;
    Dsum = Dsum + b.D;
    i0 = i0 + n;
end
iL = nx + (1:2);
A(iL, iL) = -wb*J2;
A(iL, iV) = wb*eye(2);
% KCL rows
A(iV, 1:nx) = Crow;
A(iV, iV)   = Dsum;
A(iV, iL)   = -lam_i*eye(2);

lam = eig(A, E);
lam = lam(isfinite(lam) & abs(lam) < 1e5);
out.lam = lam;
inb = abs(imag(lam)) <= 0.9*wb;
lamB = lam(inb);
[out.sigma_dom, im] = max(real(lamB));
out.lam_dom = lamB(im);
out.oob_maxRe = max([real(lam(~inb)); -inf]);
end

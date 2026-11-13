function S = modal(Asys, P)
%MODAL  Eigenanalysis of the full model in the sub-synchronous band.
%   It deflates the reference mode, restricts the modes to the
%   sub-synchronous band and applies a conditioning gate.
%
%   The method covers the sub-synchronous converter-network interaction band.
%   So sigma_dom and zeta_min are taken over the modes with
%   |Im(lambda)| <= w_band (2 pi 55 rad/s), after the single structural
%   zero/reference mode is removed. Super-synchronous modes (network LC
%   resonances of the EMT model, converter-filter resonances) are NOT
%   ignored: the gate S.oob_stable checks that every out-of-band mode is
%   stable. Any violation must flag the scenario (the caller does this).
%
%   Returns:
%     S.sigma_dom, S.zeta_min : band-limited quantities for labels
%     S.lam_dom, S.lam_zeta   : the associated eigenvalues
%     S.oob_maxRe, S.oob_stable : out-of-band stability gate
%     S.nzero  : number of near-zero modes (gate: exactly 1)
%     S.cond   : deflated condition number; S.trust_dom conditioning gate

w_band = 2*pi*55;

[V, D] = eig(Asys);
lam = diag(D);
s = condeig(Asys);

nearzero = abs(lam) < P.tol.zeroMode;
S.nzero = nnz(nearzero);

inband = ~nearzero & (abs(imag(lam)) <= w_band);
oob    = ~nearzero & ~inband;

sv = svd(Asys);
S.condFull = sv(1)/max(sv(end), eps);
if numel(sv) >= 2, S.cond = sv(1)/max(sv(end-1), eps); else, S.cond = S.condFull; end

lamB = lam(inband);
zeta = -real(lamB)./max(abs(lamB), eps);
[S.zeta_min, iz] = min(zeta);
[S.sigma_dom, is] = max(real(lamB));
kk = find(inband);
S.i_zeta = kk(iz);  S.i_dom = kk(is);
S.lam_dom  = lam(S.i_dom);
S.lam_zeta = lam(S.i_zeta);
S.f_dom = abs(imag(S.lam_dom))/(2*pi);

S.oob_maxRe  = max([real(lam(oob)); -inf]);
S.oob_stable = S.oob_maxRe < 0;

S.lam = lam;  S.inband = inband;  S.V = V;  S.condeig = s;
S.trust_dom = ~(S.cond > P.gate.condA_huge && s(S.i_dom) > P.gate.condeig_big);
S.w_band = w_band;
end

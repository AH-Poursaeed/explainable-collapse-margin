function R = rot2(th)
%ROT2  Planar rotation e^{J2 th}, J2 = [[0,-1],[1,0]].
R = [cos(th), -sin(th); sin(th), cos(th)];
end

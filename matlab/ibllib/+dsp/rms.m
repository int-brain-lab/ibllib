function r = rms(X, dim)

if nargin<=1, dim=1; end
r = sqrt(mean(X.^2, dim));

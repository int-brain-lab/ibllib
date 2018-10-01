function X = fexpand(X, Ns)
% X = real(ifft(dsp.fexpand(X)))
% X = real(ifft(dsp.fexpand(X, Ns)))
% Reconstruct the negative frequencies prior to ifft. If number of output samples is not supplied, assumes an odd number.

if isempty(X), return, end
if nargin < 2, Ns = 1; end

siz = size(X);
siz(1) = siz(1)*2 -1 - (~mod(Ns,2));
X = cat(1, X(:,:), conj(flipud(X(2:end-(~mod(Ns,2)),:))));
X = reshape(X,siz);
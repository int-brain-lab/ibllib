function X = freduce(X)
% X = dsp.freduce(fft(X))
% Reduces the spectrum to positive frequencies (single sided)

if isempty(X), return, end

siz = size(X);

siz(1) = floor(size(X,1)/2+1) ;
X((siz(1)+1):end,:) = [];

X = reshape(X,siz);
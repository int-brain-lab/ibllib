function X = resample(X, Si_in, Si_ou)
% X = dsp.resample(X, Si_in, Si_ou)
% resample signal using interpolation in frequency domain

[Nech_in Ntr] = size(X);
Nech_ou = round(Nech_in * Si_in/Si_ou);
X = dsp.freduce(fft(X));
fscale_in = dsp.fscale(Nech_in, Si_in, 'real');
fscale_ou = dsp.fscale(Nech_ou, Si_ou, 'real');

switch 1
    case mod(Si_in / Si_ou,1)==0
        % surechantillonage par in facteur entier
        errordlg('upsample by integer factor not implemented yet')
    case mod(Si_ou / Si_in,1)==0
        % sous-echantillonage par in facteur entier
        errordlg('downsample by integer factor not implemented yet.')
    otherwise % arbitrary resample
        X = interp1(fscale_in, X, fscale_ou);
        X(isnan(X)) = 0;
end

X = real(ifft(dsp.fexpand(X,Nech_ou)));
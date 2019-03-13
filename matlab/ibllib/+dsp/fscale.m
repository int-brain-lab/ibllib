function fscale = fscale(Ns,Si,option)
% fscale = dsp.fscale(Ns,Si)
% fscale = dsp.fscale(Ns,Si,'full')
% fscale = dsp.fscale(Ns,Si,'real')

if nargin <= 2 , option = 'full'; end

% sample the frequency scale
fscale = [0:floor(Ns/2)]/Ns/Si;

switch 1
    case strcmpi(option,'full')
        fscale = [fscale -fscale(end-1+mod(Ns,2) : -1 : 2)]';
    case strcmpi(option,'real')
        fscale = fscale';
        return
end

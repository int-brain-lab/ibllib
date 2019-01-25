function  [fig, Amp, Phi, Freq] = spectrum(W, Si, varargin)
%[fig,Amp,Phi,Freq]=dsp.spectrum(ts,Si);
%[fig,Amp,Phi,Freq]=dsp.spectrum(...,'Legend',{'toto','titi'},'Logx',1,...
%    'Unwrap',0,'Smooth',8 , 'fig', f,'Display',false);
% todo amplitudes as a power spectral density
%'Legend' : default {}
% Logx : default false
% Unwrap : default true
% Smoothed : default 0 (none), otherwise frequency moving average length in Hz
% plotArgs = {'linewidth',2}; %cell of arguments that will be forwarded to the plot function
% Paramètres
p=inputParser;
p.addRequired('W', @isnumeric);
p.addRequired('Si', @isnumeric);
p.addParamValue('Legend', {}, @iscell);
p.addParamValue('Logx',0,@(x) isnumeric(x) | islogical(x));
p.addParamValue('Unwrap',  false ,@(x) isnumeric(x) | islogical(x));
p.addParamValue('Display', true ,@(x) isnumeric(x) | islogical(x));
p.addParamValue('Smoothed',0,@isnumeric);
p.addParamValue('fig',[],@isnumeric);
p.addParamValue('plotArgs',{},@iscell);


p.parse(W,Si,varargin{:});
for ff=fields(p.Results)', eval([ff{1} '=p.Results.' ff{1} ';' ]); end

% Calcul du spectre
Freq=dsp.fscale(size(W,1),Si,'real');
Amp=20*log10(abs(dsp.freduce(fft(W))).*sqrt(2*Si/size(W,1)));
Phi=angle(dsp.freduce(fft(W)));

% Unwrap or not
if Unwrap, Phi = unwrap(Phi); end
Phi = Phi*180/pi;

% Smooth or not
if Smoothed
    nf = round(Smoothed/Freq(2)/2)*2+1;
    Amp = dsp.smooth.mwa(Amp,nf);
    Phi = dsp.smooth.mwa(Phi,nf);
end

if ~Display, return, end

%% All the displaying part
if isempty(fig),
    fig = figure('Color','w','Name','Spectre','numbertitle','off');
    ax= [subplot(2,1,1) ; subplot(2,1,2)];
else
    ax(1) = findobj('Parent',fig,'Type','axes','tag','amp');
    ax(2) = findobj('Parent',fig,'Type','axes','tag','phase');
end
% get the color order if this is not the first plot
col_ind = length(get(ax(1),'Children')); 

for m = 1: size(Amp,2)
    col = color_from_index(col_ind -1 + m);
    plot(ax(1),Freq,Amp(:,m),'color',col, plotArgs{:}); hold(ax(1),'on'),
    plot(ax(2),Freq,Phi(:,m),'color',col, plotArgs{:}); hold(ax(2),'on'),
end
grid(ax(1),'on'),grid(ax(2),'on')
% set the tags on axes
set(ax(1),'Tag','amp','nextplot','add');
set(ax(2),'Tag','phase','nextplot','add');

% Display
xlabel([ax(2)],'Frequency (Hz)')
ylabel([ax(1)],'Amplitude (dB)')
ylabel([ax(2)],'Phase (degrees)')
% légende optionnelle
if ~isempty(Legend), legend(Legend); end
% échelle fréquences log
if Logx~=0,
    set(ax,'Xscale','log','xlim',Freq([1 end]));
end
% link the two axes
linkaxes(ax,'x')

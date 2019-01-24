classdef smooth
    % Smooth is a collection of methods to perform smoothing of a statistical serie:
    % Low-Pass, Moving average, Exponential
    %
    % Low-Pass
    %           ts = dsp.smooth.lp(ts,fcut)
    %           ts = dsp.smooth.lp(ts, [ fcut1 fcut2])
    %           fcut : between [0-1] is the frequency relative to Nyquist
    %
    % Moving Average
    %        [ts_ ]= dsp.smooth.mwa(ts,lwin)
    %        lwin is the length of the averaging window in samples
    %       (NB : it is good practice to have an odd number of samples so the window has a central sample)
    %
    % Exponential smoothing
    %        ts_ = dsp.exp_single(ts,alpha)
    %        alpha is the update factor between 0 and 1
    %
    %
    % % TODO : weighted moving average with a provided window
    % http://connor-johnson.com/2014/02/01/smoothing-with-exponentially-weighted-moving-averages/
    methods(Static)
        %% low pass filter
        function ts_ = lp(ts, fac)
            %ts = Smooth.lp(ts, [ fcut1 fcut2])
            %           fcut : between [0-1] is the frequency relative to Nyquist
            [ts, fnames ] = dsp.smooth.handle_input(ts);
            if length(fac)==1, fac = [fac-eps fac]; end
            lwin =  ceil(max(2./nonzeros(fac))) *2 ; % we keep at least two periods for the padding
            ts_ = [ repmat(ts(1,:),lwin,1) ; ts ; repmat(ts(end,:),lwin,1)]; % padd the serie
            ts_ = dsp.ffilter.lp(ts_ ,1, [1/2.*fac]);
            ts_ = ts_(lwin + [1:length(ts)],:);
            ts_ = dsp.smooth.handle_output(ts_, fnames);
        end
        
        function [ts_ ]= mwa(ts,lwin)
            %% moving average
            % [ts_ ]= Smooth.mwa(ts,lwin)
            % lwin is the length of the averaging window in samples
            % (NB : it is good practice to have an odd number of samples so the window has a central sample)
            [ts, fnames ] = dsp.smooth.handle_input(ts);
            ts_ = [ repmat(ts(1,:),lwin,1) ; ts ; repmat(ts(end,:),lwin,1)];
            ts_ = filter2( ones(lwin,1)./lwin,ts_);
            ts_ = ts_(lwin + [1:length(ts)],:);
            ts_ = dsp.smooth.handle_output(ts_, fnames);
        end
        
        function [ts_ ]= exp_single(ts,alpha)
            % smoothing according to the single exponential smoothing model
            %           ts_ = exp_single(ts,alpha)
            % alpha default is 0.01
            [ts, fnames ] = dsp.smooth.handle_input(ts);
            alpha = 0.01;
            ts_ = ts;
            for n = 1: size(ts,2)
                for m = 1: size(ts,1)-1
                    ts_(m+1,n) = alpha.* ts(m,n) + (1-alpha).*ts_(m,n);
                end
            end
            ts_ = dsp.smooth.handle_output(ts_, fnames);
        end
    end
    
    methods (Access=private , Static=true)
        % Handle input arguments so that vectors, matrices and structures
        % can be handled by the functions
        function [ts, fnames ] = handle_input(ts)
            fnames = {};
            if isstruct(ts),
                fnames  = fieldnames(ts);
                ts = cell2mat(struct2cell(ts)');
            end
        end
        % Same thing : reconstruct output structure if necessary
        function [ts_ ] = handle_output(ts_, fnames)
            if isempty(fnames), return, end
            for f = 1:length(fnames)
                S.(fnames{f}) = ts_(:,f);
            end
            ts_ = S;
        end
        
    end

end





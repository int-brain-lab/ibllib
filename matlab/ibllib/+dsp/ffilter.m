classdef ffilter
    % Fourier domain filters
    
    methods(Static)
        %% low pass filter
        function ts_ = lp(ts, si, b)
            [ns, ntr] = size(ts);
            f = dsp.freduce(dsp.fscale(ns, si));
            filc=(f<=b(1))+(f>b(1) & f<b(2)).*(0.5*(1+sin(pi*(f-((b(1)+b(2))/2))/(b(1)-b(2)))));
            ts_ = real(ifft(bsxfun(@times, fft(ts), dsp.fexpand(filc, ns))));
        end
    end

end






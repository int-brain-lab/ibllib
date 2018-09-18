function tests = test_dsp
tests = functiontests(localfunctions);
end

function test_fscale(testCase)
%% dsp.fscale
res = [0;100;200;300;400;500;-400;-300;-200;-100];
assert(all(abs(dsp.fscale(10,0.001)-res)< 1e4))
assert(all(abs(dsp.fscale(11,0.001,'real')-res(1:6))< 1e4))

res = [0;90.9090909090909;181.818181818182;272.727272727273;363.636363636364;454.545454545455;-454.545454545455;-363.636363636364;-272.727272727273;-181.818181818182;-90.9090909090909];
assert(all(abs(dsp.fscale(11,0.001)-res)< 1e4))
assert(all(abs(dsp.fscale(11,0.001,'real')-res(1:6))< 1e4))
end

function test_freduce(testCase)
%% dsp.freduce
%test odd
res = [0;90.9090909090909;181.818181818182;272.727272727273;363.636363636364;454.545454545455;-454.545454545455;-363.636363636364;-272.727272727273;-181.818181818182;-90.9090909090909];
assert(all( dsp.freduce(res) == res(1:6)))
% test even
res = [0;100;200;300;400;500;-400;-300;-200;-100];
assert(all( dsp.freduce(res) == res(1:6)))
% test 2D
assert(all( flatten( dsp.freduce(repmat(res,1,2)) == repmat(res(1:6),1, 2))))
% test 3D
assert(all( flatten( dsp.freduce(repmat(res,1,2,3))  == repmat(res(1:6),1, 2,3))))
end

function test_expand(testCase)
%% dsp.fexpand
% test odd
res = rand(11,1);
X = dsp.freduce(fft(res));
R = real(ifft(dsp.fexpand(X,11)));
assert( all( abs(R - res) < 1e6))
R = real(ifft(dsp.fexpand(X))); % single arg
assert( all( abs(R - res) < 1e6))
% test even
res = rand(10,1);
X = dsp.freduce(fft(res));
X = real(ifft(dsp.fexpand(X,10)));
assert( all( abs(X - res) < 1e6))
% test 2D
res = rand(10,2);
X = dsp.freduce(fft(res));
X = real(ifft(dsp.fexpand(X,10)));
assert( all( abs(X(:) - res(:)) < 1e6))
% test 3D
res = rand(10,2,3);
X = dsp.freduce(fft(res));
X = real(ifft(dsp.fexpand(X,10)));
assert( all( abs(X(:) - res(:)) < 1e6))
end
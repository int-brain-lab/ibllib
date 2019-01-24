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

%% Test ffilter
function test_ffilter(testCase)
ts = rand(500,1).*200+3400;
si = 0.001;
ts_ = dsp.ffilter.lp(ts, si, [150 200]);
% mean(ts) mean(ts_)
% dsp.Spectrum([ts ts_], 0.001)
testCase.assertTrue(abs(mean(ts)-mean(ts_)) < 1e-4)
ts = rand(500,2).*200+3400;
ts_ = dsp.ffilter.lp(ts, si, [150 200]);
testCase.assertTrue(all(abs(mean(ts)-mean(ts_)) < 1e-4))
end

%% Test Smooth
function test_smooth(testCase)
%%
Depth = [1:500]';
Vp = rand(500,1).*200+3400;
Vs = Vp*2/3;
Rho = ones(500,1)+1.5+rand(500,1).*0.1;
C = struct('Vp',Vp,'Vs',Vs);

% single serie input
[vp1] = dsp.smooth.lp(Vp, [0.1 0.2]);
assert(all(size(vp1)==size(Vp)))
[vp2] = dsp.smooth.mwa(Vp, 10);
assert(all(size(vp2)==size(Vp)))
[vp3] = dsp.smooth.exp_single(Vp, 0.01);
assert(all(size(vp3)==size(Vp)))

% array input
[vpvs1] = dsp.smooth.lp([Vp Vs], [0.1 0.2]);
assert(all(size(vpvs1)==size([Vp Vs])))
assert(all(vpvs1(:,1)==vp1))
[vpvs2] = dsp.smooth.mwa([Vp Vs], 10);
assert(all(size(vpvs2)==size([Vp Vs])))
assert(all(vpvs2(:,1)==vp2))
[vpvs3] = dsp.smooth.exp_single([Vp Vs], 0.1);
assert(all(size(vpvs3)==size([Vp Vs])))
assert(all(vpvs3(:,1)==vp3))

% structure input
[c1] = dsp.smooth.lp(C, [0.1 0.2]);
assert(all(flatten( cell2mat(struct2cell(c1)')==vpvs1 )))
[c2] = dsp.smooth.mwa(C, 10);
assert(all(flatten( cell2mat(struct2cell(c2)')==vpvs2 )))
[c3] = dsp.smooth.exp_single(C, 0.1);
assert(all(flatten( cell2mat(struct2cell(c3)')==vpvs3 )))
end










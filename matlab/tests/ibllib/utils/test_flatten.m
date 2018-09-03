function tests = test_flatten
tests = functiontests(localfunctions);
end

function test_flatten_divers(testCase)
tso.a = {'toto','titi','tata'}';
tso.b = {'toto',[],'tata'}';
tso.c = tso.a;
tso.d = [1,3,4]';
tso.e = [1,NaN,4]';
tso.f = [NaN,NaN,NaN]';

ts = struct('a', {'toto','titi','tata'}, 'b', {'toto',[],'tata'}, 'c',{{'toto'},{'titi'},{'tata'}}, 'd',{1,3,4}, 'e',{1,[],4}, 'f',{[],[],[]} );

tsf = flatten(ts);

verifyTrue(testCase,  all(cellfun(@(x,y) all(x==y), tsf.a,  tso.a)))
verifyTrue(testCase,  all(cellfun(@(x,y) all(x==y), tsf.b,  tso.b)))
verifyTrue(testCase,  all(cellfun(@(x,y) all(x==y), tsf.c,  tso.c)))

verifyTrue(testCase,  all(tso.d==tsf.d))
verifyTrue(testCase,  all(tso.e(~isnan(tso.e))==tsf.e(~isnan(tsf.e)) ))
verifyTrue(testCase,  all(tso.e(~isnan(tso.f))==tsf.e(~isnan(tsf.f)) ))
end

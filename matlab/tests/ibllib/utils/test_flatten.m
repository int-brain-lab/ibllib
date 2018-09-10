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

ts = struct('a', {'toto','titi','tata'},...
            'b', {'toto',[],'tata'},...
            'c',{{'toto'},{'titi'},{'tata'}},...
            'd',{1,3,4},...
            'e',{1,[],4},...
            'f',{[],[],[]} );

tsf = flatten(ts);

verifyTrue(testCase,  all(cellfun(@(x,y) all(x==y), tsf.a,  tso.a)));
verifyTrue(testCase,  all(cellfun(@(x,y) all(x==y), tsf.b,  tso.b)));
verifyTrue(testCase,  all(cellfun(@(x,y) all(x==y), tsf.c,  tso.c)));

verifyTrue(testCase,  all(tso.d==tsf.d));
verifyTrue(testCase,  all(tso.e(~isnan(tso.e))==tsf.e(~isnan(tsf.e)) ));
verifyTrue(testCase,  all(tso.e(~isnan(tso.f))==tsf.e(~isnan(tsf.f)) ));
end


function test_flatten_recursive(testCase)
sn = struct('nest_number',{1;2}, 'nest_str',{'n1';'n2'});
sm = struct('nest_number',{3;4}, 'nest_str',{'m1';'m2'});
s = struct('name',{'toto'; 'titi'},'nfield',{sn;sm},'number',{1;2});

expect.name = {'toto';'titi'};
expect.number = [1;2];
expect.nfield = struct('nest_number',{[1;2]; [3;4]}, 'nest_str',{{'n1';'n2'} ; {'m1';'m2'}});
res = flatten(s);


verifyTrue(testCase, isequal(expect.nfield, res.nfield));
verifyTrue(testCase, isequal(expect.name, res.name));
verifyTrue(testCase, isequal(expect.number, res.number));

end

function flatten_test_scalar_structure(testCase)
s = struct('name',{'toto' ; 'titi'},'number',{1;2});
ss.toto = 'titi';
ss.nest = s;
res = flatten(ss);
verifyTrue(testCase, isequal(flatten(sn), res.nest));
end
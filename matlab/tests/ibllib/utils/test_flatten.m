function tests = test_flatten
tests = functiontests(localfunctions);
end

function test_flatten_struct(testCase)
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

% now test with an empty struct
D = flatten(struct(...
    'dataset_id', {},...
    'local_path', {}));
verifyFalse(testCase, isempty(D))

% test with a scalar struct:
D = struct('a', 1, 'b', 'turlu', 'c', [1 2]);
D = flatten(D);
verifyTrue(testCase, ischar(D.b));
D = flatten(D, 'wrap_scalar', true);
verifyTrue(testCase, iscell(D.b));
end

function test_flatten_arrays(testCase)
% simple array
a = round(rand(5,2));
assertTrue(testCase, all(a(:)==flatten(a)))
% test with logicals !
a = logical(a);
assertTrue(testCase, all(a(:)==flatten(a)))
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

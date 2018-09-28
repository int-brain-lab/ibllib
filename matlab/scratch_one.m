testCase.one = One('alyx_login', 'test_user', 'alyx_pwd', 'TapetesBloc18',...
    'alyx_url', 'https://test.alyx.internationalbrainlab.org');
testCase.eid = '86e27228-8708-48d8-96ed-9aa61ab951db';

%%
% eid, ses = one.search(users='olivier', date_range=['2018-08-24', '2018-08-24'])
% pprint(eid)


chem = '/media/owinter/CGG/IBL/Globus';
block_files = io.dir(chem, 'pattern', '*_Block.mat', 'recursive', true);


%%
clear test
m = 1;
data = load(block_files{1}); data = data.block;
wval = data.inputs.wheelValues;
wt = data.inputs.wheelTimes-data.events.expStartTimes;

test(m).wval = wval(1:50000);
test(m).wt = wt(1:50000);
[mon,mof] = wheel.findWheelMoves3(test(m).wval, test(m).wt, 1000, []);
test(m).mon = mon;
test(m).mof = mof;

m=2;
data = load(block_files{3}); data = data.block;
wval = data.inputs.wheelValues;
wt = data.inputs.wheelTimes-data.events.expStartTimes;

test(m).wval = wval;
test(m).wt = wt;
[mon,mof] = wheel.findWheelMoves3(test(m).wval, test(m).wt, 1000, []);
test(m).mon = mon;
test(m).mof = mof;


m=length(test)+1;
data = load(block_files{103}); data = data.block;
wval = data.inputs.wheelValues;
wt = data.inputs.wheelTimes-data.events.expStartTimes;

test(m).wval = wval;
test(m).wt = wt;
[mon,mof] = wheel.findWheelMoves3(test(m).wval, test(m).wt, 1000, []);
test(m).mon = mon;
test(m).mof = mof;

%%
save('/home/owinter/MATLAB/Rigbox/wheelAnalysis/test_wheel.mat', 'test')

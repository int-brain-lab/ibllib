testCase.one = one('alyx_login', 'test_user', 'alyx_pwd', 'TapetesBloc18',...
    'alyx_url', 'https://test.alyx.internationalbrainlab.org');
testCase.eid = '86e27228-8708-48d8-96ed-9aa61ab951db';

%%
% eid, ses = one.search(users='olivier', date_range=['2018-08-24', '2018-08-24'])
% pprint(eid)

eids  = testCase.one.search('users','olivier','date_range',datenum(['2018-08-24'; '2018-08-24']));
eids_ = testCase.one.search('users','olivier','date_range',(['2018-08-24'; '2018-08-24']));
assert(eids == eids_)

eids = testCase.one.search('users','olivier');

testCase.one.search('users','etsitunexistaispas')


eids  = testCase.one.search('users',{'olivier', 'nbonacchi'})

        usr = ['olivier', ]
        sl, sd = myone.search(users=usr, details=True)
        self.assertTrue(isinstance(sl, list) and isinstance(sd, list))
        self.assertTrue(all([set(usr).issubset(set(u)) for u in [s['users'] for s in sd]]))
        # when the user is a string instead of a list
        sl1, sd1 = myone.search(users=['olivier'], details=True)
        sl2, sd2 = myone.search(users='olivier', details=True)
        self.assertTrue(sl1 == sl2 and sd1 == sd2)
        # test for the dataset type
        dtyp = ['spikes.times', 'titi.tata']
        sl, sd = myone.search(dataset_types=dtyp, details=True)
        self.assertTrue(len(sl) == 0)
        dtyp = ['channels.site']
        sl, sd = myone.search(dataset_types=dtyp, details=True)
        self.assertTrue(len(sl) == 2)
        dtyp = ['spikes.times', 'channels.site']
        sl, sd = myone.search(dataset_types=dtyp, details=True)
        self.assertTrue(len(sl) == 1)
        # test empty return for non-existent user
        self.assertTrue(len(myone.search(users='asdfa')) == 0)

classdef test_one < matlab.unittest.TestCase
 
    properties
        one
        eid
        eid_light
    end
 
    methods(TestMethodSetup)
        function createObject(testCase)
            testCase.one = one('alyx_login', 'test_user', 'alyx_pwd', 'TapetesBloc18',...
                'alyx_url', 'https://test.alyx.internationalbrainlab.org');
            testCase.eid = '86e27228-8708-48d8-96ed-9aa61ab951db';
        end
    end
 
    methods(Test)
 
        function test_list(testCase)
            eid_ = '86e27228-8708-48d8-96ed-9aa61ab951db';
            eids_ = {'86e27228-8708-48d8-96ed-9aa61ab951db', '3bca3bef-4173-49a3-85d7-596d62f0ae16'};
            %% first functionality is a straight list of datasets without keyword
            % test for a single session, list datasets
            [dt, details]= testCase.one.list(eid_);
            testCase.verifyEqual(length(dt),29);
            % test for multiple sessions, list datasets
            [dt, details]= testCase.one.list(eids_);
            testCase.verifyEqual(length(dt),29);
            testCase.verifyEqual(length(details.url),34);
            %% second functionality is a straight query to the endpoint
            [list ~] = testCase.one.list([],'keyword','data');
            testCase.assertTrue(length(list)>40);
            testCase.one.list([],'keyword','subject');
            testCase.one.list([],'keyword','users');
%             testCase.one.list([],'keyword','labs'); % Need to update Alyx test for this to work
        end
%         for eid in EIDs:
%             dt = one.list(eid)  # returns dataset-type
%             dt = one.list(eid, details=True)
%             dt = one.list(eid, keyword='dataset-type')  # returns list
%             dt = one.list(eid, keyword='dataset-type', details=True)  # returns SessionDataInfo
%             for key in ('subject', 'users', 'lab', 'type', 'start_time', 'end_time'):
%                 dt = one.list(eid, keyword=key)  # returns dataset-type
%                 print(key, ': ', dt)
%             ses = one.list(eid, keyword='all')
%             usr, ses = one.list(eid, keyword='users', details=True)
% 
%     def test_list_error(self):
%         one = self.One
%         a = 0
%         eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
%         try:
%             one.list(eid, keyword='tutu')  # throws an error
%         except KeyError:
%             a = 1
%             pass
%         self.assertTrue(a == 1)
        
        function test_search(testCase)
            % test datenum string and double
            eids  = testCase.one.search('users','olivier','date_range',datenum(['2018-08-24'; '2018-08-24']));
            eids_ = testCase.one.search('users','olivier','date_range',(['2018-08-24'; '2018-08-24']));
            testCase.verifyEqual(eids , eids_, testCase.eid)
            % test empty
            eids = testCase.one.search('users','etsitunexistaispas');
            testCase.assertTrue(isempty(eids));
            % test 2 users and single session output
            [eids, ses]= testCase.one.search('users',{'olivier', 'nbonacchi'});
            % test single user string and multiple sessions output
            [eids, ses]= testCase.one.search('users','olivier');
            % test for the dataset type
            dtyp = {'spikes.times', 'titi.tata'};
            [eids, ses]= testCase.one.search('dataset_types',dtyp);
            testCase.assertTrue(isempty(eids))
            dtyp = 'channels.site';
            [eids, ses]= testCase.one.search('dataset_types',dtyp);
            testCase.assertTrue(length(eids)==2)
            % test automatic typo correction
            [eids_]= testCase.one.search('data',dtyp);
            testCase.assertEqual(eids, eids_);
            dtyp = {'spikes.times', 'channels.site'};
            [eids, ses] = testCase.one.search('dataset_types',dtyp);
            % test empty return for non-existent user
            [eids, ses] = testCase.one.search('subjects',{'turlu'});
        end
        
        function test_load(testCase)
            % Test with 3 actual datasets predefined
            dataset_types = {'clusters.peakChannel', 'clusters._phy_annotation', 'clusters.probes'};
            eid_ = 'https://test.alyx.internationalbrainlab.org/sessions/86e27228-8708-48d8-96ed-9aa61ab951db';
            [pc, pa, cp] = testCase.one.load(eid_, 'dataset_types', dataset_types);
            % same with dclass output and check correspondency
            D = testCase.one.load(eid_, 'dataset_types', dataset_types, 'dclass_output', true);
            testCase.assertTrue(all(pc==D.data{1}) & all(pa==D.data{2}) & all(cp==D.data{3}))
            % Test with a session that doesn't have any dataset on the Flat Iron
            dataset_types = {'wheel.velocity', 'wheel.timestamps'};
            [a1, a2] = testCase.one.load(eid_, 'dataset_types', dataset_types);
            testCase.assertTrue(isempty(a1) & isempty(a2))
            % Test with an empty dataset interleaved
            dataset_types = {'clusters.peakChannel', 'turlu', 'clusters.probes'};
            [pc_, tu, cp_] = testCase.one.load(eid_, 'dataset_types', dataset_types);
            testCase.assertTrue(all(pc==pc_) & isempty(tu) & all(cp==cp_))
            % test a single dtype in scalar
            a = testCase.one.load(testCase.eid, 'dataset_types', 'eye.area');
            % Test without a dataset list should download everything and output a dictionary
            eid_ = '3bca3bef-4173-49a3-85d7-596d62f0ae16';
            a = testCase.one.load(eid_);
            testCase.assertTrue(length(a.data) == 5)
        end
        
    end
end
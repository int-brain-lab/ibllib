classdef test_one_tutorial < matlab.unittest.TestCase
 
    properties
        one
        eid
        eid2
    end
 
    methods(TestMethodSetup)
        function createObject(testCase)
            testCase.one = One('alyx_login', 'test_user', 'alyx_pwd', 'TapetesBloc18',...
                'alyx_url', 'https://test.alyx.internationalbrainlab.org');
        end
    end
 
    methods(Test)
 
        function test_tuto(testCase)
            
            one = testCase.one;
            
            [eid, ses] = one.search('users', {'olivier'}, 'date_range', datenum([2018 8 24 ; 2018 8 24])) ;
            
            one.search;
            
            one.list(eid);
            
            [dtypes details] = one.list(eid);
            
            ses_info = one.list(eid, 'keyword', 'all')
            
            one.list([],'keyword', 'labs');
            one.list([],'keyword', 'datasets');
            one.list([],'keyword', 'users');
            one.list([],'keyword', 'subjects');
            
            dataset_types = {'clusters.templateWaveforms', 'clusters.probes', 'clusters.depths'};
            eid = 'cf264653-2deb-44cb-aa84-89b82507028a';
            [wf, pr, d ]= one.load(eid, 'data' ,dataset_types);

            assert(~isempty(wf))
            assert(~isempty(pr))
            assert(~isempty(d))
        end
        
    end
end
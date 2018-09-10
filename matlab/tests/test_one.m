classdef test_one < matlab.unittest.TestCase
 
    properties
        one
        eid
        eid_light
    end
 
    methods(TestMethodSetup)
        function createObject(testCase)
            testCase.one = one('user', 'test_user', 'password', 'TapetesBloc18',...
                'base_url', 'https://test.alyx.internationalbrainlab.org');
            testCase.eid = '86e27228-8708-48d8-96ed-9aa61ab951db';
        end
    end
 
    methods(Test)
 
        function test_list(testCase)
            r1 = testCase.one.list(testCase.eid)
            testCase.verifyEqual(length(r1),29);
        end
        
        
        
    end
end
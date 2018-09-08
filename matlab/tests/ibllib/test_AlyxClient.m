classdef test_AlyxClient < matlab.unittest.TestCase
 
    properties
        ac
    end
 
    methods(TestMethodSetup)
        function createObject(testCase)
            testCase.ac = AlyxClient('user', 'test_user', 'password', 'TapetesBloc18',...
                'base_url', 'https://test.alyx.internationalbrainlab.org');
        end
    end
 
    methods(Test)
 
        function test_get_sessions(testCase)
            % tests automatic replacement of base_url or not
            r1 = testCase.ac.get_session('86e27228-8708-48d8-96ed-9aa61ab951db');
            r2 = testCase.ac.get_session(...
                'https://test.alyx.internationalbrainlab.org/sessions/86e27228-8708-48d8-96ed-9aa61ab951db');
            testCase.verifyEqual(r1,r2)
        end
    end
end
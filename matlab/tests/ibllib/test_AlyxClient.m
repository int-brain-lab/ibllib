classdef test_AlyxClient < matlab.unittest.TestCase
 
    properties
        ac
        subjects
        water_types
    end
 
    methods(TestMethodSetup)
        function createObject(testCase)
            testCase.ac = AlyxClient('user', 'test_user', 'password', 'TapetesBloc18',...
                'base_url', 'https://test.alyx.internationalbrainlab.org');
            testCase.subjects = testCase.ac.get('/subjects');
            testCase.assertTrue(length(testCase.subjects.nickname)>=2)
            testCase.water_types = testCase.ac.get('/water-type');
            testCase.assertEqual(testCase.water_types.name, {'Water'; 'Hydrogel'})
        end
    end
 
    methods(Test)
 
        function test_endpoint_url_format(testCase)
            sub = testCase.ac.get('subjects/flowers');            
            sub2 = testCase.ac.get('/subjects/flowers');
            testCase.assertEqual(sub, sub2);
            sub2 = testCase.ac.get([testCase.ac.base_url '/subjects/flowers']);
            testCase.assertEqual(sub, sub2);
        end
        
        function test_get_sessions(testCase)
            % tests automatic replacement of base_url or not
            r1 = testCase.ac.get_session('86e27228-8708-48d8-96ed-9aa61ab951db');
            r2 = testCase.ac.get_session(...
                'https://test.alyx.internationalbrainlab.org/sessions/86e27228-8708-48d8-96ed-9aa61ab951db');
            testCase.verifyEqual(r1,r2);
        end
        
        function test_create_delete_water_admin(testCase)
            wa_ = struct(...
                'subject', testCase.subjects.nickname{1},...
                'date_time', time.serial2iso8601(now),...
                'water_type', 'Water',...
                'user', testCase.ac.user,...
                'adlib', true,...
                'water_administered', 0.52);
            rep = testCase.ac.post('/water-administrations', wa_);
            testCase.assertTrue(rep.water_administered == 0.52);
            % read after write
            rep = testCase.ac.get(rep.url);
            testCase.assertTrue(rep.water_administered == 0.52);
            % now delete the water administration
            testCase.ac.delete(rep.url);
            try
                testCase.ac.get(rep.url)    
                flag = false;
            catch err
                testCase.assertEqual(err.identifier, 'MATLAB:webservices:HTTP404StatusCodeError');
                flag = true;
            end
            testCase.assertTrue(flag);
        end
    end
end
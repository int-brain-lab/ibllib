classdef test_io_read < matlab.unittest.TestCase
    
    methods(Test)
        function test_json_params(testCase)
            par = struct('a','tata','o','toto', 'i','titi', 'num', 1);
            fpar = io.write.jsonpref('test_io_prefs', par);
            par_ = io.read.jsonpref('test_io_prefs');
            testCase.assertEqual(par, par_)
            delete([io.getappdir filesep '.' 'test_io_prefs'])
        end
        
        function test_empty_params(testCase)
           par = io.read.jsonpref('this_doesnotexist');
           testCase.assertEmpty(par)
        end
    end
end
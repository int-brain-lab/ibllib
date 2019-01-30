classdef test_io_read < matlab.unittest.TestCase
    
    properties
        prefs
    end
    
    methods(TestMethodSetup)
        function create_dir_structures(testCase)
            testCase.prefs.data_folder = '';
            try, testCase.prefs.data_folder = getpref('unittests', 'DATA_FOLDER'); end
        end
    end
    
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
        
        function test_nii_read(testCase)
            % reads all nii files provided in the test folder.
            if ~isdir(testCase.prefs.data_folder), return, end
            nii_files = io.dir(testCase.prefs.data_folder, 'pattern', '*.nii');
            for m=1:length(nii_files)
               [v, h] = io.read.nii(nii_files{m});
               testCase.assertFalse(isempty(v))
               testCase.assertTrue(prod(h.Dimensions) == prod(size(v)))
            end
        end
        
        function test_json_readwrite(testCase)
            par = struct('a','tata','o','toto', 'i','titi', 'num', 1, 'chem', 'some\text\backslashes');
            json_file = tempname;
            io.write.json(json_file, par);
            par_ = io.read.json(json_file);
            testCase.assertEqual(par, par_);
            delete(json_file)
        end
    end
end

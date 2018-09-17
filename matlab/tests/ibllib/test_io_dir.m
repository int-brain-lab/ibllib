classdef test_io_dir < matlab.unittest.TestCase
    
    properties
        tdir
        results
    end
    
    
    methods(TestMethodSetup)
        function create_dir_structures(testCase)
            testCase.tdir = [tempdir 'iotest' filesep];
            % create a directory with subfolders and files
            mkdir([testCase.tdir 'subdir' filesep 'subsubdir'])
            % create the full desired output of the io.dir function as a
            % dunction of pattern
            testCase.results(1).ext = '*.to';
            testCase.results(1).files = {...
                [testCase.tdir 'toto1.to'], ...
                [testCase.tdir 'subdir' filesep 'toto2.to'],  ...
                [testCase.tdir 'subdir' filesep 'subsubdir' filesep 'toto3.to']};
            
            testCase.results(2).ext = '*.ta';
            testCase.results(2).files = {...
                [testCase.tdir 'subdir' filesep 'subsubdir' filesep 'toto4.ta']};
            
            testCase.results(3).ext = '*.*';
            testCase.results(3).files = [testCase.results(1).files  testCase.results(2).files];
            
            for m = 1:length(testCase.results(3).files)
               fid = fopen(testCase.results(3).files{m}, 'w+');
               fwrite(fid,'toto','char');
               fclose(fid);
            end
            
        end
    end
    
    
    methods(TestMethodTeardown)
        function delete_dir(testCase)
            rmdir(testCase.tdir, 's');
        end
    end
    
    
    methods(Test)
        function test_recursive(testCase)
            % for every pattern and list of files, check output against
            % desired output
            for t = 1:length(testCase.results)
                res = io.dir(testCase.tdir, 'pattern', testCase.results(t).ext);
                testCase.assertEqual(testCase.results(t).files(:), res);
            end
        end
    end
end
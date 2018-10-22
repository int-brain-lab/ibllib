classdef test_npy < matlab.unittest.TestCase
    properties
        npyFile
    end
    
    methods(TestMethodSetup)
        function get_file(testCase)
            testCase.npyFile =  tempname;
        end
    end
    
    
    methods(TestMethodTeardown)
        function delete_file(testCase)
            delete(testCase.npyFile)
        end
    end
    
    methods(Test)
        % TODO need to setup a suite of tests with Python inputs, 3d
        % arrays etc...
        function test_readwrite(testCase)
            a = {magic(4), [1:15], [1:15]'};
            for m = 1:length(a)
                io.write.npy(testCase.npyFile, a{m});
                b = io.read.npy(testCase.npyFile);
                testCase.assertEqual(a{m},b);
            end
        end
    end
end
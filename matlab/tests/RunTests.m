function varargout = RunTests(files)
% RunTests()
% prints a list of tests available to run
% RunTests('All')
% runs all available tests
% RunTests('/home/owinter/PycharmProjects/IBL_Main/ibllib/matlab/tests/utils/test_flatten.m')
% runs a single (string) or a list (cell) of tests
varargout={}; res=[];
switch true
    case nargin==0
        test_files = get_test_files();
        for m = 1:length(test_files)
            disp(['runtests(''' test_files{m} ''');'])
        end
    case nargin==1 && ischar(files) && strcmp(files,'All')
        test_files = get_test_files();
        varargout = {RunTests(test_files)};
    case nargin==1 && ischar(files) && ~strcmp(files,'All')
        varargout = {runtests(files)};
    case nargin==1 && iscell(files)
        for m = 1:length(files)
            res = [res, runtests(files{m})];
        end
        varargout = {res};
end


function test_files = get_test_files()
chem = mfilename('fullpath');
chem = chem(1 : find(chem==filesep, 1, 'last'));
% chem = '/home/owinter/PycharmProjects/IBL_Main/ibllib/matlab/tests/';
test_files = io.dir(chem,'pattern', 'test_*.m');
% remove matlab autosave files
test_files = test_files(~cellfun(@(x) x(end)=='~', test_files));

classdef AlyxClientRigBox < AlyxClient
    
    methods
        % Constructor, pas touche
        function self = AlyxClientRigBox(varargin)
            self@AlyxClient(varargin{:})
        end
    end
    
    methods(Static)
        function Extract_Path(chem, varargin)
            % cellArrayOfFullPathBLockFiles = io.dir(chem, 'pattern', '*_Block.mat', 'recursive', true);
            % AlyxClientRigBox.Extract_Path(cellArrayOfFullPathBLockFiles)
            % AlyxClientRigBox.Extract_Path('/data/subject/a-session-i-forgot-to-register/')
            % AlyxClientRigBox.Extract_Path(...,'mainRepositoryPath', mainRepo, 'force', true)
            %% Parameters
            dp = dat.paths;
            % handle input parameters
            p = inputParser;
            addParameter(p,'mainRepositoryPath', dp.mainRepository)
            addParameter(p,'force', false)
            parse(p,varargin{:});
            for fn = fieldnames(p.Results)'; eval([fn{1} '= p.Results.' (fn{1}) ';']); end
            %% extract ALF files with the good convention
            if iscell(chem)
                block_files = chem;
            else
                block_files = io.dir(chem, 'pattern', '*_Block.mat', 'recursive', true);
            end
            for m =103:length(block_files)
                disp([ num2str(m,'%03.0f/') num2str(length(block_files),'%03.0f') '   '  block_files{m}])
                d = load(block_files{m});
                cpath = fileparts(block_files{m});
                npy_exist = isempty(io.dir(cpath, 'pattern', '_ibl_*'));
                npy_exist = npy_exist && isempty(io.dir(cpath, 'pattern', '_rigbox_*'));
                if npy_exist && ~force, continue, end
                switch true
                    case ispc
                        [a,b] = dos(['del ' cpath filesep '_ibl_*']);
                        [a,b] = dos(['del ' cpath filesep '_rigbox_*']);
                    case isunix
                        [a,b] = unix(['rm ' cpath filesep '_ibl_*']);
                        [a,b] = unix(['rm ' cpath filesep '_rigbox_*']);
                end
                % extract the files
                output_files = alf.block2ALF(d.block, 'expPath', cpath, 'namespace', '_ibl_');
            end
        end
    end
    
end

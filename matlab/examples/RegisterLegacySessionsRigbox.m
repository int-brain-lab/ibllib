%% Params
%NB: go to https://dev.alyx.internationalbrainlab.org/admin/data/datarepository/ and pick the name of the repo you want to register from
% you can only register from one repo at a time (this is the price of being portable)
repository_name = 'mainenlab_rig01' ; 
excluded_subjects = {'default'};
chem = 'C:\Globus';
% one = One('alyx_login','root','alyx_pwd','oldpass','alyx_url','https://dev.alyx.internationalbrainlab.org');
one = One('alyx_login','olivier','alyx_pwd','superpass','alyx_url','https://alyx.internationalbrainlab.org');

%% Gather info about repository
dr = one.alyx_client.get('/data-repository');
host_name =  dr.dns{find(strcmp( dr.name, repository_name))};
% dataset types, prepare the regexp to check files before register
dtypes = one.alyx_client.get('/dataset-types');
dtypes = structfun(@(x) x(~cellfun(@isempty, dtypes.filename_pattern)), dtypes, 'UniformOutput', false);
dtypes.regexp = regexptranslate('wildcard', dtypes.filename_pattern);

%% Gather info about Sessions
% recursive dir to find all Block files
alfses.block_files = unique(io.dir(chem, 'pattern', '*_Block.mat'));
% get the full path to each session
alfses.session_dirs = cellfun(@(x) x(1:find(x==filesep,1,'last')-1), alfses.block_files, 'UniformOutput', false);
% get the subject, date and session number for each session
alfses.bigsplit = cellfun( @(x) strsplit(x,filesep), alfses.session_dirs,'UniformOutput', false);
alfses.subjects =  cellfun(@(x) x{end-2}, alfses.bigsplit, 'UniformOutput', false);
alfses.yyyymmdd =  cellfun(@(x) x{end-1}, alfses.bigsplit, 'UniformOutput', false);
alfses.sessionnb = cellfun(@(x) x{end  }, alfses.bigsplit, 'UniformOutput', false);
% filter out the excluded subjects as defined above
alfses = structfun(@(x) x(~ismember(alfses.subjects, excluded_subjects)) , alfses, 'UniformOutput', false);

%% The big loop over sessions
for m = 1:length(alfses.block_files)
    disp(alfses.block_files{m})
    % get or create session
    [sesid, ses] = one.search('subject', alfses.subjects{m}, 'date_range',...
        datenum(alfses.yyyymmdd{m}).*[1 1],'number',alfses.sessionnb{m});
    try
        assert(length(ses) <= 1)
    catch
        error('Ambiguity as there are several sessions with the same subject, same number on the same day')
    end
    if isempty(sesid) % If the session is empty, create the session
        d = load(alfses.block_files{m});
        ses = [];
        ses.subject = alfses.subjects{m};
        ses.procedures = {'Behavior training/tasks'};
        ses.narrative =  'auto-generated legacy session';
        ses.start_time =  datestr(d.block.startDateTime,'yyyy-mm-ddTHH:MM:SS');
        ses.end_time =  datestr(d.block.endDateTime,'yyyy-mm-ddTHH:MM:SS');
        ses.type =  'Experiment';
        ses.number =  alfses.sessionnb{m};
        ses.users =  {one.alyx_client.user};
        ses = one.alyx_client.post('/sessions', ses);
        warning('Create session')
    else
        ses.data_dataset_session_related;
    end
    % Create the register file structure
    alfil.full_path = io.dir(alfses.session_dirs{m}, 'pattern', '*.*.*','recursive', true);
    alfil.folders =cellfun(@(x) fileparts(x), alfil.full_path, 'UniformOUtput', false );
    alfil.name = cellfun(@(x) x(find(x==filesep,1,'last')+1:end), alfil.full_path, 'UniformOutput', false);
    alfil = sel_struct(alfil, cellfun(@(x) sum(x=='.')>=2, alfil.name));
    % make sure that all files have a pattern that match an existing type. If not warn
    dtype_recognized = cellfun(@(x) any(~cellfun(@isempty , regexp(x, dtypes.regexp))), alfil.name);
    if sum(~dtype_recognized )
        unrecognized_list = (alfil.name(~dtype_recognized));
        alfil = structfun(@(x) x(dtype_recognized), alfil, 'UniformOutput', false);    
    end
    % look at the files that are already up there and do not register them
    if ~isempty(ses.data_dataset_session_related)
        [~,i2reg] = setxor(alfil.name,ses.data_dataset_session_related.name);
        if isempty(i2reg), continue, end
        alfil = structfun(@(x) x(i2reg), alfil, 'UniformOutput', false);
        warning('Sessions already exist and found datasets to upload !')
        disp(alfil.name)
    end
    if sum(~dtype_recognized )
        warning('Skipping unknown datasetypes below: ')
        disp(unrecognized_list)
    end
    % loop over each folder containing files to register to batch register all files within the folder
    for folder = unique(alfil.folders)'
        indf = find(strcmp(alfil.folders, folder{1}));
        % find the relative path
        icut = strfind(folder{1}, [alfses.subjects{m} filesep alfses.yyyymmdd{m} filesep alfses.sessionnb{m}]);
        relative_path = folder{1}(icut:end);
        % create the registration strucure
        register_struct.created_by = one.alyx_client.user;
        register_struct.dns = host_name;
        register_struct.path =  relative_path;
        register_struct.filenames = alfil.name(indf);
        rep = one.alyx_client.post('/register-file', register_struct);
    end
end

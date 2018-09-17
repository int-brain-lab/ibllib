function varargout = load(self, eid, varargin)


%% handles input arguments
TYPO_PROOF = {  ...
    'data', 'dataset_types';...
    'dataset', 'dataset_types';...
    'datasets', 'dataset_types';...
    'dataset-types', 'dataset_types';...
    'dataset_types', 'dataset_types';...
    'dtypes', 'dataset_types';...
    'dtype', 'dataset_types';...
    };
% substitute eventual typo with the proper parameter name
for  ia = 1:2:length(varargin)
    it = find(strcmpi(varargin{ia}, TYPO_PROOF(:,1)),1);
    if isempty(it), continue; end
    varargin(ia) = TYPO_PROOF(it,2);
end
% parse input arguments
p = inputParser;
addOptional(p,'dataset_types', {});
addOptional(p,'dry_run',false);
addOptional(p,'force_replace',false);
addOptional(p, 'dclass_output', false);
parse(p,varargin{:});
for fn = fieldnames(p.Results)', eval([fn{1} '= p.Results.' (fn{1}) ';']); end
if ischar(dataset_types), dataset_types = {dataset_types}; end

%% real stuff
% eid could be a full URL or just an UUID, reformat as only UUID string
eid = strsplit(eid, '/'); eid = eid{end};
ses = self.alyx_client.get_session(eid);
% if the dtypes are empty, request full download and output a dclass
if isempty(dataset_types)
    dataset_types = ses.data_dataset_session_related.dataset_type;
    dclass_output = true;
end
[~, ises, iargin] = intersect(ses.data_dataset_session_related.dataset_type, dataset_types);

%% Create the data structure
D = flatten(struct(...
    'dataset_id', ses.data_dataset_session_related.id(ises),...
    'local_path', repmat({''}, length(ises), 1),...
    'dataset_type', ses.data_dataset_session_related.dataset_type(ises),...
    'url', ses.data_dataset_session_related.data_url(ises),...
    'eid', repmat({eid}, length(ises), 1 )), 'wrap_scalar', true);
D.data = cell(length(ises), 1);

%% Loop over each dataset and read if necessary
for m = 1:length(ises)
    url_server_side = strrep( D.url{m},  self.par.HTTP_DATA_SERVER, '');
    % create the local path while keeping ALF convention for folder structure
    if isunix
        local_path = [self.par.CACHE_DIR url_server_side];
    else
        local_path = [self.par.CACHE_DIR strrep(url_server_side, '/', filesep)];
    end
    if ~dry_run && (force_replace || ~exist(local_path, 'file'))
        disp(['Downloading ' local_path])
        res =  self.ftp.mget(url_server_side, self.par.CACHE_DIR);
        assert(strcmp(res, local_path))
    end
    % loads the data
    D.local_path{m} = local_path;
    [~, ~, ext] = fileparts(local_path);
    switch ext
        case '.npy'
            D.data{m} = io.read.npy(local_path);
        otherwise
            warning(['Dataset extension not supported yet: *' ext])
    end
end
% sort the output structure according to the input order
id = [];
if ~isempty(D.dataset_type)
    [~, id] = ismember(dataset_types, D.dataset_type(:));
    D = structfun(@(x) x(nonzeros(id)), D, 'UniformOutput', false);
end
%% Handle the output
if dclass_output
    varargout = {D};
else
    varargout = cell(length(dataset_types), 1);
    varargout(sort(iargin)) = D.data(sort(nonzeros(id)));
end


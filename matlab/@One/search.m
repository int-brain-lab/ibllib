function [eids,ses] = search(self,varargin)
% [eids,ses] = one.search('keyword', values)
% [eids,ses] = one.search('data', {'channels.brainLocation'})
% [eids,ses] = one.search('subjects', 'LEW008')
% [eids,ses] = one.search(..., 'users', {'miles', 'morgan'}) % sessions with 2 users miles and morgan
% [eids,ses] = one.search(..., 'date-range', datenum([2018 8 28 ; 2018 8 31]) )

%% Typo proof parameters
% first column are possible input arguments and values are actual fields
SEARCH_TERMS = {  ...
    'data', 'dataset_types';...
    'dataset', 'dataset_types';...
    'datasets', 'dataset_types';...
    'dataset-types', 'dataset_types';...
    'dataset_types', 'dataset_types';...
    'users', 'users';...
    'user', 'users';...
    'subject', 'subjects';...
    'subjects', 'subjects';...
    'date_range', 'date_range';...
    'date-range', 'date_range';...
    'lab', 'labs';...
    'labs', 'labs';
};
% substitute eventual typo with the proper parameter name
for  ia = 1:2:length(varargin)
    it = find(strcmpi(varargin{ia}, SEARCH_TERMS(:,1)),1);
    assert(~isempty(it));
    varargin(ia) = SEARCH_TERMS(it,2);
end

%% Handle parameters
p = inputParser;
addParameter(p,'dataset_types', {})
addParameter(p,'users', {})
addParameter(p,'subjects', {})
addParameter(p,'labs', {})
addParameter(p,'date_range', [], @(x) (isnumeric(x) & any(numel(x)==[0 2]) | ischar(x)))
addParameter(p,'details', false)
parse(p,varargin{:});
for fn = fieldnames(p.Results)', eval([fn{1} '= p.Results.' (fn{1}) ';']); end
%%
if nargin ==1, eids = SEARCH_TERMS; ses=[]; return, end
if ischar(dataset_types), dataset_types = {dataset_types}; end
if ischar(users), users = {users}; end
if ischar(subjects), subjects = {subjects}; end
if ischar(labs), labs = {labs}; end
if ~isempty(date_range) && isa(date_range,'double')
    date_range = mat2cell(datestr(date_range, 'yyyy-mm-dd'),[1 1],10);
elseif ~isempty(date_range) && ischar(date_range)
    date_range = mat2cell(date_range,[1 1],10);
end
% create the url to send to the REST API
url = '/sessions?';
url = append2url(url,'dataset_types', dataset_types);
url = append2url(url,'users', users);
url = append2url(url,'subject', subjects);
url = append2url(url,'date_range', date_range);
url = append2url(url,'labs', labs);

ses = self.alyx_client.get(url);
if isempty(ses), [ses, eids] = deal([]); return, end
% url = arrayfun(@(x) x.url(end-35:end), ses, 'UniformOutput', false) % si pas de flatten
if ischar(ses.url)
    eids = ses.url((end-35:end));
else
    eids = cellfun(@(txt) txt(end-35:end) , ses.url,'UniformOutput', false);
end

function url = append2url(url, keyname, cel)
if isempty(cel), return, end
cel = cellfun(@(x) [',' x], cel, 'UniformOutput', false);
str_req = cat(2,cel{:});
str_req = str_req(2: end);
url = [url '&' keyname '=' str_req];

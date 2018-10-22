function [eids,ses] = search(self,varargin)
% [eids,ses] = one.search('keyword', values)
% [eids,ses] = one.search('data', {'channels.brainLocation'})
% [eids,ses] = one.search('subjects', 'LEW008')
% [eids,ses] = one.search(..., 'users', {'miles', 'morgan'}) % sessions with 2 users miles and morgan
% [eids,ses] = one.search(..., 'date-range', datenum([2018 8 28 ; 2018 8 31]) )
% [eids,ses] = one.search(..., 'lab', 'mainenlab' )
% [eids,ses] = one.search(..., 'number', 2 )

% example of rest query: /sessions?&date_range=2018-08-24,2018-08-24&users=olivier
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
    'subject', 'subject';...
    'subjects', 'subject';...
    'date_range', 'date_range';...
    'date-range', 'date_range';...
    'lab', 'lab';...
    'labs', 'lab';...
    'number', 'number';...
    'numbers', 'number';...
    };
% substitute eventual typo with the proper parameter name
for  ia = 1:2:length(varargin)
    it = find(strcmpi(varargin{ia}, SEARCH_TERMS(:,1)),1);
    try
    assert(~isempty(it));
    catch
       error(['Incorrect input parameter name: ''' varargin{ia} '''']) 
    end
    varargin(ia) = SEARCH_TERMS(it,2);
end

%% Handle parameters
p = inputParser;
% automatically create the paramters according to the search terms (date
for search_param=setxor(unique(SEARCH_TERMS(:,2)),'date_range')'
    addParameter(p,search_param{1}, {})
end
addParameter(p,'date_range', [], @(x) (isnumeric(x) & any(numel(x)==[0 2]) | ischar(x)))
addParameter(p,'details', false)
parse(p,varargin{:});
for fn = fieldnames(p.Results)', eval([fn{1} '= p.Results.' (fn{1}) ';']); end
%%
if nargin ==1, eids = unique(SEARCH_TERMS(:,2)); ses=[]; return, end
% make sure the date is in a proper format
if ~isempty(date_range) && isa(date_range,'double')
    date_range = mat2cell(datestr(date_range, 'yyyy-mm-dd'),[1 1],10);
elseif ~isempty(date_range) && ischar(date_range)
    date_range = mat2cell(date_range,[1 1],10);
end
% create the url to send to the REST API
url = '/sessions?';
% append each filter term to the URL
for search_param = unique(SEARCH_TERMS(:,2))'
    url = append2url(url, search_param{1}, eval(search_param{1}));
end
ses = self.alyx_client.get(url);
if isempty(ses), [ses, eids] = deal([]); return, end
% url = arrayfun(@(x) x.url(end-35:end), ses, 'UniformOutput', false) % si pas de flatten
if ischar(ses.url)
    eids = ses.url((end-35:end));
else
    eids = cellfun(@(txt) txt(end-35:end) , ses.url,'UniformOutput', false);
end

function url = append2url(url, keyname, cel)
switch true
    case ischar(cel), cel = {cel};
    case isnumeric(cel), cel = num2str(cel);
end
if isempty(cel), return, end
cel = cellfun(@(x) [',' x], cel, 'UniformOutput', false);
str_req = cat(2,cel{:});
str_req = str_req(2: end);
url = [url '&' keyname '=' str_req];

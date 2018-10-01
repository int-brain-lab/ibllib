function varargout = list(self, eid, varargin)
% [dtypes details] = one.list(eid)
%       gets dataset types belonging to a session. eid can be a string of a
%       cell array of strings.
% [dtypes details] = one.list('cf264653-2deb-44cb-aa84-89b82507028a')
%       dtypes is a cell array containing dataset types belonging to the current session
%       details is a table containing a record per dataset belonging to the
%       session with more attributes
% session_info = one.list(eid,'keyword','all')
%       returns all metadata about one or several sessions as a strucure
% attrlist = one.list([],'keyword', 'data')
%        Lists the range of possible values for the keyword specified.
%        Keyword can be any of {'labs', 'datasets', 'users', subjects')


%% handle input arguments
KEY_NAMES = get_typo_list(eid);
% substitute eventual typo with the proper parameter name
for  ia = 1:2:length(varargin)
    it = find(strcmpi(varargin{ia}, KEY_NAMES(:,1)),1);
    if isempty(it), continue; end
    varargin(ia) = KEY_NAMES(it,2);
end
% parse input arguments
p = inputParser;
addOptional(p,'keyword', 'dataset-type');
parse(p,varargin{:});
for fn = fieldnames(p.Results)', eval([fn{1} '= p.Results.' (fn{1}) ';']); end
% make sure a typo in the keyword won't blow up
try
    ik = find(strcmpi(KEY_NAMES(:,1), keyword));
    keyword = KEY_NAMES{ik,2};
catch
    lc = cellfun(@(x) [newline ' ' x], unique(KEY_NAMES(:,2)), 'UniformOutput', false);
    error(['Keyword: ' keyword ' is not a field of the session table. Allowed fields are: '...
        newline  lc{:}])
end


%% redirect to nested functions depending on input configuration
switch true
    case iscell(eid), recursive_call; return
    case isempty(eid), list_rest_endpoint; return
    otherwise, list_session_field;
end

%% List REST table functionality

%% List session field functionality
% list of possible keywords
%     {'all'         }
%     {'dataset-type'}
%     {'end_time'    }
%     {'lab'         }
%     {'start_time'  }
%     {'subject'     }
%     {'type'        }
%     {'users'       }
% If no keyword sepecified just output a list of datasets for the session queried


    function recursive_call()
        for m = 1:length(eid)
            [tmp{1:2}] = self.list( eid{m}, varargin{:});
            if m ==1, varargout = tmp; continue, end
            % for next iterations, need to concatenate stuff
            for n = 1:length(tmp)
                switch true
                    case iscell(tmp{n})
                        varargout{n} = unique(cat(1, varargout{n}, tmp{n}));
                    case isstruct(tmp{n})
                        varargout{n} = cat_struct( varargout{n}, tmp{n});
                    case ischar(tmp{n})
                        varargout{n} = unique(cat(1, varargout{n}, tmp(n)));
                end
            end
        end
    end

    function list_rest_endpoint()
        details = self.alyx_client.get(['/' keyword] );
        % this switch links keynames to the specific field to list
        switch true
            case any(strcmp(keyword, {'labs', 'dataset-types'}))
                varargout{1} = details.name;
            case strcmp(keyword, 'users')
                varargout{1} = details.username;
            case strcmp(keyword, 'subjects')
                varargout{1} = details.nickname;
        end
        varargout{2} = details;
    end

    function list_session_field
        session_info = self.alyx_client.get_session(eid);
        switch true
            case strcmp(keyword, 'dataset-type')
                session_data_info = session_info.data_dataset_session_related;
                session_data_info.eid = repmat({eid}, size(session_data_info.dataset_type,1),1);
                dataset_list = unique(session_data_info.dataset_type);
                varargout = {dataset_list, session_data_info};
            case strcmp(keyword, 'all')
                varargout = {session_info, []};
            otherwise
                varargout = { session_info.(keyword), []};
        end
    end

end

function tlist = get_typo_list(eid)
if ~isempty(eid)
    tlist = cellstr({...
        'subjects',  'subject';...
        'subject',  'subject';...
        'user',  'users';...
        'users',  'users';...
        'lab',  'lab';...
        'labs',  'lab';...
        'type',  'type';...
        'start_time',  'start_time';...
        'start-time', 'start_time';...
        'end_time',  'end_time';...
        'end-time', 'end_time';...
        'all',  'all';...
        'data',  'dataset-type';...
        'dataset',  'dataset-type';...
        'datasets',  'dataset-type';...
        'dataset-types', 'dataset-type';...
        'dataset_types',  'dataset-type';...
        'dataset-type', 'dataset-type';...
        'dataset_type',  'dataset-type';...
        'dtypes',  'dataset-type';...
        'dtype',  'dataset-type';...
        });
else % list for the REST table endpoints
    tlist = cellstr({...
        'data', 'dataset-types';...
        'dataset', 'dataset-types';...
        'datasets', 'dataset-types';...
        'dataset-types', 'dataset-types';...
        'dataset_types', 'dataset-types';...
        'dataset-type', 'dataset-types';...
        'dataset_type', 'dataset-types';...
        'dtypes', 'dataset-types';...
        'dtype', 'dataset-types';...
        'users', 'users';...
        'user', 'users';...
        'subject', 'subjects';...
        'subjects', 'subjects';...
        'lab',  'labs';...
        'labs',  'labs'...
        });
end
end
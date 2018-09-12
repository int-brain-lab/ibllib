function varargout = load(eid, varargin)


% handles input arguments
p = inputParser;
addRequired(p, 'eid');
addOptional(p,'dry_run', false);
parse(p,eid,varargin{:});
for fn = fieldnames(p.Results)', eval([fn{1} '= p.Results.' (fn{1}) ';']); end


getenv('HOME')

end


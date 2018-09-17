function in = flatten(in, varargin)
% out = flatten(in);
% out = flatten(in, 'wrap_scalar', [false]);
% If numeric, returns a 1D vector
% If cell array, returns a 1D cell array
% if struct array, un-nest the struct-array and returns a struct
% if wrap_scalar is set to false, a scalar structure will not be serialized
% and wrapped into strings and will result as a different type as a vector

wrap_scalar = true;
p = inputParser;
addOptional(p,'wrap_scalar', false);
parse(p,varargin{:});
for fn = fieldnames(p.Results)', eval([fn{1} '= p.Results.' (fn{1}) ';']); end

switch true
    case isnumeric(in) | iscell(in) | ischar(in)
        in = in(:);
    case isstruct(in) & length(in)==1
        ff = fieldnames(in);
        for m = 1: length(ff)
            if isstruct(in.(ff{m})), in.(ff{m}) = flatten(in.(ff{m})); end
        end
        if wrap_scalar, in = flatten_struct(in); end
    case isstruct(in) & length(in) ~= 1
        in = flatten_struct(in);
end


function tmp = flatten_struct(in)
% flatten, ie. unnests a structure array
ff = fieldnames(in);
in = struct2cell(in);
tmp = struct();
% loop over the structure fields
for m = 1: length(ff)
    switch true
        % only scalar numeric data, convert to array taking care of
        % possible empty values
        case all(cellfun(@(x) isnumeric(x) && length(x)<=1, in(m,:)'))
            if any(cellfun(@isempty, in(m,:)' ))
                tmp.(ff{m}) = cellfun(@emptyCheck, in(m,:)');
            else
                tmp.(ff{m}) = cell2mat(in(m,:)');
            end
            % case with nested scalar cells need to un-nest them
        case all(cellfun(@(x) isscalar(x) & iscell(x), in(m,:)'))
            tmp.(ff{m}) = cellfun(@(x) x{1}, (in(m,:)'), 'UniformOutput', false);
            % if it's another nested structure, recursive vall and output a
            % nested array, which is probably the only smart use of nested arrays !
        case all(cellfun(@(x) isstruct(x), in(m,:)'))
            nnff = cellfun(@(x) flatten(x), in(m,:), 'UniformOutput', false);
            tmp.(ff{m}) = cat( 1, nnff{:});
        otherwise
            tmp.(ff{m}) = in(m,:)';
    end
end


function x = emptyCheck(x)
if isempty(x), x = NaN; end
return

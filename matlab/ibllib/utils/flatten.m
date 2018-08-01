function in = flatten(in)
% If numeric, returns a 1D vector
% If cell array, returns a 1D cell array
% if struct array, un-nest the struct-array and returns a struct

switch true
    case isnumeric(in) | iscell(in)
        in = in(:);
        return
    case isstruct(in)
        if length(in)==1, return, end
        in = flatten_struct(in);
    otherwise
        return
end


function tmp = flatten_struct(in)
ff = fieldnames(in);
in = struct2cell(in);
tmp = struct();
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
        otherwise,
            tmp.(ff{m}) = in(m,:)';
    end
end


function x = emptyCheck(x)
if isempty(x), x = NaN; end
return


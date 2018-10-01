function s = cat_struct(s1, s2, op)
% s = cat_struct(s1, s2, op)
% op: 'inter'
% todo: op: 'union', 'setA', 'setB' and tests
op = 'inter';
if ~strcmpi(op, 'inter')
    warning('Operator not implemented yet')
end

fn1 = fieldnames(s1);
fn2 = fieldnames(s2);

%% inter operation
selfields = intersect(fn1, fn2);

s1_consistent_length = length(unique(structfun(@length , s1)))<=1;
s2_consistent_length = length(unique(structfun(@length , s2)))<=1;

if ~s1_consistent_length || ~s2_consistent_length
    s1  = flatten(s1, 'wrap_scalar', true);
    s2  = flatten(s2, 'wrap_scalar', true);
end

for ff = selfields'
%     assert(strcmp(class(s1.(ff{1})), class(s2.(ff{1}))))
%     switch class(s1.(ff{1}))
%         case 'struct'
%             s.(ff{1}) = cat(1, flatten(s1.(ff{1})), flatten(s2.(ff{1})) );
%         case 'char'
%             s.(ff{1}) = cat(1, {s1.(ff{1})}, {s2.(ff{1})} );
%         otherwise
            s.(ff{1}) = cat(1, s1.(ff{1}), s2.(ff{1}) );
    end
end

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
for ff = selfields'
   s.(ff{1}) = cat(1, flatten(s1.(ff{1})), flatten(s2.(ff{1})) ); 
end


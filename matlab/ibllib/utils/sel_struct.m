function s = sel_struct(s,ind)
% s = sel_struct(s,ind)
% slice/sort all fields of a structure according to ind

s = structfun(@(x) x(ind), s, 'UniformOUtput', false);
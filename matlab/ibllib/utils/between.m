function bool = between(d, bounds, le_ge)
% bool = between(d, [0 100]): le ge
% bool = between(d, [0 100], [false false]): lt, gt

assert(numel(bounds)==2)
bounds = sort(bounds);
if nargin <=2, le_ge=[true true]; end

if le_ge(1)
    bool = (d >= bounds(1));
else
    bool = (d >  bounds(1));
end

if le_ge(2)
    bool = bool & (d <= bounds(2));
else
    bool = bool & (d <  bounds(2));
end
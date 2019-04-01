function [grp , mapper_sel, sub] = aggregate(value, mapper , func , mapper_sel)
% [grp , mapper_sel, sub] = aggregate(value, mapper , func , mapper_sel)
% Pivot table
% Example : given a list of individual students with the following attributes :    age  and grade
% count the number of student per age :
%   [Age.count , Age.age] = aggregate(student.grade , student.age , @length );
% get the average grade per age :
%   [Age.AverageGrade] = aggregate(student.grade , student.age , @mean );
% get the average grade for ages 15  and 17 :
%   [Age.AverageGrade] = aggregate(student.grade , student.age , @mean , [15 17]');

% if vectors, needs to be colomn vector only
if sum(size(value)<=1) , value = value(:); end
if sum(size(mapper)<=1), mapper = mapper(:); end
%%
ncols = size(value,2);
% create the subscripts defining the groups
if nargin <=3
    mapper_sel = unique(mapper(all(~isnan(mapper),2),:),'rows');
end
[~ , sub] = ismember(mapper,mapper_sel,'rows');
% for each column of the input, group_by
grp = zeros(size(mapper_sel,1), ncols);
for m = 1:ncols
    grp(:,m) = accumarray(sub(sub>0), value(sub>0,m),[],func);
end
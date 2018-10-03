function dt = jsonrest2serial(dt)
% dt = time.jsonrest2serial(json_cell_array)
% dt = time.jsonrest2serial({'2018-09-21T12:00:00'       ,'2018-10-03T14:16:13.929370'})

ind = cellfun(@(x) ~any(x=='.'), dt);
dt(ind) = cellfun(@(x) [x '.000000'], dt(ind), 'UniformOutput', false);
dt = datenum(dt, 'yyyy-mm-ddTHH:MM:SS.FFF');

end
function col = color_from_index(ind)
% col = color_from_index(ind)

color_order = [0 0.447 0.741;0.85 0.325 0.098;0.929 0.694 0.125;0.494 0.184 0.556;0.466 0.674 0.188;0.301 0.745 0.933;0.635 0.078 0.184];
col = color_order(mod(ind-1, size(color_order, 1)) + 1 , :);

end


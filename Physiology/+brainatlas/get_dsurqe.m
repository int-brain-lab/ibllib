function [V, h, cv, labels] = get_dsurqe



nii_file = '/datadisk/BrainAtlas/ATLASES/DSURQE_40micron/DSURQE_40micron_average.nii';
V.phy = io.read.nii(nii_file);
nii_file = '/datadisk/BrainAtlas/ATLASES/DSURQE_40micron/DSURQE_40micron_labels.nii';
[V.lab, H] = io.read.nii(nii_file);


%https://scalablebrainatlas.incf.org/mouse/WHS12
res = H.PixelDimensions(1)/1e3;
assert(all( H.PixelDimensions(1:3) - H.PixelDimensions(1)< eps))


V.lab = flip( permute(V.lab, [3, 1, 2]), 1);
V.phy = flip( permute(V.phy, [3, 1, 2, 4]), 1);

% excludes from the label volume
labels = io.read.json(label_file);
labels = [ struct2cell(labels)  fields(labels)];
labels  = labels(2:end,:);
labs_exclude = {'optic tract', 'nucleus accumbens', 'trigeminal', 'trigeminal tract', 'optic nerve'};
[~, il] = intersect(labels(:,1), labs_exclude);
V.lab(ismember(V.lab, il)) = 0;


fv = isosurface(permute(V.lab~=0,[3, 2, 1]),0.5);
% in this case the volume is out in pixel unit, convert to SI
fv.vertices = fv.vertices.*res;
fv.faces= fv.faces;

cv = CartesianVolume(V.lab, res, mean(fv.vertices));
fv.vertices = bsxfun( @minus, fv.vertices, mean(fv.vertices));

h.fig_volume = figure('Color','w'); h.p = patch(fv); h.ax = gca;
set(h.ax, 'DataAspectRatio',[1 1 1], 'zdir', 'reverse')
xlabel(h.ax, 'x'), ylabel(h.ax, 'y'), zlabel(h.ax, 'z')
h.p.FaceColor = 'red';
h.p.EdgeColor = 'none';
h.p.FaceAlpha = 0.7;
view(69,42);
camlight;


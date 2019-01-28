function logo = logo_ibl(typ)
% logo = logo_ibl;
% logo = logo_ibl('square'); (DEFAULT)
% logo = logo_ibl('full');

if nargin <= 0
    typ = 'square';
end
chem = fileparts(mfilename('fullpath'));
switch typ
    case 'full', impath = [chem filesep 'LogoIBL.jpg'];
    otherwise, impath = [chem filesep 'LogoIBL_square.jpg'];
end

logo = imread(impath);

%735 344
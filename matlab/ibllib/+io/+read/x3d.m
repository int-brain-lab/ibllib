function [P, F] = x3d(x3d_file)
% [P, F] = io.read.x3d(filename)
% P: point coordinates, F: indexed face sets
 
fid = fopen(x3d_file); S = fread(fid,Inf,'*char')'; fclose(fid);

% this is very brittle, need some more examples to check variability of the format
strFace = '<IndexedFaceSet solid="false" colorPerVertex="false" normalPerVertex="false" coordIndex=';
strPoint = '<Coordinate DEF="VTKcoordinates0000" point="';

ipoint = strfind(S, strPoint);
iface = strfind(S, strFace);
iquote = strfind(S, '"' );

% read the faces information
start_face = iquote(find(between(iquote, iface + [1 length(strFace)]), 1, 'last'))+1;
end_face = iquote(find(iquote > start_face, 1, 'first')) - 1;
     
F = sscanf( S(start_face:end_face), '%i%i%i%i\n');
F = reshape(F, 4, length(F)/4)';

% then the points coordinates
start_point = iquote(find(between(iquote, ipoint + [1 length(strPoint)]), 1, 'last'))+1;
end_point = iquote(find(iquote > start_point, 1, 'first')) - 1;

P = sscanf( S(start_point:end_point), '%f%f%f,\n');
P = reshape(P,3, length(P)/3)';



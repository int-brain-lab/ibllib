function data = npy(filename)
% Function to read NPY files into matlab. 
% *** Only reads a subset of all possible NPY files, specifically N-D arrays of certain data types. 
% See https://github.com/kwikteam/npy-matlab/blob/master/npy.ipynb for
% more. 
%

%[shape, dataType, fortranOrder, littleEndian, totalHeaderLength, ~] = io.read.npy_header(filename);
FH = io.read.npy_header(filename);

if FH.littleEndian
    fid = fopen(filename, 'r', 'l');
else
    fid = fopen(filename, 'r', 'b');
end

try
    
    [~] = fread(fid, FH.totalHeaderLength, 'uint8');

    % read the data
    data = fread(fid, prod(FH.arrayShape), [FH.dataType '=>' FH.dataType]);
    
    if length(FH.arrayShape)>1 && ~FH.fortranOrder
        data = reshape(data, FH.shape(end:-1:1));
        data = permute(data, [length(FH.arrayShape):-1:1] );
    elseif length(FH.arrayShape)>1
        data = reshape(data, FH.arrayShape);
    end
    
    fclose(fid);
    
catch me
    fclose(fid);
    rethrow(me);
end

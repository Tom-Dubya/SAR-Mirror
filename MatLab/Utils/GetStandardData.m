% If the provided path is to a .hdf5 file, assumes the data was saved along
% the lines of Armadillo's default save configuration. That is, the top
% level group is named "dataset."
% If the provided path is to a .mat file, assumes the data was generated
% using ConvertHDF5ToMat. That is, its organization is the same as
% Armadillo's default save configuration.
function data = GetStandardData(path)
    arguments
        path (1,1) string
    end

    if lower(path).endsWith(".hdf5")
        data = abs(h5read(path, "/dataset"));
    else
        data = load(path).output.dataset;
        if isfield(data, "real")
            data = data.real + data.imag * 1j;
        else
            data = abs(data);
        end
    end
end
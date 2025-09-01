% Integratable with IteratePath.
% Saves an HDF5 file as a MAT file, preserving the root name.
% Notes:
% - Ignores files without a .hdf5 extension.
function ConvertHDF5ToMat(dataPath)
    arguments
        dataPath (1,1) string {mustBeFile}
    end

    if ~lower(dataPath).endsWith(".hdf5")
        return
    end

    datasets = h5info(dataPath).Datasets;
    output = struct();
    for i = 1 : size(datasets, 1)
        datasetName = datasets(i).Name;
        data = h5read(dataPath, "/" + datasetName);
        output.(datasetName) = data;
    end

    [~, root, ~] = fileparts(dataPath);
    save(strcat(root, ".mat"), "output", "-v7.3");
end
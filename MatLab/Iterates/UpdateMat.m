% Integratable with IteratePath.
% Updates a .mat file to be saved in v7.3 (hdf5).
function UpdateMat(dataPath)
    arguments
        dataPath (1,1) string {mustBeFile}
    end

    data = load(dataPath).data;
    fields = fieldnames(data);
    save(dataPath, "dataPath", "-v7.3");
    for i = 1:length(fields)
        name = fields{i};
        value = data.(name);
        eval([name, " = value;"]);
        save(dataPath, name, "-append");
    end
end
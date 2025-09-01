% Assumes that the given path is within a /data/ master directory.
% Returns an output path that mirrors the supplied path's relative
% structure. I.e. /data/images/vehicles/... -> /output/images/vehicles/...
function outputDir = GetOutputPath(path, rootSeparatorName, masterOutputName, includeExtension)
    arguments
        path (1,1) string
        rootSeparatorName {mustBeNonempty} = "Data"
        masterOutputName {mustBeNonempty} = "output"
        includeExtension {mustBeNonempty} = false
    end

    outputDir = masterOutputName + filesep;
    [inputPath, inputName, inputExtension] = fileparts(path);
    inputPathComponents = split(inputPath, filesep);
    dataDepth = find(inputPathComponents == rootSeparatorName);
    if dataDepth < numel(inputPathComponents)
        outputDir = strcat(outputDir, strjoin(inputPathComponents(dataDepth + 1:end), filesep));
    end

    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    if includeExtension
        inputName = strcat(inputName, inputExtension);
    end

    outputDir = strcat(outputDir, filesep, inputName);
end
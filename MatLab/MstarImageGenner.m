function CreateSARImage(imageData, outPath)
    arguments
        imageData
        outPath (1,1) string
    end

    if ~lower(outPath).endsWith(".jpg")
        outPath = strcat(outPath, ".jpg");
    end

    lowDb = -70;
    highDb = 0;
    normalisedData = squeeze(real(20 * log10(imageData ./ max(imageData, [], "all"))));
    minNum = min(normalisedData, [], "all");
    maxNum = max(normalisedData, [], "all");
    scaled = mat2gray(normalisedData, [lowDb, highDb]);
    imwrite(scaled, outPath);

    % Coloured image generation
    cmap = parula(256);
    idx = round(scaled * 255);
    % imwrite(ind2rgb(idx, cmap), outPath);
end

function MstarImageGen(dataPath)
    arguments
        dataPath (1,1) string
    end

    paths = dir(fullfile(dataPath, '**', '*.*'));
    paths = paths(~[paths.isdir]);
    count = numel(paths);
    outputDir = "output" + filesep;
    for pathIndex = 1 : count
        currentPath = fullfile(paths(pathIndex).folder, paths(pathIndex).name);
        if isempty(currentPath)
            continue;
        end
        disp("[DEBUG] Processing: " + currentPath + " (" + pathIndex + " of " + count + ")");

        cOutputDir = outputDir;
        [inputPath, inputName, ~] = fileparts(currentPath);
        inputPathComponents = split(inputPath, filesep);
        dataDepth = find(inputPathComponents == "Data");
        if dataDepth < numel(inputPathComponents)
            cOutputDir = strcat(cOutputDir, strjoin(inputPathComponents(dataDepth + 1:end), filesep));
        end
        mkdir(cOutputDir);

        cOutputPath = strcat(cOutputDir, filesep, inputName);
        load(currentPath);
        allImages = output.dataset.real + output.dataset.imag * 1j;
        for i = 1 : size(allImages, 1)
            CreateSARImage(allImages(i, :, :), cOutputPath + "_" + i);
        end
    end
    disp("[DEBUG] Finished processing all valid data files.");
end
% Integratable with IteratePath.
% A few assumptions are made:
% 1) HDF5 file have images stored in the "dataset" dataset
% 2) MAT files have images stored in the "dataset" dataset, under the 
% "output" grouping
function CreateSARImage(dataPath, low, high, dbConversion, normalise)
    arguments
        dataPath (1,1) string {mustBeFile}
        low (1,1) double
        high (1,1) double
        dbConversion (1,1) bool
        normalise (1,1) bool
    end

    addpath("../Utils");

    if lower(dataPath).endsWith(".hdf5")
        imageData = abs(h5read(dataPath, "/dataset"));
    else
        imageData = load(dataPath).output.dataset;
        if isfield(imageData, "real")
            imageData = imageData.real + imageData.imag * 1j;
        else
            imageData = abs(imageData);
        end
    end


    if dbConversion
        imageData = 20 * log10(imageData ./ max(imageData, [], "all"));
    end
    normalisedData = real(imageData);
    nonInfNormalisedData = normalisedData(~isinf(normalisedData));
    minNum = real(min(nonInfNormalisedData, [], "all"));
    normalisedData(isinf(normalisedData)) = minNum;
    if normalise
        lowDbAdjustment = mean(normalisedData(:), 'omitnan') - std(normalisedData(:), 'omitnan');
        low = low + lowDbAdjustment;
    end
    outPath = strcat(GetOutputPath(dataPath), ".jpg");
    imwrite(mat2gray(normalisedData, [low high]), outPath);
end
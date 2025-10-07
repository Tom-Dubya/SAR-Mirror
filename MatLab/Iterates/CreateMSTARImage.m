% Integratable with IteratePath.
% IMPORTANT: Assumes low and high values < -10000 and > 10000 correspond to
% the minimum and maximum values found in the data respectively.
function CreateMSTARImage(dataPath, low, high, dbConversion, normalise)
    arguments
        dataPath (1,1) string {mustBeFile}
        low (1,1) double
        high (1,1) double
        dbConversion (1,1) logical
        normalise (1,1) logical
    end

    addpath("Utils");

    outDirectory = GetOutputPath(dataPath);
    allData = GetStandardData(dataPath);
    for i = 1 : size(allData, 1)
        imageData = squeeze(allData(i, :, :));
        if dbConversion
            imageData = 20 * log10(imageData ./ max(imageData, [], "all"));
        end
        normalisedData = real(imageData);
        nonInfNormalisedData = normalisedData(~isinf(normalisedData));
        minNum = real(min(nonInfNormalisedData, [], "all"));
        maxNum = real(max(nonInfNormalisedData, [], "all"));
        normalisedData(isinf(normalisedData)) = minNum;
        if normalise
            lowDbAdjustment = mean(normalisedData(:), 'omitnan') - std(normalisedData(:), 'omitnan');
            low = low + lowDbAdjustment;
        end

        if low < -10000
            low = minNum;
        end

        if high > 10000
            high = maxNum;
        end

        outPath = strcat(outDirectory, "_" + i + ".jpg");
        imwrite(mat2gray(normalisedData, [low high]), outPath);
    end
end
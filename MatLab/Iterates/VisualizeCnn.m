% Integratable with IteratePath.
function VisualizeCnn(dataPath, cnnNet, labelNames, utilisedLabel, imageSize, colorMap, alpha)
    arguments
        dataPath (1,1) string {mustBeFile}
        cnnNet (1,1) dlnetwork
        labelNames
        utilisedLabel string {mustBeNonempty} = ""
        imageSize {mustBeNonempty} = [128, 128]
        colorMap {mustBeNonempty} = jet(256)
        alpha double {mustBeNonempty} = 0.5
    end

    if ~lower(dataPath).endsWith(".jpg")
        return
    end

    originalImage = imread(dataPath);
    scaledImage = imresize(originalImage, imageSize);
    scores = predict(cnnNet, single(scaledImage));
    predictedLabel = scores2label(scores, labelNames);
    if utilisedLabel == ""
        utilisedLabel = predictedLabel;
    end

    channel = find(string(utilisedLabel) == labelNames);
    outputPath = GetOutputPath(dataPath);
    for i = 1 : numel(cnnNet.Layers)
        layer = cnnNet.Layers(i);

        scoreMap = GetGradCamDynamic(cnnNet, single(scaledImage), channel, layer.Name);
        normalisedScoreMap = mat2gray(extractdata(scoreMap));
        normalisedScoreMap = imresize(normalisedScoreMap, [128, 128]);
        heatMapRGB = ind2rgb(gray2ind(normalisedScoreMap, 256), colorMap);
        overlayedImage = SimpleOverlay(im2double(scaledImage), heatMapRGB, alpha);
        [overlayedImage, overlayedImageColorMap] = rgb2ind(overlayedImage, 256);

        if i == 1
            imwrite(overlayedImage, overlayedImageColorMap, outputPath + '.gif', 'gif', 'LoopCount', Inf, 'DelayTime', 0.5);
        else
            imwrite(overlayedImage, overlayedImageColorMap, outputPath + '.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 0.5);
        end
    end
end
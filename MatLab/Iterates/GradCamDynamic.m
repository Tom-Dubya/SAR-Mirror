% Integratable with IteratePath.
function GradCamDynamic(dataPath, cnnNet, labelNames, cnnLayer, utilisedLabel, imageSize, colorMap, alpha)
    arguments
        dataPath (1,1) string {mustBeFile}
        cnnNet (1,1) dlnetwork
        labelNames
        cnnLayer string
        utilisedLabel string {mustBeNonempty} = ""
        imageSize {mustBeNonempty} = [128, 128]
        colorMap {mustBeNonempty} = jet(256)
        alpha single {mustBeNonempty} = 0.5
    end

    if ~(lower(dataPath).endsWith(".jpg") || lower(dataPath).endsWith(".png"))
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
    scoreMap = GetGradCamDynamic(cnnNet, single(scaledImage), channel, cnnLayer);
    normalisedScoreMap = mat2gray(extractdata(scoreMap));
    normalisedScoreMap = imresize(normalisedScoreMap, [128, 128]);
    heatMapRGB = ind2rgb(gray2ind(normalisedScoreMap, 256), colorMap);
    overlayedImage = SimpleOverlay(im2double(scaledImage), heatMapRGB, alpha);
    imwrite(overlayedImage, GetOutputPath(dataPath) + ".png", "png");
end
% Integratable with IteratePath.
% Depending on the CNN, this isn't the GradCam you'll need. 
% See GradCamDynamic.
function GradCam(dataPath, cnnNet, labelNames, utilisedLabel, imageSize, colorMap, alpha)
    arguments
        dataPath (1,1) string {mustBeFile}
        cnnNet (1,1) dlnetwork
        labelNames
        utilisedLabel string {mustBeNonempty} = ""
        imageSize {mustBeNonempty} = [128, 128]
        colorMap {mustBeNonempty} = jet(256)
        alpha single {mustBeNonempty} = 0.5
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
    scoreMap = gradCAM(cnnNet, scaledImage, channel);
    normalisedScoreMap = mat2gray(scoreMap);
    heatMapRGB = ind2rgb(gray2ind(normalisedScoreMap, 256), colorMap);
    overlayedImage = SimpleOverlay(im2double(scaledImage), heatMapRGB, alpha);
    imwrite(overlayedImage, GetOutputPath(dataPath) + ".png", "png");
end
% Integratable with IteratePath.
function VisualizeLayerFeatures(dataPath, cnnNet, cnnLayerName, imageSize, colorMap, alpha)
    arguments
        dataPath (1,1) string {mustBeFile}
        cnnNet (1,1) dlnetwork
        cnnLayerName (1,1) string
        imageSize {mustBeNonempty} = [128, 128]
        colorMap {mustBeNonempty} = jet(256)
        alpha double {mustBeNonempty} = 0.5
    end

    if ~lower(dataPath).endsWith(".jpg")
        return
    end

    originalImage = imread(dataPath);
    scaledImage = imresize(originalImage, imageSize);
    dlInput = dlarray(single(scaledImage), 'SSC');      

    act = forward(cnnNet, dlInput, 'Outputs', cnnLayerName);
    outputPath = GetOutputPath(dataPath);
    for i = 1 : size(act, 3)
        actClean = squeeze(real(act(:, :, i)));
        actClean(isnan(actClean)) = 0;
        actClean(isinf(actClean)) = 0;
        heatMap = mat2gray(extractdata(actClean));
        heatMapUpscaled = imresize(heatMap, imageSize);
        heatMapRGB = ind2rgb(gray2ind(heatMapUpscaled, 256), colorMap);
        overlayedImage = SimpleOverlay(im2double(scaledImage), heatMapRGB, alpha);
        [overlayedImage, overlayedImageColorMap] = rgb2ind(overlayedImage, 256);

        if i == 1
            imwrite(overlayedImage, overlayedImageColorMap, outputPath + '.gif', 'gif', 'LoopCount', Inf, 'DelayTime', 0.5);
        else
            imwrite(overlayedImage, overlayedImageColorMap, outputPath + '.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 0.5);
        end
    end
end
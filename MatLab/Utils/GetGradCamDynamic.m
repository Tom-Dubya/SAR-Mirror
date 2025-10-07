function heatMap = GetGradCamDynamic(cnnNet, inputImage, targetClass, convLayerName)
    arguments
        cnnNet (1,1) dlnetwork
        inputImage
        targetClass
        convLayerName string
    end

    function [convActivations, gradients] = modelGradients(net, dlImg, targetClass, convLayerName)
        % Since GradCamDynamic should be used in instances where the
        % softmax layer has failed us, we simply find the most immediate
        % layer before the softmax in order to use as a reference.
        numLayers = size(net.Layers, 1);
        layerI = 0;
        finalLayer = net.Layers(numLayers - layerI);
        while contains(finalLayer.Name, 'softmax')
            layerI = layerI + 1;
            finalLayer = net.Layers(numLayers - layerI);
        end

        [convActivations, logits] = forward(net, dlImg, 'Outputs', {char(convLayerName), finalLayer.Name});
        score = logits(targetClass);
        gradients = dlgradient(score, convActivations);
    end
    
    inputImage = dlarray(inputImage, 'SSC');
    [convActivations, gradients] = dlfeval(@modelGradients, cnnNet, inputImage, targetClass, convLayerName);
    
    % Gradients is in Height x Width x Channel, where each channel
    % represents a 'feature map' that the layer produces. We take the
    % average of every feature for a layer based heat-map. We can examine
    % each feature by itself using VisualizeCnnLayer.
    alpha = mean(mean(gradients, 1), 2);  % 1×1×Channels
    
    % Weighted combination of feature maps
    weightedActivations = sum(convActivations .* alpha, 3);
    
    % ReLU to keep only positive influences and see what guides the CNN to
    % reach its conclusions.
    heatMap = max(weightedActivations, 0);
    
    % Normalize for visualisation
    heatMap = gather(heatMap);
    heatMap = heatMap - min(heatMap(:));
    heatMap = heatMap / (max(heatMap(:)) + 1e-10);
end
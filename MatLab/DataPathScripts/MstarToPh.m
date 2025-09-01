% To-Do: Clean Up
function MstarToPh(dataPathsFile)
    arguments
        dataPathsFile (1,1) string {mustBeFile}
    end
    IterateDataPaths(dataPathsFile, @ConvertToPH)
end

function data = ConvertToPH(filePath)
    c = 299792458;
    load(filePath);

    sampleCount = size(output.azim, 1);
    crossNumPixelsImage = output.xSamples;
    numPixelsImage = output.ySamples;
    if crossNumPixelsImage ~= numPixelsImage
        crossNumPixelsImage = min(crossNumPixelsImage, numPixelsImage);
        numPixelsImage = min(crossNumPixelsImage, numPixelsImage);
    end
    cropping = 20;
    crossNumPixelsCrop = crossNumPixelsImage - cropping;
    numPixelsCrop = numPixelsImage - cropping;

    data.numPulses = sampleCount;
    data.numXSamples = zeros(sampleCount, 1);
    data.numYSamples = zeros(sampleCount, 1);
    data.centreX = zeros(sampleCount, 1);
    data.centreY = zeros(sampleCount, 1);
    data.sceneWidth = zeros(sampleCount, 1);
    data.sceneHeight = zeros(sampleCount, 1);
    data.minAzim = zeros(sampleCount, 1);
    data.maxAzim = zeros(sampleCount, 1);
    data.deltaF = zeros(sampleCount, 1);
    data.minF = zeros(sampleCount, numPixelsCrop);
    data.maxF = zeros(sampleCount, numPixelsCrop);
    data.x_vec = zeros(sampleCount, crossNumPixelsImage);
    data.y_vec = zeros(sampleCount, numPixelsImage);
    data.x_mat = zeros(sampleCount, numPixelsImage, crossNumPixelsImage);
    data.y_mat = zeros(sampleCount, numPixelsImage, crossNumPixelsImage);
    data.z_mat = zeros(sampleCount, numPixelsImage, crossNumPixelsImage);
    data.AntX = output.antennaX;
    data.AntY = output.antennaY;
    data.AntZ = output.antennaZ;
    data.AntAzim = zeros(sampleCount, numPixelsCrop);
    data.AntElev = zeros(sampleCount, numPixelsCrop);
    data.phdata = zeros(sampleCount, crossNumPixelsCrop, numPixelsCrop);
     
    for i = 1 : sampleCount
        % Fetching necessary info from metadata
        azi = output.azim(i);
        complex_image = output.magnitude(i, :) .* exp(1j * output.phase(i, :));
        if numel(complex_image) ~= crossNumPixelsImage * numPixelsImage
        end

        complex_image = reshape(complex_image, output.xSamples, output.ySamples);
        depression = output.depression(i);
        centreFrequency = output.centreFrequency(i) * 1e9;
        bandwidth = output.bandwidth(i) * 1e9;

        % These metadata points aren't consistently present at all in MSTAR. 
        crossRangePixelSpacing = output.crossRangePixelSpacing(i);
        rangePixelSpacing = output.rangePixelSpacing(i);
        if (crossRangePixelSpacing == 0)
            crossRangePixelSpacing = rangePixelSpacing;
        end
     
        sceneWidth = crossRangePixelSpacing * crossNumPixelsImage;
        sceneHeight = rangePixelSpacing * numPixelsImage;
                
        lowerFrequency = centreFrequency - bandwidth / 2;
        upperFrequency = centreFrequency + bandwidth / 2;

        % Creating Taylor window to remove later
        taylorWindow = kron(taylorwin(crossNumPixelsCrop, 4, -35), taylorwin(numPixelsCrop, 4, -35).');
        
        % Creating vector of frequencies and array of frequencies with appropriate
        % size
        f = linspace(lowerFrequency, upperFrequency, crossNumPixelsCrop).';
        fRep = repmat(f, 1, numPixelsCrop);
            
        % Creating necessary arrays   
        arr_img_comp = zeros(crossNumPixelsImage, numPixelsImage);
        arr_img_fft = zeros(crossNumPixelsImage, numPixelsImage);
        arr_img_fft_polar = zeros(crossNumPixelsCrop, numPixelsCrop);
        
        % Creating vector of azimuth angles
        thetas = linspace(180 - azi - 1.5, 180 - azi + 1.5, numPixelsCrop);
        thetaRep = repmat(thetas, crossNumPixelsCrop, 1);

        %Crop to fixed size numPixelsCrop while maintaining centering
        imageCentre = floor(size(complex_image) / 2);
        complex_image = complex_image(imageCentre(1) - floor(crossNumPixelsImage/2) + 1:imageCentre(1) + ceil(crossNumPixelsImage/2),...
            imageCentre(2) - floor(numPixelsImage / 2) + 1:imageCentre(2) + ceil(numPixelsImage / 2));
        imageCentre = floor(size(complex_image)/2);
        arr_img_comp(:,:) = complex_image;
        
        %form the polar meshgrid
        k_1 = (4 * pi / c * cosd(depression) * fRep .* sind(thetaRep));
        k_2 = (4 * pi / c * cosd(depression) * fRep .* cosd(thetaRep));
        yy = linspace(min(k_2(:)), max(k_2(:)), numPixelsCrop);
        xx = linspace(min(k_1(:)), max(k_1(:)), crossNumPixelsCrop);
        [XX,YY] = meshgrid(yy, xx);
       
        %Transform to phase history domain
        arr_img_fft(:,:) = fftshift(fft2(ifftshift(complex_image)));
        
        %Undo the taylor window and crop to numPixelsImage
        arr_img_fft_crop = (1 ./ taylorWindow .* arr_img_fft(imageCentre(1) - round(crossNumPixelsCrop / 2) + 1:imageCentre(1)...
            + crossNumPixelsCrop - round(crossNumPixelsCrop / 2), imageCentre(2) - round(numPixelsCrop / 2) + 1:imageCentre(2) ...
            + numPixelsCrop - round(numPixelsCrop / 2), 1));
        
        % Interpolate to polar grid
        arr_img_fft_polar(:,:) = interp2(XX, YY, arr_img_fft_crop, k_2, k_1, 'spline', 0);
        
        % Saving information for imaging algorithm
        data.numXSamples(i) = crossNumPixelsImage;
        data.numYSamples(i) = numPixelsImage;
        data.centreX(i) = imageCentre(1);
        data.centreY(i) = imageCentre(2);
        data.sceneWidth(i) = sceneWidth;
        data.sceneHeight(i) = sceneHeight;
        data.minAzim(i) = 180 - azi - 1.5;
        data.maxAzim(i) = 180 - azi + 1.5;
        data.deltaF(i) = (bandwidth / numPixelsCrop);  %freq spacing
        data.minF(i, :) = lowerFrequency * ones(numPixelsCrop, 1); 
        data.maxF(i, :) = upperFrequency * ones(numPixelsCrop, 1); 
        data.x_vec(i, :) = linspace(-sceneWidth / 2, sceneWidth / 2, crossNumPixelsImage);  %location of pixels in x and y directions
        data.y_vec(i, :) = linspace(-sceneHeight / 2, sceneHeight / 2, numPixelsImage); 
        [data.x_mat(i, :, :), data.y_mat(i, :, :)] = meshgrid(data.x_vec(i, :), data.y_vec(i, :));%creating grid with locations
        data.AntAzim(i, :) =  linspace(180 - azi - 1.5, 180 - azi + 1.5, numPixelsCrop); %saving azimuth angles
        data.AntElev(i, :) = depression * ones(numPixelsCrop, 1); %saving elevation info as vector
        data.phdata(i, :, :) = arr_img_fft_polar; %saving PH data
    end
    outputDir = "output" + filesep;
    mkdir(outputDir);
    [~, name, ~] = fileparts(filePath);
    save(outputDir + "PH_" + name, "data", "-v7.3");
end
% To-Do: Clean Up

function SampleToPh(dataPathsFile)
    arguments
        dataPathsFile (1,1) string {mustBeFile}
    end
    IterateDataPaths(dataPathsFile, @ConvertToPH)
end

function data = ConvertToPH(filePath)
    c = 299792458;
    load(filePath);

    crossNumPixelsImage = size(complex_img, 1);
    numPixelsImage = size(complex_img, 2);
    if crossNumPixelsImage ~= numPixelsImage
        crossNumPixelsImage = min(crossNumPixelsImage, numPixelsImage);
        numPixelsImage = min(crossNumPixelsImage, numPixelsImage);
    end
    cropping = 20;
    crossNumPixelsCrop = crossNumPixelsImage - cropping;
    numPixelsCrop = numPixelsImage - cropping;

    % Fetching necessary info from metadata
    azi = double(azimuth);
    complex_image = complex_img;
    depression = double(elevation);
    centreFrequency = double(center_freq);
    bandwidth = double(bandwidth);

    % These metadata points aren't consistently present at all in MSTAR. 
    crossRangePixelSpacing = xrange_pixel_spacing;
    rangePixelSpacing = range_pixel_spacing;
    if (crossRangePixelSpacing == 0)
        crossRangePixelSpacing = rangePixelSpacing;
    end
     
    sceneWidth = crossRangePixelSpacing * crossNumPixelsImage;
    sceneHeight = rangePixelSpacing * numPixelsImage;
                
    lowerFrequency = centreFrequency - bandwidth / 2;
    upperFrequency = centreFrequency + bandwidth / 2;

    % Creating Taylor window to remove later
    sll = taylor_weights;
    taylorWindow = kron(taylorwin(crossNumPixelsCrop, 4, sll), taylorwin(numPixelsCrop, 4, sll).');
        
    % Creating vector of frequencies and array of frequencies with appropriate size
    f = linspace(lowerFrequency, upperFrequency, crossNumPixelsCrop).';
    fRep = repmat(f, 1, numPixelsCrop);
            
    % Creating necessary arrays   
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
    data.numXSamples = crossNumPixelsImage;
    data.numYSamples = numPixelsImage;
    data.centreX = imageCentre(1);
    data.centreY = imageCentre(2);
    data.sceneWidth = sceneWidth;
    data.sceneHeight = sceneHeight;
    data.minAzim = 180 - azi - 1.5;
    data.maxAzim = 180 - azi + 1.5;
    data.deltaF = bandwidth / numPixelsCrop;
    data.minF = lowerFrequency;
    data.maxF = upperFrequency; 
    data.x_vec = linspace(-sceneWidth / 2, sceneWidth / 2, crossNumPixelsImage);  %location of pixels in x and y directions
    data.y_vec = linspace(-sceneHeight / 2, sceneHeight / 2, numPixelsImage); 
    [data.x_mat, data.y_mat] = meshgrid(data.x_vec(:), data.y_vec(:));%creating grid with locations
    data.z_mat = zeros(numPixelsImage, crossNumPixelsImage);
    data.AntAzim =  linspace(180 - azi - 1.5, 180 - azi + 1.5, numPixelsCrop).'; %saving azimuth angles
    data.AntElev = depression * ones(numPixelsCrop, 1); %saving elevation info as vector
    data.phdata = arr_img_fft_polar; %saving PH data

    outputDir = "output" + filesep;
    mkdir(outputDir);
    [~, name, ~] = fileparts(filePath);
    save(outputDir + "PH_" + name, "data", "-v7.3");
end
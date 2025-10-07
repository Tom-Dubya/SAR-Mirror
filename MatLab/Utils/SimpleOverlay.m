% Assumes that the given path is within a /data/ master directory.
% Returns an output path that mirrors the supplied path's relative
% structure. I.e. /data/images/vehicles/... -> /output/images/vehicles/...
function output = SimpleOverlay(xData, yData, xAlpha)
    arguments
        xData,
        yData,
        xAlpha (1,1) single {mustBeNonempty} = 0.5
    end
    output = (1 - xAlpha) * yData + xAlpha * xData;
end


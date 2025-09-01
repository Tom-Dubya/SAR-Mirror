#include "mstar_aggregator.h"

#include <regex>
#include <utility>
#include "../extern/mstar2raw.h"
#include "../utils/file_utils.h"

/*-------------------------------------------------------------------------
 * To-Do:
 * - Flesh out debug implementation
 * - Provide a way to easily generate a DataPaths file for MSTAR folders
 *------------------------------------------------------------------------*/

mstar_aggregator::mstar_aggregator(std::string mstarPath, const bool debug)
{
    this->mstarPath = std::move(mstarPath);
    this->debug = debug;
}

void mstar_aggregator::load(const bool fullLoad)
{
    if (debug)
    {
        std::cout << "[Debug] Loading -> " << mstarPath << std::endl;
    }

    const std::vector<std::string> paths = get_files_in_directory_with_validation(mstarPath, R"(.*\d{3})", R"(\.\D{3})");
    const unsigned long long dataCount = paths.size();

    if (debug)
    {
        std::cout << "[Debug] Encountered " << dataCount << " targets." << std::endl;
    }

    std::vector<arma::vec> allMagnitude;
    std::vector<arma::vec> allPhase;
    nRows = arma::vec(dataCount);
    nCols = arma::vec(dataCount);
    azim = arma::vec(dataCount);
    roll = arma::vec(dataCount);
    pitch = arma::vec(dataCount);
    yaw = arma::vec(dataCount);
    depression = arma::vec(dataCount);
    groundPlaneSquint = arma::vec(dataCount);
    slantPlaneSquint = arma::vec(dataCount);
    range = arma::vec(dataCount);

    targetX = arma::vec(dataCount);
    targetY = arma::vec(dataCount);
    targetZ = arma::vec(dataCount);
    antennaX = arma::vec(dataCount);
    antennaY = arma::vec(dataCount);
    antennaZ = arma::vec(dataCount);
    heading = arma::vec(dataCount);
    xVelocity = arma::vec(dataCount);

    slowTime = arma::vec(dataCount);
    rangeResolution = arma::vec(dataCount);
    crossRangeResolution = arma::vec(dataCount);
    rangePixelSpacing = arma::vec(dataCount);
    crossRangePixelSpacing = arma::vec(dataCount);

    centreFrequency = arma::vec(dataCount);
    bandwidth = arma::vec(dataCount);
    polarisationType = arma::vec(dataCount);

    auto verify_and_assign = [](std::map<std::string, std::string>& header, arma::mat& destination, const int index, const std::string& dataName)
    {
        if (header.find(dataName) != header.end())
        {
            destination.at(index) = std::stod(header[dataName]);
        }
    };

    // We want to ensure the number of rows and columns stay consistent.
    int highestNRows = -1, highestNCols = -1, lowestNRows = -1, lowestNCols = -1;
    for (int i = 0; i < dataCount; i++)
    {
        const std::string& path = paths[i];
        std::map<std::string, std::string> iHeader;
        arma::vec iMagnitude;
        arma::vec iPhase;
        decompress_mstar(path, iHeader, iMagnitude, iPhase);

        const int nRow = std::stoi(iHeader["NumberOfRows"]);
        if (highestNRows == -1)
        {
            highestNRows = nRow;
            lowestNRows = nRow;
        }

        const int nCol = std::stoi(iHeader["NumberOfColumns"]);
        if (highestNCols == -1)
        {
            highestNCols = nCol;
            lowestNCols = nCol;
        }

        if (nRow > highestNRows)
        {
            highestNRows = nRow;
        }

        if (nCol > highestNCols)
        {
            highestNCols = nCol;
        }

        if (nRow < lowestNRows)
        {
            lowestNRows = nRow;
        }

        if (nCol < lowestNCols)
        {
            lowestNCols = nCol;
        }

        allMagnitude.push_back(iMagnitude);
        allPhase.push_back(iPhase);

        /*-------------------------------------------------------------------------
         * The desired latitude and longitude are not used right now, but they could
         * be looked at for calculating a motion compensation point.
         *------------------------------------------------------------------------*/
        nRows.at(i) = nRow;
        nCols.at(i) = nCol;
        verify_and_assign(iHeader, azim, i, "TargetAz");
        verify_and_assign(iHeader, roll, i, "TargetRoll");
        verify_and_assign(iHeader, pitch, i, "TargetPitch");
        verify_and_assign(iHeader, yaw, i, "TargetYaw");
        verify_and_assign(iHeader, depression, i, "MeasuredDepression");
        verify_and_assign(iHeader, groundPlaneSquint, i, "MeasuredGroundPlaneSquint");
        verify_and_assign(iHeader, slantPlaneSquint, i, "MeasuredSlantPlaneSquint");
        verify_and_assign(iHeader, range, i, "MeasuredRange");
        verify_and_assign(iHeader, targetX, i, "MeasuredAimpointLatitude");
        verify_and_assign(iHeader, targetY, i, "MeasuredAimpointLongitude");
        verify_and_assign(iHeader, targetZ, i, "MeasuredAimpointElevation");
        verify_and_assign(iHeader, antennaX, i, "MeasuredAntennaLatitude");
        verify_and_assign(iHeader, antennaY, i, "MeasuredAntennaLongitude");
        verify_and_assign(iHeader, antennaZ, i, "MeasuredAircraftAltitude");
        verify_and_assign(iHeader, heading, i, "MeasuredAircraftHeading");
        verify_and_assign(iHeader, xVelocity, i, "X_Velocity");
        verify_and_assign(iHeader, slowTime, i, "CollectionTime");
        verify_and_assign(iHeader, rangeResolution, i, "RangeResolution");
        verify_and_assign(iHeader, crossRangeResolution, i, "CrossRangeResolution");
        verify_and_assign(iHeader, rangePixelSpacing, i, "RangePixelSpacing");
        verify_and_assign(iHeader, crossRangePixelSpacing, i, "CrossRangePixelSpacing");

        std::regex gHzPattern(" *GHz");
        if (iHeader.find("CenterFrequency") != iHeader.end())
        {
            std::string headerValue = iHeader["CenterFrequency"];
            centreFrequency.at(i) = std::stod(std::regex_replace(headerValue, gHzPattern, ""));
        }

        if (iHeader.find("Bandwidth") != iHeader.end())
        {
            std::string headerValue = iHeader["Bandwidth"];
            bandwidth.at(i) = std::stod(std::regex_replace(headerValue, gHzPattern, ""));
        }

        if (iHeader.find("Polarization") != iHeader.end())
        {
            std::string polarization = iHeader["Polarization"];
            int polarizationId = -1;
            if (polarization == "HH")
            {
                polarizationId = 1;
            }
            else if (polarization == "HV")
            {
                polarizationId = 2;
            }
            else if (polarization == "VH")
            {
                polarizationId = 3;
            }
            else if (polarization == "VV")
            {
                polarizationId = 4;
            }
            polarisationType.at(i) = polarizationId;
        }
    }

    numXSamples = lowestNCols;
    numYSamples = lowestNRows;
    const long long maxSamples = fullLoad ? highestNRows * highestNCols : lowestNRows * lowestNCols;
    magnitude = arma::mat(dataCount, maxSamples);
    phase = arma::mat(dataCount, maxSamples);

    const int range = lowestNRows * lowestNCols - 1;
    for (int i = 0; i < dataCount; i++)
    {
        if (fullLoad)
        {
            magnitude.row(i).subvec(0, allMagnitude[i].size() - 1) = allMagnitude[i].t();
            phase.row(i).subvec(0, allPhase[i].size() - 1) = allPhase[i].t();
        }
        else
        {
            magnitude.row(i) = allMagnitude[i].subvec(0, range).t();
            phase.row(i) = allPhase[i].subvec(0, range).t();
        }
    }
}

void mstar_aggregator::save(const std::string& rawSavePath)
{
    std::string parent, file, extension;
    get_file_info(rawSavePath, parent, file, extension);
    const std::string savePath = parent + "/" + file + ".hdf5";
    std::filesystem::create_directory(parent);

    arma::vec temp(1);
    temp[0] = numXSamples;
    temp.save(arma::hdf5_name(savePath, "xSamples"));
    temp[0] = numYSamples;
    temp.save(arma::hdf5_name(savePath, "ySamples", arma::hdf5_opts::append));

    nCols.save(arma::hdf5_name(savePath, "nCols", arma::hdf5_opts::append));
    nRows.save(arma::hdf5_name(savePath, "nRows", arma::hdf5_opts::append));
    magnitude.save(arma::hdf5_name(savePath, "magnitude", arma::hdf5_opts::append));
    phase.save(arma::hdf5_name(savePath, "phase", arma::hdf5_opts::append));
    azim.save(arma::hdf5_name(savePath, "azim", arma::hdf5_opts::append));
    roll.save(arma::hdf5_name(savePath, "roll", arma::hdf5_opts::append));
    pitch.save(arma::hdf5_name(savePath, "pitch", arma::hdf5_opts::append));
    yaw.save(arma::hdf5_name(savePath, "yaw", arma::hdf5_opts::append));

    depression.save(arma::hdf5_name(savePath, "depression", arma::hdf5_opts::append));
    groundPlaneSquint.save(arma::hdf5_name(savePath, "groundPlaneSquint", arma::hdf5_opts::append));
    slantPlaneSquint.save(arma::hdf5_name(savePath, "slantPlaneSquint", arma::hdf5_opts::append));
    range.save(arma::hdf5_name(savePath, "range", arma::hdf5_opts::append));

    targetX.save(arma::hdf5_name(savePath, "targetX", arma::hdf5_opts::append));
    targetY.save(arma::hdf5_name(savePath, "targetY", arma::hdf5_opts::append));
    targetZ.save(arma::hdf5_name(savePath, "targetZ", arma::hdf5_opts::append));
    antennaX.save(arma::hdf5_name(savePath, "antennaX", arma::hdf5_opts::append));
    antennaY.save(arma::hdf5_name(savePath, "antennaY", arma::hdf5_opts::append));
    antennaZ.save(arma::hdf5_name(savePath, "antennaZ", arma::hdf5_opts::append));
    heading.save(arma::hdf5_name(savePath, "heading", arma::hdf5_opts::append));
    xVelocity.save(arma::hdf5_name(savePath, "xVelocity", arma::hdf5_opts::append));

    slowTime.save(arma::hdf5_name(savePath, "slowTime", arma::hdf5_opts::append));
    rangeResolution.save(arma::hdf5_name(savePath, "rangeResolution", arma::hdf5_opts::append));
    crossRangeResolution.save(arma::hdf5_name(savePath, "crossRangeResolution", arma::hdf5_opts::append));
    rangePixelSpacing.save(arma::hdf5_name(savePath, "rangePixelSpacing", arma::hdf5_opts::append));
    crossRangePixelSpacing.save(arma::hdf5_name(savePath, "crossRangePixelSpacing", arma::hdf5_opts::append));

    centreFrequency.save(arma::hdf5_name(savePath, "centreFrequency", arma::hdf5_opts::append));
    bandwidth.save(arma::hdf5_name(savePath, "bandwidth", arma::hdf5_opts::append));
    polarisationType.save(arma::hdf5_name(savePath, "polarisationType", arma::hdf5_opts::append));
}
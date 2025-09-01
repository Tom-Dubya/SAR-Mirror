#include "sample_corr_bp.h"
#include "../constants.h"
#include "../utils/io_utils.h"
#include "../utils/matrix_math.h"

int sample_corr_bp::load()
{
    const std::string& dataPath = this->dataPath;
    load_data(numXSamples, dataPath, "numXSamples");
    load_data(numYSamples, dataPath, "numYSamples");
    load_data(centerX, dataPath, "centreX");
    load_data(centerY, dataPath, "centreY");
    load_data(sceneWidth, dataPath, "sceneWidth");
    load_data(sceneHeight, dataPath, "sceneHeight");
    load_data(minAzimuth, dataPath, "minAzim");
    load_data(maxAzimuth, dataPath, "maxAzim");
    load_data(frequencyStepSize, dataPath, "deltaF");
    load_data(freqMin, dataPath, "minF");
    load_data(freqMax, dataPath, "maxF");
    load_data(pixelX, dataPath, "x_mat");
    load_data(pixelY, dataPath, "y_mat");
    load_data(pixelZ, dataPath, "z_mat");
    load_data(antAzim, dataPath, "AntAzim");
    load_data(antElev, dataPath, "AntElev");
    load_data(phase, dataPath, "phdata");
    return 0;
}

int sample_corr_bp::get_image_data()
{
    const int totalSamples = numXSamples * numYSamples;
    finalImage = arma::cx_mat(numXSamples, numYSamples);
    finalCorrImage = arma::cx_mat(numXSamples, numYSamples);

    const double rangeExtent = c / (2 * frequencyStepSize);
    const int numFreqBins = phase.n_rows;
    const int numPhasePulses = phase.n_cols;
    const int fftSampleCount = 4 * numPhasePulses;
    arma::vec rangeProfile = arma::linspace(-fftSampleCount / 2,fftSampleCount / 2 - 1, fftSampleCount) * rangeExtent / fftSampleCount;
    arma::cx_mat finalImageBuffer(numPhasePulses, totalSamples);

#pragma omp parallel for
    for (int j = 0; j < numPhasePulses; j++)
    {
        const arma::cx_double phaseCorrConstant(0.0, -4.0 * freqMin * pi / c);
        const arma::cx_vec rc = cx_fftshift(arma::ifft(phase.col(j), fftSampleCount));
        const double antennaElevation = antElev.at(j) * radian;
        const double antennaAzimuth = antAzim.at(j) * radian;
        const arma::mat& dRData = pixelX * cos(antennaElevation) * cos(antennaAzimuth)
            + pixelY * cos(antennaElevation) * sin(antennaAzimuth)
            + pixelZ * sin(antennaElevation);
        const arma::uvec& index = arma::find((dRData > arma::min(rangeProfile)) % (dRData < arma::max(rangeProfile)));
        const arma::vec& validDRData = dRData.elem(index);
        arma::cx_mat phaseCorr = arma::exp(phaseCorrConstant * dRData);
        arma::mat interpReal;
        arma::mat interpImag;
        arma::interp1(rangeProfile, arma::real(rc), validDRData,interpReal);
        arma::interp1(rangeProfile, arma::imag(rc), validDRData,interpImag);
        arma::cx_mat finalImage = arma::cx_mat(interpReal, interpImag) % phaseCorr.elem(index);
        for (int k = 0; k < index.n_elem; k++)
        {
            finalImageBuffer.at(j, index(k)) = finalImage(k);
        }
    }

    finalImage = arma::reshape(arma::sum(finalImageBuffer), numXSamples, numYSamples);
    if (correlated)
    {
        const long long fftLength = numPhasePulses * 2 - 1;;
        const long long fftPaddedLength = pow(2, ceil(log2(fftLength)));
        arma::cx_vec correlatedData(totalSamples);
#pragma omp parallel for
        for (int j = 0; j < totalSamples; j++)
        {
            const arma::cx_vec& tmpCol = finalImageBuffer.col(j);
            const arma::cx_vec& tmpColConj = conj(tmpCol);
            correlatedData.at(j) = arma::sum(ffftconv_cx(tmpCol, tmpColConj, fftPaddedLength)) - arma::sum(tmpCol % tmpColConj);
        }
        finalCorrImage = arma::reshape(correlatedData, numXSamples, numYSamples);
    }
    return 0;
}

void sample_corr_bp::generic_run(const std::vector<std::string>& inputPaths, const std::string& savePath, const int from, const int to)
{
    for (int i = from; i < to; i++)
    {
        const std::string& path = inputPaths[i];
        sample_corr_bp ph_mstar_corr_bp(path, true);
        ph_mstar_corr_bp.load();
        ph_mstar_corr_bp.get_image_data();
        std::string parent, file, extension;
        get_file_info(path, parent, file, extension);
        save_data(ph_mstar_corr_bp.finalImage, savePath, file);
        if (ph_mstar_corr_bp.correlated)
        {
            save_data(ph_mstar_corr_bp.finalCorrImage, savePath, file + "_Corr");
        }
        std::cout << "Completed " << path << " (" << i << " / " << (to - from) << " : " << from << " - " << to << ")" << std::endl;
    }
}

int sample_corr_bp::clear()
{
    pixelX.clear();
    pixelY.clear();
    pixelZ.clear();
    antAzim.clear();
    antElev.clear();
    phase.clear();
    return 0;
}

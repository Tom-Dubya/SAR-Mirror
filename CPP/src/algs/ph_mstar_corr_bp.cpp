#include "ph_mstar_corr_bp.h"
#include "../constants.h"
#include "../utils/io_utils.h"
#include "../utils/matrix_math.h"

int ph_mstar_corr_bp::load()
{
    const std::string& dataPath = this->dataPath;
    load_data(numPulses, dataPath, "numPulses");
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
    load_data(antX, dataPath, "AntX");
    load_data(antY, dataPath, "AntY");
    load_data(antZ, dataPath, "AntZ");
    load_data(antAzim, dataPath, "AntAzim");
    load_data(antElev, dataPath, "AntElev");
    load_data(phase, dataPath, "phdata");
    return 0;
}

int ph_mstar_corr_bp::get_image_data()
{
    const int totalSamples = numXSamples * numYSamples;
    finalImages = arma::cx_cube(numPulses, numXSamples, numYSamples);
    finalCorrImages = arma::cx_cube(numPulses, numXSamples, numYSamples);
    for (int i = 0; i < numPulses; i++)
    {
        const double freqStepSize = frequencyStepSize.at(i);
        const double rangeExtent = c / (2 * freqStepSize);
        arma::cx_mat phaseSlice = phase.row(i);
        const int numFreqBins = phaseSlice.n_rows;
        const int numPhasePulses = phase.n_cols;
        const int fftSampleCount = 4 * numPhasePulses;
        arma::mat pixelXSlice = pixelX.row(i);
        arma::mat pixelYSlice = pixelY.row(i);
        arma::mat pixelZSlice = pixelZ.row(i);
        arma::vec rangeProfile = arma::linspace(-fftSampleCount / 2,fftSampleCount / 2 - 1, fftSampleCount) * rangeExtent / fftSampleCount;
        arma::cx_mat finalImageBuffer(numPhasePulses, totalSamples);
#pragma omp parallel for
        for (int j = 0; j < numPhasePulses; j++)
        {
            const double minFreq = freqMin.at(i, j);
            const arma::cx_double phaseCorrConstant(0.0, -4.0 * minFreq * pi / c);
            const arma::cx_vec rc = cx_fftshift(arma::ifft(phaseSlice.col(j), fftSampleCount));
            const double antennaElevation = antElev.at(i, j) * radian;
            const double antennaAzimuth = antAzim.at(i, j) * radian;
            const arma::mat& dRData = pixelXSlice * cos(antennaElevation) * cos(antennaAzimuth)
                + pixelYSlice * cos(antennaElevation) * sin(antennaAzimuth)
                + pixelZSlice * sin(antennaElevation);
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

        finalImages.row(i) = arma::reshape(arma::sum(finalImageBuffer), numXSamples, numYSamples);
        if (correlated)
        {
            long long fftLength = numPhasePulses * 2 - 1;;
            long long fftPaddedLength = pow(2, ceil(log2(fftLength)));
            arma::cx_vec correlatedData(totalSamples);
#pragma omp parallel for
            for (int j = 0; j < totalSamples; j++)
            {
                const arma::cx_vec& tmpCol = finalImageBuffer.col(j);
                const arma::cx_vec& tmpColConj = conj(tmpCol);
                correlatedData.at(j) = arma::sum(ffftconv_cx(tmpCol, tmpColConj, fftPaddedLength)) - arma::sum(tmpCol % tmpColConj);
            }
            finalCorrImages.row(i) = arma::reshape(correlatedData, numXSamples, numYSamples);
        }
    }
    return 0;
}

void ph_mstar_corr_bp::generic_run(const std::vector<std::string>& inputPaths, const std::string& savePath, const int from, const int to)
{
    for (int i = from; i < to; i++)
    {
        const std::string& path = inputPaths[i];
        ph_mstar_corr_bp ph_mstar_corr_bp(path, true);
        ph_mstar_corr_bp.load();
        ph_mstar_corr_bp.get_image_data();
        std::string parent, file, extension;
        get_file_info(path, parent, file, extension);
        save_data(ph_mstar_corr_bp.finalImages, savePath, file);
        if (ph_mstar_corr_bp.correlated)
        {
            save_data(ph_mstar_corr_bp.finalCorrImages, savePath, file + "_Corr");
        }
        std::cout << "Completed " << path << " (" << i << " / " << (to - from) << " : " << from << " - " << to << ")" << std::endl;
    }
}

int ph_mstar_corr_bp::clear()
{
    numFftSamp.clear();
    centerX.clear();
    centerY.clear();
    sceneWidth.clear();
    sceneHeight.clear();
    minAzimuth.clear();
    maxAzimuth.clear();
    frequencyStepSize.clear();
    freqMin.clear();
    freqMax.clear();
    pixelX.clear();
    pixelY.clear();
    pixelZ.clear();
    antX.clear();
    antY.clear();
    antZ.clear();
    antAzim.clear();
    antElev.clear();
    phase.clear();
    finalImages.clear();
    finalCorrImages.clear();
    return 0;
}

#include "target_cp_corr_bp.h"

#include <armadillo>
#include <iostream>

#include "../constants.h"
#include "../utils/io_utils.h"
#include "../utils/matrix_math.h"
#include "../utils/stopwatch.h"

int target_cp_corr_bp::load()
{
    const std::string& dataPath = this->dataPath;
    load_data(antX, dataPath, "x");
    load_data(antY, dataPath, "y");
    azim = normalise(unwrap(arma::vectorise(arma::atan2(antY, antX))));
    load_data(antZ, dataPath, "z");
    load_data(radius, dataPath, "r0");
    load_data(frequencyGHz, dataPath, "freq");
    frequencyGHz = frequencyGHz.t() / 1e9;
    load_data(phase, dataPath, "fq");
    load_data(sceneSize, dataPath, "sceneSize");
    return 0;
}

int target_cp_corr_bp::get_image_data()
{
    stopwatch timer = stopwatch();

    int numSamples = numXSamples * numYSamples;
    arma::uvec azimuthSelector;
    if (minAzimuth > maxAzimuth)
    {
        azimuthSelector = arma::find((azim >= minAzimuth) || (azim <= maxAzimuth));
    }
    else
    {
        azimuthSelector = arma::find((azim >= minAzimuth) % (azim <= maxAzimuth));
    }
    const arma::vec& validAzimuth = azim.elem(azimuthSelector);
    const arma::cx_mat& validPolarized = phase.cols(azimuthSelector).eval();
    unsigned long numPulse = validPolarized.n_cols;
    const int numFftSamples = numPulse * fftSamplingFactor;

    // Setting up the imaging grid.
    arma::mat xGrid;
    arma::mat yGrid;
    mesh_grid(xGrid, yGrid,
        arma::linspace(centerX - sceneSize / 2, centerX + sceneSize / 2, numXSamples),
        arma::linspace(centerY - sceneSize / 2, centerY + sceneSize / 2, numYSamples));

    double deltaFrequency = arma::diff(frequencyGHz.rows(0, 1)).eval()[0] * 1e9;

    double maxWr = c / (2 * deltaFrequency);
    const arma::mat& range = arma::linspace(-numFftSamples / 2.0, numFftSamples / 2.0 - 1, numFftSamples) * maxWr / numFftSamples;
    double rangeMin = arma::min(range).eval()[0];
    double rangeMax =  arma::max(range).eval()[0];

    double minimumFrequency = arma::min(frequencyGHz).eval()[0] * 1e9;
    arma::cx_double phaseCorrConstant(0.0, 4.0 * minimumFrequency * pi / c);

    arma::cx_mat tmp(numPulse, numSamples);

#pragma omp parallel for
    for (int i = 0; i < numPulse; i++)
    {
        const arma::mat& dRData = arma::sqrt(arma::square(antX(i) * arma::ones(numXSamples, numYSamples) - xGrid)
            + arma::square(antY(i) * arma::ones(numXSamples, numYSamples) - yGrid)
            + arma::square(antZ(i) * arma::ones(numXSamples, numYSamples))) - radius(i) * arma::ones(numXSamples, numYSamples);

        const arma::uvec& index = arma::find((dRData > rangeMin) % (dRData < rangeMax));
        const arma::vec& validDRData = dRData.elem(index);

        arma::cx_vec timeData = cx_fftshift(arma::ifft(phase.col(i), numFftSamples));

        // The dual linear interpolations are currently the biggest bottleneck, representing ~60% of execution time.
        arma::vec interpResultsReal;
        arma::vec interpResultsImag;
        arma::interp1(range, arma::real(timeData), validDRData, interpResultsReal);
        arma::interp1(range, arma::imag(timeData), validDRData, interpResultsImag);

        // Transposing accumulates to ~9% of the execution time here.
        // Ideally, this could be avoided if Armadillo didn't make the distinction between row and column vectors.
        arma::cx_mat phaseCorr = arma::exp(phaseCorrConstant * dRData);
        arma::cx_mat tmpValue = arma::cx_vec(interpResultsReal, interpResultsImag) % phaseCorr.elem(index);
        for (int j = 0; j < index.n_elem; j++)
        {
            tmp.at(i, index(j)) = tmpValue(j);
        }
    }

    imageData = arma::reshape(arma::real(arma::sum(tmp)), numXSamples, numYSamples);
    if (correlated)
    {
        long long fftLength = numPulse * 2 - 1;;
        long long fftPaddedLength = pow(2, ceil(log2(fftLength)));
        arma::vec convResults(numSamples);
#pragma omp parallel for
        for (int j = 0; j < numSamples; j++)
        {
            const arma::cx_vec& tmpCol = tmp.col(j);
            const arma::cx_vec& tmpColConj = conj(tmpCol);
            convResults.at(j) = arma::sum(ffftconv(tmpCol, tmpColConj, fftPaddedLength)) - arma::sum(tmpCol % tmpColConj).real();
        }
        correlatedImageData = arma::reshape(convResults, numXSamples, numYSamples);
    }
    std::cout << "Successfully generated image data: " << timer.elapsed_milliseconds() << " ms elapsed" << std::endl;
    return 0;
}

void target_cp_corr_bp::generic_run(const std::vector<std::string>& inputPaths, const std::string& savePath, const int from, const int to)
{
    for (int i = from; i < to; i++)
    {
        const std::string& path = inputPaths[i];
        std::string parent, file, extension;
        get_file_info(path, parent, file, extension);
        target_cp_corr_bp target(path, 4, 160, 160, 0, 0);
        if (target.load() != 0)
        {
            std::cout << "[Error] gen_target_cp failed for <" << path << ">: data loading." << std::endl;
            return;
        }
        target.get_image_data();
        target.save_image_data(savePath, file);
        if (target.correlated)
        {
            save_data(target.correlatedImageData, savePath, file + "_Corr");
        }
        std::cout << "Completed " << path << " (" << i << " / " << (to - from) << " : " << from << " - " << to << ")" << std::endl;
    }
}

int target_cp_corr_bp::clear()
{
    phase.clear();
    azim.clear();
    antX.clear();
    antY.clear();
    antZ.clear();
    radius.clear();
    frequencyGHz.clear();
    return 0;
}


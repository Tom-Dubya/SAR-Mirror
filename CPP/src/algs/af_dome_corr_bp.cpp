#include "af_dome_corr_bp.h"

#include <armadillo>
#include <filesystem>
#include <iostream>

#include "../constants.h"
#include "../polarization_types.h"
#include "../utils/io_utils.h"
#include "../utils/matrix_math.h"
#include "../utils/stopwatch.h"

int af_dome_corr_bp::load()
{
    const std::string& dataPath = this->dataPath;
    load_data(azim, dataPath, "azim");
    load_data(polarized_phase, dataPath, polarizationToString(polarization));
    load_data(frequencyGHz, dataPath, "fghz");
    arma::mat elevationData;
    load_data(elevationData, dataPath, "elev");
    elevation = arma::mat(polarized_phase.n_cols, 1, arma::fill::value(elevationData(0)));
    return 0;
}

int af_dome_corr_bp::get_image_data()
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
    const arma::cx_mat& validPolarized = polarized_phase.cols(azimuthSelector).eval();
    unsigned long numPulse = validPolarized.n_cols;

    // Setting up the imaging grid.
    arma::mat xGrid;
    arma::mat yGrid;
    mesh_grid(xGrid, yGrid,
        arma::linspace(centerX - sceneWidth / 2, centerX + sceneWidth / 2, numXSamples),
        arma::linspace(centerY - sceneHeight / 2, centerY + sceneHeight / 2, numYSamples));

    arma::mat cosElevation = cos(elevation * radian);
    double deltaFrequency = arma::diff(frequencyGHz.rows(0, 1)).eval()[0] * 1e9;

    double maxWr = c / (2 * deltaFrequency);
    const arma::mat& range = arma::linspace(-numFftSamp / 2.0, numFftSamp / 2.0 - 1, numFftSamp) * maxWr / numFftSamp;
    double rangeMin = arma::min(range).eval()[0];
    double rangeMax =  arma::max(range).eval()[0];

    double minimumFrequency = arma::min(frequencyGHz).eval()[0] * 1e9;
    arma::cx_double phaseCorrConstant(0.0, 4.0 * minimumFrequency * pi / c);

    arma::cx_mat tmp(numPulse, numSamples);

#pragma omp parallel for
    for (int i = 0; i < numPulse; i++)
    {
        double azimuthValue = validAzimuth(i) * radian;
        const arma::mat& dRData = (xGrid * cosElevation(i)* cos(azimuthValue) + yGrid * cosElevation(i) * sin(azimuthValue)).t();
        const arma::vec& validDRData = dRData.elem(arma::find((dRData > rangeMin) % (dRData < rangeMax)));
        arma::cx_vec timeData = cx_fftshift(arma::ifft(validPolarized.col(i), numFftSamp));

        // The dual linear interpolations are currently the biggest bottleneck, representing ~60% of execution time.
        arma::vec interpResultsReal;
        arma::vec interpResultsImag;
        arma::interp1(range, arma::real(timeData), validDRData, interpResultsReal);
        arma::interp1(range, arma::imag(timeData), validDRData, interpResultsImag);

        // Transposing accumulates to ~9% of the execution time here.
        // Ideally, this could be avoided if Armadillo didn't make the distinction between row and column vectors.
        tmp.row(i) = arma::trans(arma::cx_vec(interpResultsReal, interpResultsImag) % arma::exp(phaseCorrConstant * validDRData));
    }

    long long fftLength = numPulse * 2 - 1;;
    long long fftPaddedLength = pow(2, ceil(log2(fftLength)));
    arma::mat convResults(fftLength, numSamples);

#pragma omp parallel for
    for (int j = 0; j < numSamples; j++)
    {
        const arma::cx_vec& tmpCol = tmp.col(j);
        const arma::cx_vec& tmpColConj = conj(tmpCol);
        convResults.col(j) = ffftconv(tmpCol, tmpColConj, fftPaddedLength).subvec(0, fftLength - 1);
    }

    imageData = arma::reshape(arma::sum(convResults, 0) - convResults.row(0), numXSamples, numYSamples);
    std::cout << "Successfully generated image data: " << timer.elapsed_milliseconds() << " ms elapsed" << std::endl;
    return 0;
}



int af_dome_corr_bp::clear()
{
    polarized_phase.clear();
    azim.clear();
    elevation.clear();
    frequencyGHz.clear();
    return 0;
}

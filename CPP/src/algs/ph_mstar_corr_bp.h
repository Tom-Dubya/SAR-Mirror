#ifndef PH_MSTAR_CORR_BP_H
#define PH_MSTAR_CORR_BP_H

#include "base_correlated_back_projection.h"
#include <armadillo>


class ph_mstar_corr_bp : public base_correlated_back_projection
{
public:
    bool correlated;

    int numPulses;

    int numXSamples;

    int numYSamples;

    arma::vec numFftSamp;

    arma::vec centerX;

    arma::vec centerY;

    arma::vec sceneWidth;

    arma::vec sceneHeight;

    arma::vec minAzimuth;

    arma::vec maxAzimuth;

    arma::vec frequencyStepSize;

    arma::mat freqMin;

    arma::mat freqMax;

    arma::cube pixelX;

    arma::cube pixelY;

    arma::cube pixelZ;

    arma::vec antX;

    arma::vec antY;

    arma::vec antZ;

    arma::mat antAzim;

    arma::mat antElev;

    arma::cx_cube phase;

    arma::cx_cube finalImages;

    arma::cx_cube finalCorrImages;

    explicit ph_mstar_corr_bp(const std::string& dataPath, const bool correlated = true)
    {
        this->correlated = correlated;
        this->dataPath = dataPath;
    }

    int load() override;

    int get_image_data() override;

    int clear() override;

    static void generic_run(const std::vector<std::string>& inputPaths, const std::string& savePath, const int from, const int to);

    void set_azimuth_bounds(const int min, const int max)
    {
        minAzimuth = min;
        maxAzimuth = max;
    }
};



#endif //PH_MSTAR_CORR_BP_H

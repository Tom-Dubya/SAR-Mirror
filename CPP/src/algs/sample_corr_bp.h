#ifndef SAMPLE_CORR_BP_H
#define SAMPLE_CORR_BP_H

#include "base_correlated_back_projection.h"
#include <armadillo>


class sample_corr_bp : public base_correlated_back_projection
{
public:
    bool correlated;

    int numPulses;

    int numXSamples;

    int numYSamples;

    double numFftSamp;

    double centerX;

    double centerY;

    double sceneWidth;

    double sceneHeight;

    double minAzimuth;

    double maxAzimuth;

    double frequencyStepSize;

    double freqMin;

    double freqMax;

    arma::mat pixelX;

    arma::mat pixelY;

    arma::mat pixelZ;

    arma::vec antAzim;

    arma::vec antElev;

    arma::cx_mat phase;

    arma::cx_mat finalImage;

    arma::cx_mat finalCorrImage;

    explicit sample_corr_bp(const std::string& dataPath, const bool correlated = true)
    {
        this->correlated = correlated;
        this->dataPath = dataPath;
    }

    int load() override;

    int get_image_data() override;

    int clear() override;

    static void generic_run(const std::vector<std::string>& inputPaths, const std::string& savePath, const int from, const int to);
};



#endif //SAMPLE_CORR_BP_H

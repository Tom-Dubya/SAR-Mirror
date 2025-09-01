#ifndef TARGET_CP_CORR_BP_H
#define TARGET_CP_CORR_BP_H
#include "base_correlated_back_projection.h"

#include <armadillo>


class target_cp_corr_bp : public base_correlated_back_projection
{
    public:
        int numXSamples;

        int numYSamples;

        int fftSamplingFactor;

        float centerX;

        float centerY;

        double sceneSize;

        float minAzimuth;

        float maxAzimuth;

        arma::cx_mat phase;

        arma::mat azim;

        arma::mat antX;

        arma::mat antY;

        arma::mat antZ;

        arma::mat radius;

        arma::mat frequencyGHz;

        arma::mat correlatedImageData;

        target_cp_corr_bp(const std::string &dataPath,
            const int fftSamplingFactor, const int numXSamp, const int numYSamp,
            const float centerX, const float centerY, const bool correlated = true)
        {
            this->dataPath = dataPath;
            this->minAzimuth = 0;
            this->maxAzimuth = 360;
            this->fftSamplingFactor = fftSamplingFactor;
            this->numXSamples = numXSamp;
            this->numYSamples = numYSamp;
            this->centerX = centerX;
            this->centerY = centerY;
            this->sceneSize = 10;
            this->correlated = correlated;
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



#endif
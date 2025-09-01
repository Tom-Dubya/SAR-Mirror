#ifndef AF_DOME_CORR_BP_H
#define AF_DOME_CORR_BP_H
#include "base_correlated_back_projection.h"

#include <armadillo>

#include "../polarization_types.h"

class af_dome_corr_bp : public base_correlated_back_projection
{
    public:
        int numXSamples;

        int numYSamples;

        int numFftSamp;

        float centerX;

        float centerY;

        float sceneWidth;

        float sceneHeight;

        float minAzimuth;

        float maxAzimuth;

        polarization_types polarization;

        arma::cx_mat polarized_phase;

        arma::mat azim;

        arma::mat elevation;

        arma::mat frequencyGHz;

        af_dome_corr_bp(const std::string &dataPath, const polarization_types polarization,
            const float sceneWidth, const float sceneHeight,
            const int numFftSamp, const int numXSamp, const int numYSamp,
            const float centerX, const float centerY, const bool correlated = true)
        {
            this->dataPath = dataPath;
            this->polarization = polarization;
            this->minAzimuth = 0;
            this->maxAzimuth = 360;
            this->sceneWidth = sceneWidth;
            this->sceneHeight = sceneHeight;
            this->numFftSamp = numFftSamp;
            this->numXSamples = numXSamp;
            this->numYSamples = numYSamp;
            this->centerX = centerX;
            this->centerY = centerY;
            this->correlated = correlated;
        }

        int load() override;

        int get_image_data() override;

        int clear() override;

        void set_azimuth_bounds(const int min, const int max)
        {
            minAzimuth = min;
            maxAzimuth = max;
        }
};



#endif

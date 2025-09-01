#ifndef MSTAR_AGGREGATOR_H
#define MSTAR_AGGREGATOR_H

#include <armadillo>
#include <map>
#include <string>


class mstar_aggregator
{
    public:
        std::string mstarPath;

        bool debug;

        std::map<std::string, std::string> header;

        int numXSamples = -1;

        int numYSamples = -1;

        arma::mat magnitude;

        arma::mat phase;

        arma::vec nRows;

        arma::vec nCols;

        arma::vec azim;

        arma::vec roll;

        arma::vec pitch;

        arma::vec yaw;

        arma::vec depression;

        arma::vec groundPlaneSquint;

        arma::vec slantPlaneSquint;

        arma::vec range;

        arma::vec targetX;

        arma::vec targetY;

        arma::vec targetZ;

        arma::vec antennaX;

        arma::vec antennaY;

        arma::vec antennaZ;

        arma::vec heading;

        arma::vec xVelocity;

        arma::vec slowTime;

        arma::vec rangeResolution;

        arma::vec crossRangeResolution;

        arma::vec rangePixelSpacing;

        arma::vec crossRangePixelSpacing;

        arma::vec centreFrequency;

        arma::vec bandwidth;

        arma::vec polarisationType;

        explicit mstar_aggregator(std::string mstarPath, bool debug = false);

        void load(bool fullLoad);

        void save(const std::string& rawSavePath);
};



#endif //MSTAR_AGGREGATOR_H

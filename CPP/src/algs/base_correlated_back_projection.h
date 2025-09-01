#ifndef BASE_CORRELATED_BACK_PROJECTION_H
#define BASE_CORRELATED_BACK_PROJECTION_H

#include <armadillo>
#include <filesystem>
#include <string>

#include "../utils/io_utils.h"


class base_correlated_back_projection
{
    public:
        bool correlated;

        std::string dataPath;

        arma::mat imageData;

        virtual int load() = 0;

        virtual int get_image_data() = 0;

        bool save_image_data(const std::string& savePath, const std::string& saveName) const
        {
            return save_data(imageData, savePath, saveName);
        }

        virtual int clear() = 0;

        virtual ~base_correlated_back_projection() = default;
};



#endif
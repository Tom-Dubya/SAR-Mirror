#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <armadillo>
#include <string>

#include "file_utils.h"

inline void load_debug_message(const bool debug, const std::string& dataPath, const std::string& dataName)
{
    if (debug)
    {
        std::cout << "Failed to find " + dataName + " data associated with " + dataPath << std::endl;
    }
}

inline bool load_data(arma::mat& destination, const std::string& dataPath, const std::string& dataName, const bool debug = false)
{
    if (!destination.load(arma::hdf5_name(dataPath, dataName)) && !destination.load(arma::hdf5_name(dataPath, "data/" + dataName)))
    {
        load_debug_message(debug, dataPath, dataName);
        return false;
    }
    return true;
};

inline bool load_data(arma::cx_mat& destination, const std::string& dataPath, const std::string& dataName, const bool debug = false)
{
    if (!destination.load(arma::hdf5_name(dataPath, dataName)) && !destination.load(arma::hdf5_name(dataPath, "data/" + dataName)))
    {
        load_debug_message(debug, dataPath, dataName);
        return false;
    }
    return true;
};

inline bool load_data(arma::cube& destination, const std::string& dataPath, const std::string& dataName, const bool debug = false)
{
    if (!destination.load(arma::hdf5_name(dataPath, dataName)) && !destination.load(arma::hdf5_name(dataPath, "data/" + dataName)))
    {
        load_debug_message(debug, dataPath, dataName);
        return false;
    }
    return true;
};

inline bool load_data(arma::cx_cube& destination, const std::string& dataPath, const std::string& dataName, const bool debug = false)
{
    if (!destination.load(arma::hdf5_name(dataPath, dataName)) && !destination.load(arma::hdf5_name(dataPath, "data/" + dataName)))
    {
        load_debug_message(debug, dataPath, dataName);
        return false;
    }
    return true;
};

inline bool load_data(double& destination, const std::string& dataPath, const std::string& dataName, const bool debug = false)
{
    if (arma::vec temp; !temp.load(arma::hdf5_name(dataPath, dataName)) && !temp.load(arma::hdf5_name(dataPath, "data/" + dataName)))
    {
        load_debug_message(debug, dataPath, dataName);
        return false;
    }
    else
    {
        destination = static_cast<double>(temp.at(0));
    }
    return true;
};

inline bool load_data(int& destination, const std::string& dataPath, const std::string& dataName, const bool debug = false)
{
    if (arma::vec temp; !temp.load(arma::hdf5_name(dataPath,  dataName)) && !temp.load(arma::hdf5_name(dataPath, "data/" + dataName)))
    {
        load_debug_message(debug, dataPath, dataName);
        return false;
    }
    else
    {
        destination = static_cast<int>(temp.at(0));
    }
    return true;
};

static bool save_data(const arma::mat& data, const std::string& savePath, const std::string& saveName)
{
    std::filesystem::create_directory(savePath);
    std::string outputName = saveName;
    if (has_extension(saveName))
    {
        std::string parent, name, extension;
        get_file_info(saveName, parent, name, extension);
        outputName = name;
    }
    return data.save(savePath + "/" + outputName + ".hdf5", arma::hdf5_binary);
}

static bool save_data(const arma::cx_mat& data, const std::string& savePath, const std::string& saveName)
{
    std::filesystem::create_directory(savePath);
    std::string outputName = saveName;
    if (has_extension(saveName))
    {
        std::string parent, name, extension;
        get_file_info(saveName, parent, name, extension);
        outputName = name;
    }
    return data.save(savePath + "/" + outputName + ".hdf5", arma::hdf5_binary);
}

static bool save_data(const arma::cube& data, const std::string& savePath, const std::string& saveName)
{
    std::filesystem::create_directory(savePath);
    std::string outputName = saveName;
    if (has_extension(saveName))
    {
        std::string parent, name, extension;
        get_file_info(saveName, parent, name, extension);
        outputName = name;
    }
    return data.save(savePath + "/" + outputName + ".hdf5", arma::hdf5_binary);
}

static bool save_data(const arma::cx_cube& data, const std::string& savePath, const std::string& saveName)
{
    std::filesystem::create_directory(savePath);
    std::string outputName = saveName;
    if (has_extension(saveName))
    {
        std::string parent, name, extension;
        get_file_info(saveName, parent, name, extension);
        outputName = name;
    }
    return data.save(savePath + "/" + outputName + ".hdf5", arma::hdf5_binary);
}
#endif //IO_UTILS_H

#include <filesystem>
#include <iostream>

#include "src/algs/af_dome_corr_bp.h"
#include "src/algs/mstar_aggregator.h"
#include "src/algs/sample_corr_bp.h"
#include "src/utils/string_utils.h"

using namespace std;
using namespace arma;

int main(int argc, char* argv[])
{
    int partition = 0;
    int partitionCount = 1;

    if (argc > 1)
    {
        partition = std::stoi(argv[1]);
        partitionCount = std::stoi(argv[2]);
    }

    std::filesystem::path currentPath = std::filesystem::current_path();
    while (!ends_with(currentPath.string(), "SAR"))
    {
        currentPath = currentPath.parent_path();
    }
    std::string dataPath = currentPath.string() + "/Data/";

    ifstream inputStream(dataPath + "DataPaths.txt");
    if (!inputStream.is_open())
    {
        std::cout << "Failed to read data paths" << std::endl;
        return -1;
    }

    std::string input;
    std::vector<std::string> inputPaths{};
    while (std::getline(inputStream, input))
    {
        inputPaths.insert(inputPaths.end(), input);
    }

    const int count = inputPaths.size();
    const int segmentSize = count / partitionCount;
    const int from = partition * segmentSize;
    int to = from + segmentSize;
    if (partition == partitionCount - 1)
    {
        to = count;
    }

    //ph_mstar_corr_bp::generic_run(inputPaths, "output/mstar", from, to);
    sample_corr_bp::generic_run(inputPaths, "output/sample", from, to);
    // target_cp_corr_bp::generic_run(inputPaths, "output/tcp", from, to);
    return 0;
}
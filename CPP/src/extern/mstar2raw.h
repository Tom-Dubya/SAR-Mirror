#ifndef MSTAR2RAW_H
#define MSTAR2RAW_H
#include <string>

#include "../utils/string_utils.h"

#ifdef __cplusplus
extern "C"
{
#endif
    void mstar2raw_main(int argc, char* argv[], int* phx_size, char** phx_header, long* chip_size, float** chip_data, long* fscene_size, unsigned short** fscene_mag, unsigned short** fscene_phase);
#ifdef __cplusplus
}

inline void decompress_mstar(const std::string& path, std::map<std::string, std::string>& header, arma::vec& magnitude, arma::vec& phase)
{
    char* argv[] = { const_cast<char *>("mstar2raw"), const_cast<char *>(path.c_str()) };
    int phxSize;
    char* phxHeader;
    long chipSize;
    float* chipData;
    long fsceneSize;
    unsigned short* fsceneMag;
    unsigned short* fscenePhase;
    mstar2raw_main(3, argv, &phxSize, &phxHeader, &chipSize, &chipData, &fsceneSize, &fsceneMag, &fscenePhase);

    header = std::map<std::string, std::string>();
    std::vector<std::string> header_data = split(phxHeader, "\n");
    for (const std::string& data : header_data)
    {
        if (data.find("= ") == std::string::npos)
        {
            continue;
        }

        std::vector<std::string> data_key_pair = split(data, "=");
        header[data_key_pair[0]] = data_key_pair[1].substr(1);
    }

    if (chipSize > 0)
    {
        arma::vec tmp_chip_data(chipSize);
        for (int i = 0; i < chipSize; ++i)
        {
            tmp_chip_data(i) = static_cast<double>(chipData[i]);
        }

        magnitude = tmp_chip_data.head(chipSize / 2);
        phase = tmp_chip_data.tail(chipSize / 2);
    }
    else
    {
        magnitude = arma::vec(fsceneSize);
        for (int i = 0; i < fsceneSize; ++i)
        {
            magnitude(i) = static_cast<unsigned short>(fsceneMag[i]);
        }

        phase = arma::vec(fsceneSize);
        for (int i = 0; i < fsceneSize ; ++i)
        {
            phase(i) = static_cast<unsigned short>(fscenePhase[i]);
        }
    }
}

#endif
#endif

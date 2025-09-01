#ifndef POLARIZATION_TYPES_H
#define POLARIZATION_TYPES_H

enum class polarization_types
{
    HH, // Horizontal-Horizontal sent and received polarized radar waves
    HV, // Horizontal-Vertical sent and received polarized radar waves
    VV // Vertical-Vertical sent and received polarized radar waves
};

static std::string polarizationToString(polarization_types polarization)
{
    switch (polarization)
    {
        case polarization_types::HH:
            return "hh";

        case polarization_types::HV:
            return "hv";

        case polarization_types::VV:
            return "vv";
    }
    return "";
}

#endif //POLARIZATION_TYPES_H

#ifndef MISC_UTILS_H
#define MISC_UTILS_H

#include <cmath>

static double std_latitude(double latitude)
{
    while (latitude > 90)
    {
        latitude -= 180;
    }

    while (latitude < -90)
    {
        latitude += 180;
    }

    return latitude;
}

static double std_longitude(double longitude)
{
    while (longitude > 180)
    {
        longitude -= 360;
    }

    while (longitude < -180)
    {
        longitude += 360;
    }

    return longitude;
}

static double longitude_to_meters(const double latitude, const double longitude)
{
    return 111320 * std::cos(latitude) * longitude;
}

static double latitude_to_meters(const double latitude)
{
    return 111320 * latitude;
}

static void geodetic_to_planar_projection(const double latitude, const double longitude, double& X, double& Y)
{
    X = longitude_to_meters(latitude, longitude);
    Y = latitude_to_meters(latitude);
}

// See: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
static void geodetic_to_ECEF(const double latitude, const double longitude, const double height, double& X, double& Y, double& Z)
{
    const double equatorial_radius_squared = std::pow(6378137.0, 2);
    const double polar_radius_squared = std::pow(6356752.3, 2);
    const double latitude_cos = std::cos(latitude);
    const double latitude_cos_squared = std::pow(latitude_cos, 2);
    const double latitude_sin = std::sin(latitude);
    const double latitude_sin_squared = std::pow(latitude_sin, 2);
    const double prime_vertical_radius_of_curviture = equatorial_radius_squared / std::sqrt(equatorial_radius_squared * latitude_cos_squared + polar_radius_squared * latitude_sin_squared);
    X = (prime_vertical_radius_of_curviture + height) * latitude_cos * std::cos(longitude);
    Y = (prime_vertical_radius_of_curviture + height) * latitude_cos * std::sin(longitude);
    Z = (polar_radius_squared / equatorial_radius_squared * prime_vertical_radius_of_curviture + height) * std::sin(latitude);
}

#endif
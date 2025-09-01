#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#ifdef linux
#define ARMA_USE_FFTW3
#endif
#include<armadillo>

inline void mesh_grid(arma::mat& X, arma::mat& Y, const arma::vec& x, const arma::vec& y)
{
    X = repmat(x, 1, y.n_elem);
    Y = repmat(y.t(), x.n_elem, 1);
}

inline arma::mat fix(const arma::mat& input)
{
    arma::mat output = input;
    output.transform([](const double value)
    {
        if (value > 0)
        {
            return floor(value);
        }

        if (value < 0)
        {
            return ceil(value);
        }

        return value;
    });
    return output;
}

inline arma::mat circ_shift(arma::mat in, const unsigned long  rowShift, const unsigned long  colShift)
{
    arma::mat output = in;
    const unsigned long cols = in.n_cols;
    const unsigned long rows = in.n_rows;
    for (int j = 0; j < cols; j++)
    {
        const unsigned long jj = (j + colShift) % cols;
        for (int i = 0; i < rows; i++)
        {
            const unsigned long ii = (i + rowShift) % rows;
            output[jj * rows + ii] = in[j * rows + i];
        }
    }
    return output;
}

inline arma::cx_mat circ_shift(arma::cx_mat in, const unsigned long  rowShift, const unsigned long  colShift)
{
    arma::cx_mat output = in;
    const unsigned long cols = in.n_cols;
    const unsigned long rows = in.n_rows;
    for (int j = 0; j < cols; j++)
    {
        const unsigned long jj = (j + colShift) % cols;
        for (int i = 0; i < rows; i++)
        {
            const unsigned long ii = (i + rowShift) % rows;
            output[jj * rows + ii] = in[j * rows + i];
        }
    }
    return output;
}

inline arma::mat fftshift(const arma::mat& input)
{
    const arma::vec dimensions = {static_cast<double>(input.n_rows), static_cast<double>(input.n_cols)};
    arma::mat shift = fix(dimensions / 2);
    return circ_shift(input, shift[0], shift[1]);
}

inline arma::cx_vec cx_fftshift(const arma::cx_mat& input)
{
    const arma::vec dimensions = {static_cast<double>(input.n_rows), static_cast<double>(input.n_cols)};
    arma::mat shift = fix(dimensions / 2);
    return circ_shift(input, shift[0], shift[1]);
}

inline arma::vec fftconv(const arma::cx_vec& first, const arma::cx_vec& second)
{
    double length = first.n_elem + second.n_elem - 1;
    double paddedLength = pow(2, ceil(log2(length)));
    arma::cx_vec first_fft = arma::fft(first, paddedLength);
    arma::cx_vec second_fft = arma::fft(second, paddedLength);
    arma::vec reals = arma::real(arma::ifft(first_fft % second_fft, paddedLength));
    return length >= reals.n_elem ? reals : reals.head(length);
}

inline arma::vec fftconv(const arma::cx_vec& first, const arma::cx_vec& second, const long long length, const long long paddedLength)
{
    arma::vec reals = arma::real(arma::ifft(arma::fft(first, paddedLength) % arma::fft(second, paddedLength), paddedLength));
    return length >= reals.n_elem ? reals : reals.head(length);
}

inline arma::vec ffftconv(const arma::cx_vec& first, const arma::cx_vec& second, const long long paddedLength)
{
    return arma::real(arma::ifft(arma::fft(first, paddedLength) % arma::fft(second, paddedLength), paddedLength));
}

inline arma::cx_vec ffftconv_cx(const arma::cx_vec& first, const arma::cx_vec& second, const long long paddedLength)
{
    return arma::ifft(arma::fft(first, paddedLength) % arma::fft(second, paddedLength), paddedLength);
}

inline arma::vec unwrap(const arma::vec& phase_angles)
{
    arma::vec unwrapped_angles = phase_angles;
    double prev_angle = phase_angles(0);
    for (size_t i = 1; i < phase_angles.n_elem; i++)
    {
        double deltaAngle = phase_angles(i) - prev_angle;
        while (deltaAngle > pi)
        {
            unwrapped_angles(i) -= 2.0 * pi;
            deltaAngle = unwrapped_angles(i) - prev_angle;
        }

        while (deltaAngle < -pi)
        {
            unwrapped_angles(i) += 2.0 * pi;
            deltaAngle = unwrapped_angles(i) - prev_angle;

        }
        prev_angle = unwrapped_angles(i);
    }
    return unwrapped_angles;
}

inline arma::vec normalise(const arma::vec& radians)
{
    arma::vec output = radians;
    for (size_t i = 0; i < output.n_elem; i++)
    {
        double currentRadians = output(i);
        while (currentRadians > 2 * pi)
        {
            currentRadians -= 2.0 * pi;
        }

        while (currentRadians < 0)
        {
            currentRadians += 2.0 * pi;
        }
        output(i) = currentRadians;
    }
    return output;
}

inline double euclidean_distance_squared(const arma::vec& iVec, const arma::vec& jVec)
{
    double sum = 0;
    for (int i = 0; i < iVec.n_elem; i++)
    {
        sum += std::pow(jVec(i) - iVec(i), 2);
    }
    return sum;
}

inline double euclidean_distance(const arma::vec& iVec, const arma::vec& jVec)
{
    double sum = 0;
    for (int i = 0; i < iVec.n_elem; i++)
    {
        sum += std::pow(jVec(i) - iVec(i), 2);
    }
    return std::sqrt(sum);
}

inline double square(const double x)
{
    return std::pow(x, 2);
}

#endif //MATRIX_MATH_H

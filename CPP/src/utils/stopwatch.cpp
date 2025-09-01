#include "iostream"
#include "stopwatch.h"

stopwatch::stopwatch()
{
    begin = std::chrono::high_resolution_clock::now();
    stopped = false;
}

void stopwatch::stop()
{
    end = std::chrono::high_resolution_clock::now();
}

void stopwatch::start()
{
    stopped = false;
}

void stopwatch::restart()
{
    begin = std::chrono::high_resolution_clock::now();
    stopped = false;
}

std::chrono::duration<long long, std::ratio<1, 1000000000>> stopwatch::get_elapsed()
{
    if (stopped)
    {
        return end - begin;
    }
    return std::chrono::high_resolution_clock::now() - begin;
}

long long stopwatch::elapsed_seconds()
{
    const auto elapsed = get_elapsed();
    return std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
}

long long stopwatch::elapsed_seconds(const std::chrono::duration<long long, std::ratio<1, 1000000000>> elapsed)
{
    return std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
}

long long stopwatch::elapsed_milliseconds()
{
    const auto elapsed = get_elapsed();
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

long long stopwatch::elapsed_milliseconds(const std::chrono::duration<long long, std::ratio<1, 1000000000>> elapsed)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

long long stopwatch::elapsed_microseconds()
{
    const auto elapsed = get_elapsed();
    return std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
}

long long stopwatch::elapsed_microseconds(const std::chrono::duration<long long, std::ratio<1, 1000000000>> elapsed)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
}

long long stopwatch::elapsed_ticks()
{
    const auto elapsed = get_elapsed();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
}

long long stopwatch::elapsed_ticks(const std::chrono::duration<long long, std::ratio<1, 1000000000>> elapsed)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
}

void stopwatch::display_elapsed()
{
    const auto elapsed = get_elapsed();
    std::cout << "Elapsed: " << elapsed_ticks(elapsed) << " ticks, " << elapsed_milliseconds(elapsed) << " ms" << std::endl;
}


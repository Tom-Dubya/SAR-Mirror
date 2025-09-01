#ifndef STOPWATCH_H
#define STOPWATCH_H
#include <chrono>


class stopwatch
{
    public:
        std::chrono::time_point<std::chrono::system_clock> begin;

        std::chrono::time_point<std::chrono::system_clock> end;

        bool stopped;

        stopwatch();

        void stop();

        void start();

        void restart();

        std::chrono::duration<long long, std::ratio<1, 1000000000>> get_elapsed();

        void display_elapsed();

        long long elapsed_seconds();

        static long long elapsed_seconds(std::chrono::duration<long long, std::ratio<1, 1000000000>> elapsed);

        long long elapsed_milliseconds();

        static long long elapsed_milliseconds(std::chrono::duration<long long, std::ratio<1, 1000000000>> elapsed);

        long long elapsed_microseconds();

        static long long elapsed_microseconds(std::chrono::duration<long long, std::ratio<1, 1000000000>> elapsed);

        long long elapsed_ticks();

        static long long elapsed_ticks(std::chrono::duration<long long, std::ratio<1, 1000000000>> elapsed);
};



#endif //STOPWATCH_H

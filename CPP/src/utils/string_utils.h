#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <string_view>
#include <regex>

static bool ends_with(const std::string_view str, const std::string_view suffix)
{
    const int stringSize = str.size();
    const int suffixSize = suffix.size();
    return stringSize >= suffixSize && str.compare(stringSize - suffixSize, suffixSize, suffix) == 0;
}

static std::vector<std::string> split(const std::string& input, const std::string& delimiter)
{
    std::vector<std::string> output;
    const std::regex re(delimiter);
    std::sregex_token_iterator it(input.begin(), input.end(), re, -1);
    for (const std::sregex_token_iterator regEnd; it != regEnd; ++it)
    {
        output.push_back(it->str());
    }
    return output;
}


#endif //STRING_UTILS_H

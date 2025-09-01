#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <filesystem>
#include <string>
#include <vector>
#include <regex>

inline bool has_extension(const std::string& pathString)
{
    const std::filesystem::path path(pathString);
    return path.has_extension();
}

inline bool get_extension(const std::string& pathString, std::string& extension)
{
    const std::filesystem::path path(pathString);
    if (!path.has_extension())
    {
        return false;
    }

    extension = path.extension().string();
    return true;
}

inline void get_file_info(const std::string& pathString, std::string& parentPath, std::string& name, std::string& extension)
{
    const std::filesystem::path path(pathString);
    parentPath = path.parent_path().string();
    name = path.stem().string();
    extension = path.extension().string();
}

inline std::vector<std::string> get_files_in_directory(const std::string& directoryPath)
{
    std::vector<std::string> file_list;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directoryPath))
    {
        if (entry.is_regular_file())
        {
            file_list.push_back(entry.path().string());
        }
    }
    return file_list;
}

inline std::vector<std::string> get_files_in_directory_with_validation(const std::string& directoryPath, const std::string& validationPattern)
{
    std::regex validationRegex(validationPattern);
    std::smatch _;
    std::vector<std::string> file_list;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directoryPath))
    {
        if (entry.is_regular_file())
        {
            std::string path = entry.path().string();
            if (std::regex_search(path, _, validationRegex))
            {
                file_list.push_back(path);
            }
        }
    }
    return file_list;
}

inline std::vector<std::string> get_files_in_directory_with_validation(const std::string& directoryPath, const std::string& validationPattern, const std::string& antiValidationPattern)
{
    const std::regex validationRegex(validationPattern);
    const std::regex antiValidationRegex(antiValidationPattern);
    std::smatch _;
    std::vector<std::string> file_list;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directoryPath))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        std::string path = entry.path().string();
        std::string fileName = entry.path().filename().string();
        if (std::regex_search(fileName, _, validationRegex) && !std::regex_search(fileName, _, antiValidationRegex))
        {
            file_list.push_back(path);
        }
    }
    return file_list;
}

#endif

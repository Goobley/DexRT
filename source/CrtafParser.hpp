#if !defined(DEXRT_CRTAF_PARSER_HPP)
#define DEXRT_CRTAF_PARSER_HPP

#include <cstdio>
#include <string>
#include <yaml-cpp/yaml.h>
#include <vector>

inline std::vector<char> read_entire_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    std::vector<char> result;
    if (!f) {
        return result;
    }

    fseek(f, 0, SEEK_END);
    int length = ftell(f);
    fseek(f, 0, SEEK_SET);
    result.resize(length + 1);

    int num_read = fread(result.data(), length, 1, f);
    if (num_read != 1) {
        return result;
    }
    result[length] = '\0';
    fclose(f);
    return result;
}

inline YAML::Node parse_crtaf_model(const std::string& path) {
    // auto yaml_buf = read_entire_file(path);
    YAML::Node file = YAML::LoadFile(path);
    return file;
}


#else
#endif
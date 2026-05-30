#pragma once

#include <vector>

std::vector<float>& operator*=(std::vector<float>&, std::vector<float>&);
std::vector<float> operator*(float scalar, const std::vector<float>& v);
std::vector<float> operator-(const std::vector<float>& v1, const std::vector<float>& v2);
std::vector<float> operator+(std::vector<float>&, std::vector<float>&);
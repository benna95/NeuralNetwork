#include "VectorOperations.h"

std::vector<float>& operator*=(std::vector<float>& v1, std::vector<float>& v2)
{
	if (v1.size() != v2.size())
	{
		return v1;
	}

	else
	{
		for (size_t i = 0; i < v1.size(); i++)
		{
			v1[i] = v1[i] * v2[i];
		}
	}

	return v1;
}

std::vector<float> operator*(float scalar,
	const std::vector<float>& v)
{
	std::vector<float> result(v.size());

	for (size_t i = 0; i < v.size(); i++)
	{
		result[i] = scalar * v[i];
	}

	return result;
}

std::vector<float> operator-(const std::vector<float>& v1,
	const std::vector<float>& v2)
{
	std::vector<float> result(v1.size());

	for (size_t i = 0; i < v1.size(); i++)
	{
		result[i] = v1[i] - v2[i];
	}

	return result;
}

std::vector<float> operator+(std::vector<float>& v1, std::vector<float>& v2)
{
	if (v1.size() != v2.size())
	{
		return v1;
	}

	else
	{
		for (size_t i = 0; i < v1.size(); i++)
		{
			v1[i] = v1[i] + v2[i];
		}
	}

	return v1;
}
#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <fstream>
#include <math.h>

float generate_noisy(float x)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // rumore gaussiano
    static std::normal_distribution<float> dist(0.0f, 0.03f);

    float noise = dist(gen);

    //return (1.0f / (1.0f + x * x)) + noise;
	//return ( 2 * x + 1) + noise;
	return 0.8f * tanh(1.5f * x - 0.2f) + 0.1f + noise;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: program N\n";
        return 1;
    }

    int N = std::stoi(argv[1]);

    std::vector<std::pair<float, float>> values;
    values.reserve(N);
	
	std::ofstream file("data.txt", std::ios::out);

    for (int i = 0; i < N; i++)
    {
        float x = -3.0f + i * (6.0f / (N - 1));

        float y = generate_noisy(x);

        values.push_back(std::make_pair(x, y));

        std::cout << x << "," << y << std::endl;
		file << x << "," << y << std::endl;
    }
	file.close();
}
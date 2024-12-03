#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <omp.h>

enum hybrid_mode
{
    hybrid_off = 2,
    full_hybrid = 0,
    lazy_hybrid = 1
};

void blurBox(unsigned char *rgb, unsigned char *out, int rows, int cols, int radius, int mode)
{
    if (radius % 2 == 0)
    {
        return;
    }

    #pragma omp parallel for
    for (int i = 0; i < rows * cols; i++)
    {
        int row = i / cols;
        int col = i % cols;

        int limit = ((radius - 1) / 2) - 1;
        if (mode == 0 || mode == -1)
        {
            for (int o = 3; o <= (mode == -1 ? 3 : radius - 2); o += 2)
            {
                int hybridLimit = ((o - 1) / 2) - 1;
                if (row > hybridLimit && row < rows - hybridLimit + 1 && col > hybridLimit && col < cols - hybridLimit + 1)
                {
                    for (int channel = 0; channel < 3; channel++)
                    {
                        int sum = 0;
                        for (int r = -hybridLimit - 1; r <= hybridLimit + 1; r++)
                        {
                            for (int c = -hybridLimit - 1; c <= hybridLimit + 1; c++)
                            {
                                sum += rgb[3 * ((row + r) * cols + (col + c)) + channel];
                            }
                        }
                        out[3 * i + channel] = sum / (o * o);
                    }
                }
            }
        }
        if (row > limit && row < rows - limit + 1 && col > limit && col < cols - limit + 1)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                int sum = 0;
                for (int r = -limit - 1; r <= limit + 1; r++)
                {
                    for (int c = -limit - 1; c <= limit + 1; c++)
                    {
                        sum += rgb[3 * ((row + r) * cols + (col + c)) + channel];
                    }
                }
                out[3 * i + channel] = sum / (radius * radius);
            }
        }
    }
}

int main()
{
    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    auto rgb = m_in.data;
    auto width = m_in.cols;
    auto height = m_in.rows;

    std::vector<unsigned char> grid(width * height * 3);
    cv::Mat m_out(m_in.rows, m_in.cols, CV_8UC3, grid.data());

    auto start = std::chrono::high_resolution_clock::now();

    blurBox(rgb, m_out.data, height, width, 3, hybrid_mode::hybrid_off);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " s\n";

    cv::imwrite("out.jpg", m_out);

    return 0;
}
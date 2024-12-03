#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

void blurBox(unsigned char *rgb, unsigned char *out, int rows, int cols, int radius)
{

    if (radius % 2 == 0)
    {
        return;
    }

    for (int i = 0; i < rows * cols; i++)
    {
        int row = i / cols;
        int col = i % cols;

        int limit = ((radius - 1) / 2);

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

int main()
{
    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    auto rgb = m_in.data;
    auto width = m_in.cols;
    auto height = m_in.rows;

    std::vector<unsigned char> grid(width * height * 3);
    cv::Mat m_out(m_in.rows, m_in.cols, CV_8UC3, grid.data());

    auto start = std::chrono::high_resolution_clock::now();
    blurBox(rgb, m_out.data, height, width, 31);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " s\n";

    cv::imwrite("out.jpg", m_out);

    return 0;
}
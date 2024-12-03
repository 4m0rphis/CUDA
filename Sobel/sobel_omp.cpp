#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>

void sobel(cv::Mat &in, cv::Mat &out)
{
    int width = in.cols;
    int height = in.rows;

    out.create(height, width, CV_8UC1);

    #pragma omp parallel for
    for (int j = 1; j < height - 1; ++j)
    {
        for (int i = 1; i < width - 1; ++i)
        {
            int h = in.at<uchar>(j - 1, i - 1) - in.at<uchar>(j - 1, i + 1) + 2 * in.at<uchar>(j, i - 1) - 2 * in.at<uchar>(j, i + 1) + in.at<uchar>(j + 1, i - 1) - in.at<uchar>(j + 1, i + 1);

            int v = in.at<uchar>(j - 1, i - 1) - in.at<uchar>(j + 1, i - 1) + 2 * in.at<uchar>(j - 1, i) - 2 * in.at<uchar>(j + 1, i) + in.at<uchar>(j - 1, i + 1) - in.at<uchar>(j + 1, i + 1);

            int res = h * h + v * v;
            res = res > 255 * 255 ? res = 255 * 255 : res;

            out.at<uchar>(j, i) = sqrt(res);
        }
    }
}

int main()
{

    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_GRAYSCALE);
    auto gray = m_in.data;
    auto width = m_in.cols;
    auto height = m_in.rows;

    std::vector<unsigned char> out_sobel(width * height);
    cv::Mat m_out(height, width, CV_8UC1, out_sobel.data());

    auto start = std::chrono::high_resolution_clock::now();

    sobel(m_in, m_out);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " s\n";

    cv::imwrite("out.jpg", m_out);

    return 0;
}

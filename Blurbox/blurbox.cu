#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

#define checkForCudaErr(value)                                    \
    {                                                             \
        cudaError_t err = value;                                  \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "Error %s at line %d in file %s\n",   \
                    cudaGetErrorString(err), __LINE__, __FILE__); \
            exit(-1);                                             \
        }                                                         \
    }

__global__ void blurBox(unsigned char *rgb, unsigned char *out, int rows, int cols, int radius)
{

    if (radius % 2 == 0)
    {
        return;
    }

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int row = i / cols;
    int col = i % cols;

    int limit = ((radius - 1) / 2) - 1;

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

int main()
{
    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    auto rgb = m_in.data;
    auto width = m_in.cols;
    auto height = m_in.rows;
    unsigned char *d_rgb, *d_out;

    checkForCudaErr(cudaMalloc(&d_rgb, width * height * 3));
    checkForCudaErr(cudaMalloc(&d_out, width * height * 3));

    checkForCudaErr(cudaMemcpy(d_rgb, rgb, width * height * 3, cudaMemcpyHostToDevice));

    int block_size = 2;
    int grid_size = (width * height + block_size - 1) / block_size;

    cudaEvent_t start, stop;

    checkForCudaErr(cudaEventCreate(&start));
    checkForCudaErr(cudaEventCreate(&stop));

    // Mesure du temps de calcul du kernel uniquement.
    checkForCudaErr(cudaEventRecord(start));

    blurBox<<<grid_size, block_size>>>(d_rgb, d_out, height, width, 11);

    checkForCudaErr(cudaEventRecord(stop));

    // cudaDeviceSynchronize();

    std::vector<unsigned char> out(width * height * 3);
    checkForCudaErr(cudaMemcpy(out.data(), d_out, width * height * 3, cudaMemcpyDeviceToHost));

    checkForCudaErr(cudaEventSynchronize(stop));
    float duration;
    checkForCudaErr(cudaEventElapsedTime(&duration, start, stop));
    std::cout << "time=" << duration << std::endl;

    checkForCudaErr(cudaEventDestroy(start));
    checkForCudaErr(cudaEventDestroy(stop));

    cv::Mat m_out(height, width, CV_8UC3, out.data());
    cv::imwrite("out.jpg", m_out);

    cudaFree(d_rgb);
    cudaFree(d_out);

    return 0;
}
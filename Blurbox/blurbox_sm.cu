#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <array>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE (32u)
#define RADIUS (31u)

using namespace std;

#define checkForCudaErr(value)                          \
    {                                                   \
        cudaError_t err = value;                        \
        if (err != cudaSuccess)                         \
        {                                               \
            fprintf(stderr, "Error %s at line %d\n",    \
                    cudaGetErrorString(err), __LINE__); \
            exit(-1);                                   \
        }                                               \
    }

__global__ void processImg(unsigned char *out, unsigned char *in, size_t pitch,
                           unsigned int width, unsigned int height)
{

    int tile_size = BLOCK_SIZE - 2 * (RADIUS / 2);
    int limit = (RADIUS - 1) / 2;

    int x_o = (tile_size * blockIdx.x) + threadIdx.x;
    int y_o = (tile_size * blockIdx.y) + threadIdx.y;

    int x_i = x_o - limit;
    int y_i = y_o - limit; // TODO: Possibly utilize         int limit = ((radius - 1) / 2) - 1; instead of hardcoding 2?!

    __shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];

    if ((x_i >= 0) && (x_i < width) && (y_i >= 0) && (y_i < height))
        sBuffer[threadIdx.y][threadIdx.x] = in[y_i * pitch + x_i];
    else
        sBuffer[threadIdx.y][threadIdx.x] = 0;

    __syncthreads();

    int sum = 0;
    if ((threadIdx.x < tile_size) && (threadIdx.y < tile_size))
    {
        for (int r = 0; r < RADIUS; ++r)
            for (int c = 0; c < RADIUS; ++c)
                sum += sBuffer[threadIdx.y + r][threadIdx.x + c];

        sum = sum / (RADIUS * RADIUS);
        if (x_o < width && y_o < height)
            out[y_o * width + x_o] = sum;
    }
}

int main()
{

    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);

    auto width = m_in.cols;
    auto height = m_in.rows;

    unsigned int size = width * height * sizeof(unsigned char);

    unsigned char *h_r = (unsigned char *)malloc(size * sizeof(unsigned char));
    unsigned char *h_g = (unsigned char *)malloc(size * sizeof(unsigned char));
    unsigned char *h_b = (unsigned char *)malloc(size * sizeof(unsigned char));

    // defining new memory for output image ( result of the process )

    unsigned char *h_r_n = (unsigned char *)malloc(size * sizeof(unsigned char));
    unsigned char *h_g_n = (unsigned char *)malloc(size * sizeof(unsigned char));
    unsigned char *h_b_n = (unsigned char *)malloc(size * sizeof(unsigned char));

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // OpenCV stores images in BGR format
            cv::Vec3b pixel = m_in.at<cv::Vec3b>(y, x);

            // Extract R, G, and B values and store them in respective arrays
            h_b[y * width + x] = pixel[0]; // Blue
            h_g[y * width + x] = pixel[1]; // Green
            h_r[y * width + x] = pixel[2]; // Red
        }
    }

    unsigned char *d_r_n = NULL;
    unsigned char *d_g_n = NULL;
    unsigned char *d_b_n = NULL;

    checkForCudaErr(cudaMalloc(&d_r_n, size));
    checkForCudaErr(cudaMalloc(&d_g_n, size));
    checkForCudaErr(cudaMalloc(&d_b_n, size));

    unsigned char *d_r = NULL;
    unsigned char *d_g = NULL;
    unsigned char *d_b = NULL;

    size_t pitch_r = 0;
    size_t pitch_g = 0;
    size_t pitch_b = 0;

    checkForCudaErr(cudaMallocPitch(&d_r, &pitch_r, width, height));
    checkForCudaErr(cudaMallocPitch(&d_g, &pitch_g, width, height));
    checkForCudaErr(cudaMallocPitch(&d_b, &pitch_b, width, height));

    checkForCudaErr(cudaMemcpy2D(d_r, pitch_r, h_r, width, width, height, cudaMemcpyHostToDevice));
    checkForCudaErr(cudaMemcpy2D(d_g, pitch_g, h_g, width, width, height, cudaMemcpyHostToDevice));
    checkForCudaErr(cudaMemcpy2D(d_b, pitch_b, h_b, width, width, height, cudaMemcpyHostToDevice));

    // int radius = 5;
    int tile_size = BLOCK_SIZE - 2 * (RADIUS / 2);
    cout << "Radius: " << RADIUS << endl;
    cout << "Tile size: " << tile_size << endl;

    // dim3 grid_size((width + tile_size - 1) / tile_size,
    //                (height + tile_size - 1) / tile_size);

    dim3 grid_size(ceil((float)width / tile_size), ceil((float)height / tile_size));

    cout << "Grid size: " << grid_size.x << " " << grid_size.y << endl;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    cout << "Block size: " << blockSize.x << " " << blockSize.y << endl;
    // auto start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;

    checkForCudaErr(cudaEventCreate(&start));
    checkForCudaErr(cudaEventCreate(&stop));

    // Mesure du temps de calcul du kernel uniquement.
    checkForCudaErr(cudaEventRecord(start));

    processImg<<<grid_size, blockSize>>>(d_r_n, d_r, pitch_r, width, height);
    processImg<<<grid_size, blockSize>>>(d_g_n, d_g, pitch_g, width, height);
    processImg<<<grid_size, blockSize>>>(d_b_n, d_b, pitch_b, width, height);

    checkForCudaErr(cudaEventRecord(stop));

    // cudaDeviceSynchronize();

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cout << "Elapsed time: " << diff.count() << " s\n";

    checkForCudaErr(cudaMemcpy(h_r_n, d_r_n, size, cudaMemcpyDeviceToHost));
    checkForCudaErr(cudaMemcpy(h_g_n, d_g_n, size, cudaMemcpyDeviceToHost));
    checkForCudaErr(cudaMemcpy(h_b_n, d_b_n, size, cudaMemcpyDeviceToHost));

    checkForCudaErr(cudaEventSynchronize(stop));
    float duration;
    checkForCudaErr(cudaEventElapsedTime(&duration, start, stop));
    std::cout << "time=" << duration << std::endl;

    checkForCudaErr(cudaEventDestroy(start));
    checkForCudaErr(cudaEventDestroy(stop));

    cv::Mat output_image(height, width, CV_8UC3);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            output_image.at<cv::Vec3b>(y, x)[0] = h_b_n[y * width + x]; // Blue
            output_image.at<cv::Vec3b>(y, x)[1] = h_g_n[y * width + x]; // Green
            output_image.at<cv::Vec3b>(y, x)[2] = h_r_n[y * width + x]; // Red
        }
    }

    cv::imwrite("out.jpg", output_image);

    free(h_r);
    free(h_g);
    free(h_b);

    free(h_r_n);
    free(h_g_n);
    free(h_b_n);

    checkForCudaErr(cudaFree(d_r));
    checkForCudaErr(cudaFree(d_g));
    checkForCudaErr(cudaFree(d_b));

    checkForCudaErr(cudaFree(d_r_n));
    checkForCudaErr(cudaFree(d_b_n));
    checkForCudaErr(cudaFree(d_g_n));
}
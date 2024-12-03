**Summary**

The project discusses the Box Blur filter, a spatial domain linear filter that averages neighboring pixel values to create a blurring effect. It details implementations in C++ and CUDA, emphasizing the impact of parallelization on performance. The filter requires an odd radius of at least 3, and the implementation leverages multiple kernels for RGB images in CUDA. The C++ versions include a single-threaded and an OpenMP parallelized variant, with the latter demonstrating significant speed improvements. The CUDA implementations show varying performance, particularly with shared memory, which can enhance speed but is limited by the hardware. The analysis concludes with insights into performance metrics based on image size and radius.

**Key Insights**
- Box Blur operates by averaging neighboring pixel values to create a blurring effect.
- The implementation in C++ shows marked performance gains when using OpenMP for parallel processing.
- CUDA implementations benefit from shared memory, significantly improving execution speed, particularly for large images.
- Performance metrics indicate that image resolution plays a critical role in processing time, with larger resolutions leading to longer computation times.
- The implementation of shared memory poses challenges, such as limiting maximum radius and potential bank conflicts.

**Frequently Asked Questions**

**What is Box Blur and how does it work?**

- Box Blur is a low-pass filter that averages the values of neighboring pixels to create a blurring effect. It requires an odd radius for its neighborhood calculations.

**How does parallelization affect the performance of Box Blur?**

- Parallelization, particularly using OpenMP in C++, can significantly reduce computation times compared to single-threaded implementations, especially for larger images.

**Why is shared memory important in CUDA implementations of Box Blur?**

- Shared memory allows for faster read/write operations compared to global memory, leading to quicker processing times. However, its limited capacity can also introduce challenges with larger images.

**What limitations does shared memory impose on Box Blur implementations?**

- The use of shared memory restricts the maximum radius to 31, as well as the block size, which can affect the overall performance and flexibility of the filter in processing larger images.
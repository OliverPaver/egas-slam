#include "../gsplat/rasterizer/bindings.h"
#include <cuda_runtime_api.h>
#include <iostream>

void checkResult(const torch::Tensor& means, const torch::Tensor& radii,
                 const torch::Tensor& depths, int cameras, bool packed,
                 int expected_intersections, bool expect_groups)
{
    at::optional<torch::Tensor> camera_ids, gaussian_ids;
    if (packed)
    {
        camera_ids = torch::zeros({means.size(0)}, means.options().dtype(torch::kInt64));
        gaussian_ids = torch::arange(means.size(0), means.options().dtype(torch::kInt64));
    }
    auto result = gsplat::isect_tiles_tensor_no_depth(means, radii, depths,
        camera_ids, gaussian_ids, cameras, 16, 4, 4, true, true);
    TORCH_CHECK(std::get<1>(result).numel() == expected_intersections, "Wrong tile coverage");
    TORCH_CHECK((std::get<3>(result).numel() > 0) == expect_groups, "Wrong Gaussian group coverage");
    const auto status = cudaDeviceSynchronize();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

int main()
{
    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    for (int count : {0, 1, 31})
    {
        checkResult(torch::zeros({1, count, 2}, options),
                    torch::zeros({1, count}, options.dtype(torch::kInt32)),
                    torch::ones({1, count}, options), 1, false, 0, false);
        std::cout << "PASS empty_or_zero_radius count=" << count << std::endl;
    }
    checkResult(torch::full({1, 1, 2}, 32.0, options),
                torch::ones({1, 1}, options.dtype(torch::kInt32)),
                torch::ones({1, 1}, options), 1, false, 4, true);
    std::cout << "PASS interior" << std::endl;
    checkResult(torch::zeros({1, 1, 2}, options),
                torch::ones({1, 1}, options.dtype(torch::kInt32)),
                torch::ones({1, 1}, options), 1, false, 1, true);
    std::cout << "PASS image_corner" << std::endl;
    checkResult(torch::full({1, 1, 2}, -32.0, options),
                torch::ones({1, 1}, options.dtype(torch::kInt32)),
                torch::ones({1, 1}, options), 1, false, 0, false);
    std::cout << "PASS outside_image" << std::endl;
    checkResult(torch::full({2, 1, 2}, 32.0, options),
                torch::ones({2, 1}, options.dtype(torch::kInt32)),
                torch::ones({2, 1}, options), 2, false, 8, true);
    std::cout << "PASS multiple_cameras" << std::endl;
    checkResult(torch::full({1, 2}, 32.0, options),
                torch::ones({1}, options.dtype(torch::kInt32)),
                torch::ones({1}, options), 1, true, 4, true);
    std::cout << "PASS packed" << std::endl;
    const auto vertices = torch::tensor({{1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}});
    const auto invalid = vertices.eq(0).all(1);
    TORCH_CHECK(!invalid[0].item<bool>() && invalid[1].item<bool>(), "Invalid vertex sentinel");
    std::cout << "PASS vertex_sentinel" << std::endl;
}

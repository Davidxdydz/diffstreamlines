#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline int2 clamp(int2 &v, int2 low, int2 high)
{
    v.x = min(max(v.x, low.x), high.x);
    v.y = min(max(v.y, low.y), high.y);
}

__device__ int2 grid_pos(float2 pos, int width)
{
    return {
        static_cast<int>(pos.x),
        static_cast<int>(pos.y)};
}

__device__ int grid_index(float2 pos, int width)
{
    int2 grid_pos = {
        static_cast<int>(pos.x),
        static_cast<int>(pos.y)};
    return (grid_pos.y * width + grid_pos.x) * 2;
}

__device__ float2 velocity_at_point(const float *__restrict__ velocity, float2 pos, int width)
{
    int i = grid_index(pos, width);
    return {
        velocity[i],
        velocity[i + 1]};
}

__global__ void streamlines_kernel_forward(
    const float *__restrict__ velocity,
    const float *__restrict__ start_locations,
    float *__restrict__ paths,
    int *__restrict__ path_lengths,
    float dt,
    int steps,
    int height,
    int width, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;
    float2 pos = {
        start_locations[id * 2],
        start_locations[id * 2 + 1]};
    paths[id * (steps + 1) * 2] = pos.x;
    paths[id * (steps + 1) * 2 + 1] = pos.y;
    for (int i = 0; i < steps; i++)
    {
        if (pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height)
        {
            path_lengths[id] = i + 1;
            return;
        }
        float2 v = velocity_at_point(velocity, pos, width);
        pos.x += v.x * dt;
        pos.y += v.y * dt;
        paths[id * (steps + 1) * 2 + i * 2 + 2] = pos.x;
        paths[id * (steps + 1) * 2 + i * 2 + 1 + 2] = pos.y;
    }
    path_lengths[id] = steps + 1;
}

std::tuple<torch::Tensor, torch::Tensor> streamlines_forward(torch::Tensor velocity, torch::Tensor start_locations, float dt, int steps)
{
    auto n = start_locations.size(0);
    int height = velocity.size(0);
    int width = velocity.size(1);

    velocity = velocity.contiguous();
    start_locations = start_locations.contiguous();
    auto paths = torch::empty({n, steps + 1, 2}, velocity.options());
    auto path_lengths = torch::empty({n}, torch::dtype(torch::kInt32).device(velocity.device()));
    const dim3 threads(256);
    const dim3 blocks((n + threads.x - 1) / threads.x);
    streamlines_kernel_forward<<<blocks, threads>>>(
        velocity.const_data_ptr<float>(),
        start_locations.const_data_ptr<float>(),
        paths.mutable_data_ptr<float>(),
        path_lengths.mutable_data_ptr<int>(),
        dt,
        steps,
        height,
        width,
        n);
    return {paths, path_lengths};
}

__global__ void streamlines_kernel_backward(
    const float *__restrict__ velocity,
    const float *__restrict__ start_locations,
    const float *__restrict__ grad_output,
    float *__restrict__ grad_velocity,
    int steps,
    int height,
    int width, int n, float dt)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;
    float2 pos = {
        start_locations[id * 2],
        start_locations[id * 2 + 1]};
    // grad_velocity[grid_index(pos, width)] = grad_output[id * (steps + 1) * 2];
    // grad_velocity[grid_index(pos, width) + 1] = grad_output[id * (steps + 1) * 2 + 1];
    atomicAdd(&grad_velocity[grid_index(pos, width)], grad_output[id * (steps + 1) * 2]);
    atomicAdd(&grad_velocity[grid_index(pos, width) + 1], grad_output[id * (steps + 1) * 2 + 1]);
    for (int i = 0; i < steps; i++)
    {
        if (pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height)
        {
            return;
        }
        float2 v = velocity_at_point(velocity, pos, width);
        pos.x += v.x * dt;
        pos.y += v.y * dt;
        // grad_velocity[grid_index(pos, width)] += grad_output[id * (steps + 1) * 2 + i * 2 + 2] * dt;
        // grad_velocity[grid_index(pos, width) + 1] += grad_output[id * (steps + 1) * 2 + i * 2 + 1 + 2] * dt;
        atomicAdd(&grad_velocity[grid_index(pos, width)], grad_output[id * (steps + 1) * 2 + i * 2 + 2] * dt);
        atomicAdd(&grad_velocity[grid_index(pos, width) + 1], grad_output[id * (steps + 1) * 2 + i * 2 + 1 + 2] * dt);
    }
}

torch::Tensor streamlines_backward(torch::Tensor grad_output, torch::Tensor velocity, torch::Tensor start_locations, float dt, int steps)
{
    int n = start_locations.size(0);
    int height = velocity.size(0);
    int width = velocity.size(1);

    velocity = velocity.contiguous();
    start_locations = start_locations.contiguous();
    auto grad_velocity = torch::zeros_like(velocity);
    const dim3 threads(256);
    const dim3 blocks((n + threads.x - 1) / threads.x);
    streamlines_kernel_backward<<<blocks, threads>>>(
        velocity.const_data_ptr<float>(),
        start_locations.const_data_ptr<float>(),
        grad_output.const_data_ptr<float>(),
        grad_velocity.mutable_data_ptr<float>(),
        steps,
        height,
        width,
        n,
        dt);
    return grad_velocity;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("streamlines_forward", &streamlines_forward);
    m.def("streamlines_backward", &streamlines_backward);
}
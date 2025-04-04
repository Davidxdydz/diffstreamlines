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
    const float *__restrict__ paths,
    const int *__restrict__ path_lengths,
    const float *__restrict__ grad_output,
    float *__restrict__ grad_velocity,
    int height,
    int width, int n, float dt, int steps)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;
    for (int i = 0; i < path_lengths[id]; i++)
    {
        float2 pos = {
            paths[id * (steps + 1) * 2 + i * 2],
            paths[id * (steps + 1) * 2 + i * 2 + 1]};
        atomicAdd(&grad_velocity[grid_index(pos, width)], grad_output[id * (steps + 1) * 2 + i * 2]);
        atomicAdd(&grad_velocity[grid_index(pos, width) + 1], grad_output[id * (steps + 1) * 2 + i * 2 + 1]);
    }
}

torch::Tensor streamlines_backward(torch::Tensor grad_output, torch::Tensor paths, torch::Tensor paths_lengths, int width, int height, float dt)
{
    int n = paths.size(0);
    int steps = paths.size(1) - 1;

    paths = paths.contiguous();
    paths_lengths = paths_lengths.contiguous();
    auto grad_velocity = torch::zeros({height, width, 2}, grad_output.options());
    const dim3 threads(256);
    const dim3 blocks((n + threads.x - 1) / threads.x);
    streamlines_kernel_backward<<<blocks, threads>>>(
        paths.const_data_ptr<float>(),
        paths_lengths.const_data_ptr<int>(),
        grad_output.const_data_ptr<float>(),
        grad_velocity.mutable_data_ptr<float>(),
        height,
        width,
        n,
        dt,
        steps);
    return grad_velocity;
}

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value)
{
    // https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__global__ void render_streamlines_kernel_forward(float *__restrict__ image, int *__restrict__ drawn_indices, const float *__restrict__ paths, const int *__restrict__ path_lengths, int n, int steps, int width, int height, const int box_radius)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;
    for (int i = 0; i < path_lengths[id]; i++)
    {
        int path_offset = id * (steps + 1) * 2;
        float2 pos = {
            paths[path_offset + i * 2],
            paths[path_offset + i * 2 + 1]};
        int2 gp = grid_pos(pos, width);
        for (int iy = gp.y - box_radius; iy <= gp.y + box_radius; iy++)
        {
            for (int ix = gp.x - box_radius; ix <= gp.x + box_radius; ix++)
            {
                if (ix < 0 || ix >= width || iy < 0 || iy >= height)
                    continue;
                float cx = (float)ix + 0.5 - pos.x;
                float cy = (float)iy + 0.5 - pos.y;
                float dist2 = (cx * cx + cy * cy);
                int index = (iy * width + ix);
                float value = exp(-0.5 * dist2 / (box_radius * box_radius));
                // max does not differentiate well, but this makes the gradient only depend on one path element
                float old = atomicMaxFloat(&image[index], value);
                // uh oh this is a race condition, lets see if it works
                if (value > old)
                {
                    drawn_indices[index] = id;
                }
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> render_streamlines_forward(torch::Tensor paths, torch::Tensor path_lenghts, int width, int height, const int box_radius)
{
    // TODO each point could be rendered in parallel, probably way faster for larger box_radius or larger n
    auto n = paths.size(0);
    int steps = paths.size(1);
    auto image = torch::zeros({height, width}, paths.options());
    auto drawn_indices = torch::full({height, width}, -1, torch::dtype(torch::kInt32).device(paths.device()));
    paths = paths.contiguous();
    path_lenghts = path_lenghts.contiguous();
    const dim3 threads(256);
    const dim3 blocks((n + threads.x - 1) / threads.x);
    render_streamlines_kernel_forward<<<blocks, threads>>>(
        image.mutable_data_ptr<float>(),
        drawn_indices.mutable_data_ptr<int>(),
        paths.const_data_ptr<float>(),
        path_lenghts.const_data_ptr<int>(),
        n,
        steps,
        width,
        height,
        box_radius);
    return {image, drawn_indices};
}

__global__ void render_streamlines_kernel_backward(float *__restrict__ paths_grad, const float *__restrict__ grad_output, const int *__restrict__ drawn_indices, const float *__restrict__ paths, int width, int height, const int box_radius)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height)
        return;
    int index = (iy * width + ix);
    int path_id = drawn_indices[index];
    if (path_id == -1)
        return;
    float2 pos = {
        paths[path_id],
        paths[path_id + 1]};
    int2 gp = grid_pos(pos, width);
    float cx = (float)ix + 0.5 - pos.x;
    float cy = (float)iy + 0.5 - pos.y;
    float dist2 = (cx * cx + cy * cy);
    float b2 = box_radius * box_radius;
    float value = exp(-0.5 * dist2 / b2);
    float grad = 2 * sqrt(dist2) / b2 * value * 2;
    atomicAdd(&paths_grad[path_id], grad * grad_output[index] * cx);
    atomicAdd(&paths_grad[path_id + 1], grad * grad_output[index] * cy);
}

torch::Tensor render_streamlines_backward(torch::Tensor grad_output, torch::Tensor paths, torch::Tensor drawn_indices, int width, int height, const int box_radius)
{
    auto paths_grad = torch::zeros_like(paths);
    paths = paths.contiguous();
    drawn_indices = drawn_indices.contiguous();
    const dim3 threads(16, 16);
    const dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    render_streamlines_kernel_backward<<<blocks, threads>>>(
        paths_grad.mutable_data_ptr<float>(),
        grad_output.const_data_ptr<float>(),
        drawn_indices.const_data_ptr<int>(),
        paths.const_data_ptr<float>(),
        width,
        height,
        box_radius);
    return paths_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("streamlines_forward", &streamlines_forward);
    m.def("streamlines_backward", &streamlines_backward);
    m.def("render_streamlines_forward", &render_streamlines_forward);
    m.def("render_streamlines_backward", &render_streamlines_backward);
}
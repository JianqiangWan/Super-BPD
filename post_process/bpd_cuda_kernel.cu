#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "disjoint.cuh"

#define PI 3.14159265
// 2** 31
__device__ const uint32_t REPLUSIVE_INIT = 2147483648;
__device__ const int direction[8][2]={1,0, 1,1, 0,1, -1,1, -1,0, -1,-1, 0,-1, 1,-1};

#define CUDA_1D_KERNEL_LOOP(index, nthreads)                            \
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; \
       index += blockDim.x * gridDim.x)


__global__ void find_parents(
    const int nthreads,
    const int height,
    const int width,
    const float theta_a,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_angles,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> parents,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> roots) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int curr_w = index % width;
    int curr_h = index / width;

    float curr_angle = input_angles[curr_h][curr_w];
    
    int pos=(curr_angle + PI/8)/(PI/4);
    if(pos >= 8) pos-=8;
    
    int next_h = curr_h +  direction[pos][0];
    int next_w = curr_w +  direction[pos][1];

    if (next_h >= height || next_h < 0 || next_w >= width || next_w < 0) {
        parents[0][curr_h][curr_w] = curr_h;
        parents[1][curr_h][curr_w] = curr_w;
        roots[curr_h][curr_w] = 1;
        return;
    }

    float next_angle = input_angles[next_h][next_w];
    float angle_diff = abs(curr_angle - next_angle);
    angle_diff = min(angle_diff, 2*PI - angle_diff);

    if (angle_diff > theta_a * PI / 180) {
        parents[0][curr_h][curr_w] = curr_h;
        parents[1][curr_h][curr_w] = curr_w;
        roots[curr_h][curr_w] = 1;
        return;
    }
    
    parents[0][curr_h][curr_w] = next_h;
    parents[1][curr_h][curr_w] = next_w;

    }
}

__global__ void get_super_BPDs_step1(
    const int nthreads,
    const int height,
    const int width,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> parents,
    int* super_BPDs) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int curr_w = index % width;
    int curr_h = index / width;

    int next_h = parents[0][curr_h][curr_w];
    int next_w = parents[1][curr_h][curr_w];
    int next_index = next_h*width + next_w;
    
    UNION(super_BPDs, index, next_index);
    }
}

__global__ void get_super_BPDs_step2(
    const int nthreads,
    int* super_BPDs) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    super_BPDs[index] = FIND(super_BPDs, index) + 1;
    }
}

__global__ void merge_nearby_root_pixels(
    const int nthreads,
    const int height,
    const int width,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> roots,
    int* super_BPDs) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int curr_w = index % width;
    int curr_h = index / width;
    
    if (!roots[curr_h][curr_w]) return;

    for (int delta_h=0; delta_h<=min(3, height-1-curr_h); delta_h++) {
        for (int delta_w=-min(3, curr_w); delta_w<=min(3, width-1-curr_w); delta_w++) {
            int next_h = curr_h + delta_h;
            int next_w = curr_w + delta_w;
            if (roots[next_h][next_w]) {
                int next_index = next_h*width + next_w;
                UNION(super_BPDs, index, next_index);
            }
        }
    }
    }
}

__global__ void find_bnd_angle_diff(
    const int nthreads,
    const int height,
    const int width,
    const int num_superpixels,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_angles,
    int* super_BPDs,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> parents,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> unique_super_BPDs_inverse,
    float* bnd_angle_diff,
    int* bnd_pair_nums) {
    
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int curr_w = index % width;
    int curr_h = index / width;
    int curr_index = curr_h*width + curr_w;
    
    // right and bottom point
    int delta_h[2] = {0,1}; 
    int delta_w[2] = {1,0};
    
    for (int i=0; i<2; i++) {
        int next_h = curr_h + delta_h[i];
        int next_w = curr_w + delta_w[i];

        if (next_w >= width || next_h >= height) continue;

        int next_index = next_h*width + next_w;

        if (super_BPDs[curr_index] != super_BPDs[next_index]) {
            int curr_position = unique_super_BPDs_inverse[curr_h][curr_w];
            int next_position = unique_super_BPDs_inverse[next_h][next_w];
            int min_position = min(curr_position, next_position);
            int max_position = max(curr_position, next_position);
            atomicAdd(bnd_pair_nums + min_position*num_superpixels + max_position, 1);
            // forward 3 steps respectively, then calculate angle diff
            int steps = 3;
            while (steps--) {
                int curr_parent_h = parents[0][curr_h][curr_w];
                int curr_parent_w = parents[1][curr_h][curr_w];
                curr_h = curr_parent_h;
                curr_w = curr_parent_w;
    
                int next_parent_h = parents[0][next_h][next_w];
                int next_parent_w = parents[1][next_h][next_w];
                next_h = next_parent_h;
                next_w = next_parent_w;
            }
            float curr_angle = input_angles[curr_h][curr_w];
            float next_angle = input_angles[next_h][next_w];
            float angle_diff = abs(curr_angle - next_angle);
            angle_diff = min(angle_diff, 2*PI - angle_diff);
            atomicAdd(bnd_angle_diff + min_position*num_superpixels + max_position, angle_diff);
        }
    }
    }
}

__global__ void classify_edges(
    const int nthreads,
    const int num_superpixels,
    const int nums,
    const float S_o,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bnd_angle_diff,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> bnd_pair_nums,
    torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> select_matrix,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> edge_h,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> edge_w,
    int* replusive_matrix) {
    
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int curr_w = index % num_superpixels;
        int curr_h = index / num_superpixels;

        if (bnd_pair_nums[curr_h][curr_w] == 0) return;

        float avg_angle_diff = bnd_angle_diff[curr_h][curr_w] / bnd_pair_nums[curr_h][curr_w];
        bnd_angle_diff[curr_h][curr_w] = avg_angle_diff;

        if (avg_angle_diff > PI - S_o * PI / 180) {

            int inter_h = curr_w / 32;
            int inter_w = curr_w % 32;

            atomicOr(replusive_matrix + curr_h*nums + inter_h, REPLUSIVE_INIT >> inter_w);

            return;
        }

        select_matrix[curr_h][curr_w] = 1;
        edge_h[curr_h][curr_w] = curr_h;
        edge_w[curr_h][curr_w] = curr_w;
    }
}


__global__ void final_step(
    const int nthreads,
    torch::PackedTensorAccessor32<uint8_t,1,torch::RestrictPtrTraits> connect_marks,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_h,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_w,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> unique_super_BPDs,
    int* super_BPDs) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    
        if (connect_marks[index]) {
            int index_h = unique_super_BPDs[edge_h[index]] - 1;
            int index_w = unique_super_BPDs[edge_w[index]] - 1;
            UNION(super_BPDs, index_h, index_w);
        }
    }   
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, \
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> \
bpd_cuda(const torch::Tensor input_angles, const int height, const int width, const float theta_a, const float S_o) {

const int kThreadsPerBlock = 1024;
const int blocks = (height*width + kThreadsPerBlock - 1) / kThreadsPerBlock;

torch::Tensor parents = torch::zeros({2, height, width}, torch::CUDA(torch::kInt32));
torch::Tensor roots = torch::zeros({height, width}, torch::CUDA(torch::kInt32));

// get parents and roots
find_parents<<<blocks, kThreadsPerBlock>>>(
    height*width,
    height,
    width,
    theta_a,
    input_angles.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    parents.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
    roots.packed_accessor32<int,2,torch::RestrictPtrTraits>()
);

// get super-BPDs, index from 0 ~ height*width - 1, init label from 1 ~ height*width
torch::Tensor super_BPDs = torch::arange(1, height*width + 1, torch::CUDA(torch::kInt32));
get_super_BPDs_step1<<<blocks, kThreadsPerBlock>>>(
    height*width,
    height,
    width,
    parents.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
    super_BPDs.contiguous().data_ptr<int>()
);

get_super_BPDs_step2<<<blocks, kThreadsPerBlock>>>(
    height*width,
    super_BPDs.contiguous().data_ptr<int>()
);
auto super_BPDs_before_dilation = super_BPDs.clone();
super_BPDs_before_dilation = super_BPDs_before_dilation.reshape({height, width});

// merge nearby root pixels
merge_nearby_root_pixels<<<blocks, kThreadsPerBlock>>>(
    height*width,
    height,
    width,
    roots.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
    super_BPDs.contiguous().data_ptr<int>()
);

get_super_BPDs_step2<<<blocks, kThreadsPerBlock>>>(
    height*width,
    super_BPDs.contiguous().data_ptr<int>()
);
auto super_BPDs_after_dilation = super_BPDs.clone();
super_BPDs_after_dilation = super_BPDs_after_dilation.reshape({height, width});

// construct RAG
auto unique_results = torch::_unique2(super_BPDs, true, true, true);
auto unique_super_BPDs = std::get<0>(unique_results);
auto unique_super_BPDs_inverse = std::get<1>(unique_results);
unique_super_BPDs_inverse = unique_super_BPDs_inverse.to(torch::kInt32);
unique_super_BPDs_inverse = unique_super_BPDs_inverse.reshape({height, width});
auto unique_super_BPDs_counts = std::get<2>(unique_results);
unique_super_BPDs_counts = unique_super_BPDs_counts.to(torch::kInt32);

int num_superpixels = unique_super_BPDs.numel();
torch::Tensor bnd_angle_diff = torch::zeros({num_superpixels, num_superpixels}, torch::CUDA(torch::kFloat32));
torch::Tensor bnd_pair_nums = torch::zeros({num_superpixels, num_superpixels}, torch::CUDA(torch::kInt32));

find_bnd_angle_diff<<<blocks, kThreadsPerBlock>>>(
    height*width,
    height,
    width,
    num_superpixels,
    input_angles.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    super_BPDs.contiguous().data_ptr<int>(),
    parents.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
    unique_super_BPDs_inverse.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
    bnd_angle_diff.contiguous().data_ptr<float>(),
    bnd_pair_nums.contiguous().data_ptr<int>()
);

// classify edges (replusive, large, small, tiny)
torch::Tensor select_matrix = torch::zeros({num_superpixels, num_superpixels}, torch::CUDA(torch::kBool));
torch::Tensor edge_h = torch::zeros({num_superpixels, num_superpixels}, torch::CUDA(torch::kInt32));
torch::Tensor edge_w = torch::zeros({num_superpixels, num_superpixels}, torch::CUDA(torch::kInt32));

const int nums = (num_superpixels + 32 -1) / 32;
torch::Tensor replusive_matrix = torch::zeros({num_superpixels, nums}, torch::CUDA(torch::kInt32));

const int blocks2 = (num_superpixels*num_superpixels + kThreadsPerBlock - 1) / kThreadsPerBlock;

classify_edges<<<blocks2, kThreadsPerBlock>>>(
    num_superpixels*num_superpixels,
    num_superpixels,
    nums,
    S_o,
    bnd_angle_diff.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    bnd_pair_nums.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
    select_matrix.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
    edge_h.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
    edge_w.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
    replusive_matrix.contiguous().data_ptr<int>()
);

bnd_angle_diff = bnd_angle_diff.masked_select(select_matrix);
edge_h = edge_h.masked_select(select_matrix);
edge_w = edge_w.masked_select(select_matrix);

// diff small to large, sim large to small
auto sort_index = bnd_angle_diff.argsort();

auto sorted_bnd_angle_diff = bnd_angle_diff.index({sort_index});
auto sorted_edge_h = edge_h.index({sort_index});
auto sorted_edge_w = edge_w.index({sort_index});

// connect edges
sorted_bnd_angle_diff = sorted_bnd_angle_diff.to(torch::kCPU);
sorted_edge_h = sorted_edge_h.to(torch::kCPU);
sorted_edge_w = sorted_edge_w.to(torch::kCPU);
replusive_matrix = replusive_matrix.to(torch::kCPU);

unique_super_BPDs_counts = unique_super_BPDs_counts.to(torch::kCPU);
unique_super_BPDs = unique_super_BPDs.to(torch::kCPU);

return std::make_tuple(unique_super_BPDs_counts, sorted_edge_h, \
sorted_edge_w, sorted_bnd_angle_diff, replusive_matrix, unique_super_BPDs, \
roots, super_BPDs_before_dilation, super_BPDs_after_dilation, super_BPDs);
}

torch::Tensor bpd_cuda_final_step(const int height, const int width, torch::Tensor connect_marks, torch::Tensor edge_h, \
torch::Tensor edge_w, torch::Tensor unique_super_BPDs, torch::Tensor super_BPDs) {

connect_marks = connect_marks.to(torch::kCUDA);
edge_h = edge_h.to(torch::kCUDA);
edge_w = edge_w.to(torch::kCUDA);
unique_super_BPDs = unique_super_BPDs.to(torch::kCUDA);
super_BPDs = super_BPDs.to(torch::kCUDA);

const int num_edges = edge_h.numel();

const int kThreadsPerBlock = 1024;
const int blocks = (num_edges + kThreadsPerBlock - 1) / kThreadsPerBlock;

final_step<<<blocks, kThreadsPerBlock>>>(
    num_edges,
    connect_marks.packed_accessor32<uint8_t,1,torch::RestrictPtrTraits>(),
    edge_h.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
    edge_w.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
    unique_super_BPDs.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
    super_BPDs.contiguous().data_ptr<int>()
);

const int blocks2 = (height*width + kThreadsPerBlock - 1) / kThreadsPerBlock;

get_super_BPDs_step2<<<blocks2, kThreadsPerBlock>>>(
    height*width,
    super_BPDs.contiguous().data_ptr<int>()
);

super_BPDs = super_BPDs.reshape({height, width});

return super_BPDs;
}




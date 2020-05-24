#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define PI 3.14159265
const uint32_t REPLUSIVE_INIT = 2147483648;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,\
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> \
bpd_cuda(const torch::Tensor input_angles, const int height, const int width, const float theta_a, const float S_o);

torch::Tensor bpd_cuda_final_step(const int height, const int width, torch::Tensor connect_marks, torch::Tensor edge_h, \
                                  torch::Tensor edge_w, torch::Tensor unique_super_BPDs, torch::Tensor super_BPDs);


int find(int* parent, int x)
{
    if(parent[x]==x) return x;
    return parent[x]=find(parent, parent[x]);
}

torch::Tensor bpd_cpu_forward(
    torch::Tensor unique_super_BPDs_counts,
    torch::Tensor edge_h,
    torch::Tensor edge_w,
    torch::Tensor bnd_angle_diff,
    torch::Tensor replusive_matrix,
    const float theta_l,
    const float theta_s) {

    int num_edges = edge_h.numel();                     
    int num_superpixels = replusive_matrix.size(0);     
    int nums_32 = replusive_matrix.size(1); 

    int parent[num_superpixels];
    for (int i=0; i<num_superpixels; i++) parent[i] = i; 

    torch::Tensor connect_marks = torch::zeros({num_edges}, torch::kByte);

    int* edge_h_ptr = edge_h.data_ptr<int>();
    int* edge_w_ptr = edge_w.data_ptr<int>();
    int* unique_super_BPDs_counts_ptr = unique_super_BPDs_counts.data_ptr<int>();
    int* replusive_matrix_ptr = replusive_matrix.contiguous().data_ptr<int>();
    float* bnd_angle_diff_ptr = bnd_angle_diff.contiguous().data_ptr<float>();
    uint8_t* connect_marks_ptr = connect_marks.data_ptr<uint8_t>();

    for (int i=0; i<num_edges; i++) {
        
        int index_h = find(parent, edge_h_ptr[i]);
        int index_w = find(parent, edge_w_ptr[i]);

        int area_h = unique_super_BPDs_counts_ptr[index_h];
        int area_w = unique_super_BPDs_counts_ptr[index_w];
        int min_area = std::min(area_h, area_w);
        float thresh = 0;

        int inter_h = index_w / 32;
        int inter_w = index_w % 32;
        int value = REPLUSIVE_INIT >> inter_w;

        if ((min_area > 250) && !(replusive_matrix_ptr[index_h*nums_32 + inter_h] & value)) {
            if (min_area > 1500) thresh = PI - theta_l*PI/180;
            else thresh = PI - theta_s*PI/180;

            if (bnd_angle_diff_ptr[i] < thresh) {
                connect_marks_ptr[i] = 1;
                parent[index_h] = index_w;
                // update area and replusive matrix
                for (int j=0; j<nums_32; j++) {

                    replusive_matrix_ptr[index_w*nums_32 + j] |= replusive_matrix_ptr[index_h*nums_32 + j];

                }
                unique_super_BPDs_counts_ptr[index_w] = area_h + area_w;
            }
        }
    }
    // tiny region
    for (int i=0; i<num_edges; i++) {

        if (connect_marks_ptr[i]) continue;
        
        // find root point
        int index_h = find(parent, edge_h_ptr[i]);
        int index_w = find(parent, edge_w_ptr[i]);

        int area_h = unique_super_BPDs_counts_ptr[index_h];
        int area_w = unique_super_BPDs_counts_ptr[index_w];
        int min_area = std::min(area_h, area_w);

        int inter_h = index_w / 32;
        int inter_w = index_w % 32;
        int value = REPLUSIVE_INIT >> inter_w;

        if ((min_area <= 250) && !(replusive_matrix_ptr[index_h*nums_32 + inter_h] & value)) {
                connect_marks_ptr[i] = 1;
                parent[index_h] = index_w;
                // update area and replusive matrix
                for (int j=0; j<nums_32; j++) {

                    replusive_matrix_ptr[index_w*nums_32 + j] |= replusive_matrix_ptr[index_h*nums_32 + j];

                }
                unique_super_BPDs_counts_ptr[index_w] = area_h + area_w;
        }
    }

    return connect_marks;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> \
forward(
    torch::Tensor input_angles,
    const int height,
    const int width,
    const float theta_a,
    const float theta_l,
    const float theta_s,
    const float S_o) {
    
    CHECK_INPUT(input_angles);
    auto cuda_return_tuple_results = bpd_cuda(input_angles, height, width, theta_a, S_o);
    auto unique_super_BPDs_counts = std::get<0>(cuda_return_tuple_results);
    auto edge_h = std::get<1>(cuda_return_tuple_results);
    auto edge_w = std::get<2>(cuda_return_tuple_results);
    auto bnd_angle_diff = std::get<3>(cuda_return_tuple_results);
    auto replusive_matrix = std::get<4>(cuda_return_tuple_results);
    auto unique_super_BPDs = std::get<5>(cuda_return_tuple_results);

    auto root_points = std::get<6>(cuda_return_tuple_results);
    auto super_BPDs_before_dilation = std::get<7>(cuda_return_tuple_results);
    auto super_BPDs_after_dilation = std::get<8>(cuda_return_tuple_results);
    auto super_BPDs = std::get<9>(cuda_return_tuple_results);

    auto connect_marks = bpd_cpu_forward(unique_super_BPDs_counts, edge_h, \
    edge_w, bnd_angle_diff, replusive_matrix, theta_l, theta_s);

    auto final_result = bpd_cuda_final_step(height, width, connect_marks, edge_h, edge_w, unique_super_BPDs, super_BPDs);

  return std::make_tuple(root_points, super_BPDs_before_dilation, super_BPDs_after_dilation, final_result);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "super-BPD post-process forward (CUDA)");
}


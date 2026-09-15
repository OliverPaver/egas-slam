#include "file_utils.h"

#include <cstdlib>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;

namespace
{
fs::path normalizedAbsolutePath(const fs::path &path)
{
    return fs::weakly_canonical(fs::absolute(path));
}

bool pathContains(const fs::path &parent, const fs::path &child)
{
    const fs::path relative = child.lexically_relative(parent);
    if (relative.empty())
        return false;
    const auto first = relative.begin();
    return first == relative.end() || *first != "..";
}

unsigned long long getCudaMemoryUsage()
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t result = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (result != cudaSuccess)
    {
        std::cerr << "Failed to query CUDA memory: " << cudaGetErrorString(result) << "\n";
        return ULLONG_MAX;
    }
    return static_cast<unsigned long long>((total_bytes - free_bytes) / (1024 * 1024));
}

void validateWorkspacePath(const fs::path &workspace,
                           const fs::path &config_filename,
                           const YAML::Node &config)
{
    if (workspace.empty())
        throw std::runtime_error("workspace_dir must not be empty");

    const fs::path resolved_workspace = normalizedAbsolutePath(workspace);
    const fs::path resolved_root = resolved_workspace.root_path();
    if (resolved_workspace == resolved_root)
        throw std::runtime_error("Refusing to clear filesystem root as workspace_dir");
    if (fs::exists(resolved_workspace) && !fs::is_directory(resolved_workspace))
        throw std::runtime_error("workspace_dir exists but is not a directory: " + resolved_workspace.string());

    const fs::path resolved_config = normalizedAbsolutePath(config_filename);
    if (pathContains(resolved_workspace, resolved_config))
        throw std::runtime_error(
            "Refusing to clear workspace_dir because it contains the active config: " +
            resolved_workspace.string());

    const fs::path resolved_cwd = normalizedAbsolutePath(fs::current_path());
    if (pathContains(resolved_workspace, resolved_cwd) ||
        pathContains(resolved_cwd, resolved_workspace))
        throw std::runtime_error(
            "Refusing to clear workspace_dir because it overlaps the current/project directory: " +
            resolved_workspace.string());

    if (const char *home_value = std::getenv("HOME"))
    {
        const fs::path resolved_home = normalizedAbsolutePath(home_value);
        if (resolved_workspace == resolved_home || pathContains(resolved_workspace, resolved_home))
            throw std::runtime_error("Refusing to clear the user home directory or its parent as workspace_dir");
    }

    if (config["READER"] && config["READER"]["input_dir"])
    {
        const fs::path input_dir = config["READER"]["input_dir"].as<std::string>();
        if (!input_dir.empty())
        {
            const fs::path resolved_input = normalizedAbsolutePath(input_dir);
            if (pathContains(resolved_workspace, resolved_input) ||
                pathContains(resolved_input, resolved_workspace))
                throw std::runtime_error(
                    "Refusing workspace_dir that overlaps READER.input_dir: " +
                    resolved_workspace.string());
        }
    }
}
}

///////////////// EIGEN LOAD AND SAVE /////////////////

MatrixX readMatrixFromTXT(const std::string &filename)
{
    std::ifstream infile(filename);
    std::vector<std::vector<double>> matrix;

    if (!infile.is_open())
    {
        throw std::runtime_error("Unable to open matrix file: " + filename);
    }

    std::string line;
    while (std::getline(infile, line))
    {
        std::vector<double> row;
        std::stringstream ss(line);
        double value;

        while (ss >> value)
        {
            if (!std::isfinite(value))
                throw std::runtime_error("Non-finite matrix value: " + filename);
            row.push_back(value);
        }
        if (!ss.eof())
            throw std::runtime_error("Malformed matrix value: " + filename);

        if (!row.empty())
        {
            matrix.push_back(row);
        }
    }

    if (infile.bad())
        throw std::runtime_error("Failed while reading matrix file: " + filename);
    infile.close();

    if (matrix.empty())
    {
        throw std::runtime_error("Empty matrix file: " + filename);
    }

    int rows = matrix.size();
    int cols = matrix[0].size();

    Eigen::MatrixXf eigenMatrix(rows, cols);

    for (int i = 0; i < rows; ++i)
    {
        if (static_cast<int>(matrix[i].size()) != cols)
        {
            throw std::runtime_error("Inconsistent matrix columns: " + filename);
        }
        for (int j = 0; j < cols; ++j)
        {
            eigenMatrix(i, j) = matrix[i][j];
        }
    }

    return eigenMatrix;
}

torch::Tensor matrixToTensor(const MatrixX &eigenMatrix)
{
    // 获取 Eigen 矩阵的尺寸
    int rows = eigenMatrix.rows();
    int cols = eigenMatrix.cols();
    // 创建一个 torch::Tensor，使用 Eigen 矩阵的数据
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor;
    if (eigenMatrix.IsRowMajor)
    {
        // 如果 Eigen 矩阵是行优先的，直接创建 tensor
        tensor = torch::from_blob(
                     const_cast<float *>(eigenMatrix.data()),
                     {rows, cols},
                     options)
                     .clone();
    }
    else
    {
        // 如果 Eigen 矩阵是列优先的，创建 tensor 后进行转置
        tensor = torch::from_blob(
                     const_cast<float *>(eigenMatrix.data()),
                     {cols, rows},
                     options)
                     .clone()
                     .transpose(0, 1);
    }
    return tensor;
}

////////////////////////////////////////////////////////////////

bool createDirectory(const std::string &path, bool overwrite = false)
{
    try
    {
        fs::path dirPath(path);

        // Check if the path already exists
        if (fs::exists(dirPath))
        {
            if (overwrite)
            {
                // If overwrite is true, remove the existing directory and all its contents
                fs::remove_all(dirPath);
            }
            else
            {
                // If overwrite is false and the directory exists, return false
                std::cout << "Directory already exists, no action taken: " << path << std::endl;
                return false;
            }
        }

        // Create the directory
        if (fs::create_directories(dirPath))
        {
            std::cout << "Directory created successfully: " << path << std::endl;
            return true;
        }
        else
        {
            std::cout << "Failed to create directory: " << path << std::endl;
            return false;
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return false;
    }
}

std::string idToFilename(int frame_id)
{
    std::ostringstream filename;
    filename << std::setw(6) << std::setfill('0') << frame_id;
    return filename.str();
}

void createWorkSpace(const std::string &config_filename)
{
    YAML::Node config = YAML::LoadFile(config_filename);
    const fs::path workspace_dir = config["workspace_dir"].as<std::string>();
    validateWorkspacePath(workspace_dir, config_filename, config);
    if (!createDirectory(workspace_dir.string(), true))
        throw std::runtime_error("Could not create workspace_dir: " + workspace_dir.string());
    std::ofstream marker(workspace_dir / ".gps_slam_workspace");
    marker << "Managed by GPS-SLAM. This directory may be cleared at run startup.\n";
    marker.close();
    if (!marker)
        throw std::runtime_error("Could not write workspace ownership marker: " + workspace_dir.string());
    // copy config file
    fs::path source = fs::path(config_filename);
    fs::path target = workspace_dir / source.filename();
    fs::copy_file(source, target, fs::copy_options::overwrite_existing);
}

void saveCfgArgs(const std::string &file_name, int maxSH)
{
    std::ofstream outfile(file_name);
    outfile
        << "Namespace(data_device='cuda', eval=False, images='images', model_path='', resolution=-1, sh_degree=" << maxSH << ", source_path='', white_background=False)"
        << std::endl;
    outfile.close();
}

// Function to get the memory usage of the GPU
unsigned long long getGPUMemoryUsage(int gpu_id)
{
#ifdef GPS_SLAM_HAS_NVML
    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlMemory_t memory;

    // Initialize NVML library
    result = nvmlInit();
    if (NVML_SUCCESS != result)
    {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result)
                  << "; falling back to cudaMemGetInfo\n";
        return getCudaMemoryUsage();
    }

    // Get the GPU handle
    result = nvmlDeviceGetHandleByIndex(gpu_id, &device);
    if (NVML_SUCCESS != result)
    {
        std::cerr << "Failed to get handle for device " << std::to_string(gpu_id) << ": "
                  << nvmlErrorString(result) << "; falling back to cudaMemGetInfo\n";
        nvmlShutdown();
        return getCudaMemoryUsage();
    }

    // Get memory information
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (NVML_SUCCESS != result)
    {
        std::cerr << "Failed to get memory info: " << nvmlErrorString(result)
                  << "; falling back to cudaMemGetInfo\n";
        nvmlShutdown();
        return getCudaMemoryUsage();
    }

    auto used_gpu_memory = memory.used / (1024 * 1024);

    // Shutdown NVML library
    nvmlShutdown();

    return used_gpu_memory;
#else
    (void)gpu_id;
    return getCudaMemoryUsage();
#endif
}

double calculateTimeInterval(struct timespec start, struct timespec end)
{
    // 计算秒和纳秒的时间差
    time_t seconds_diff = end.tv_sec - start.tv_sec;
    long nanoseconds_diff = end.tv_nsec - start.tv_nsec;

    // 将时间差转换为毫秒
    double interval_ms = (seconds_diff * 1000.0) + (nanoseconds_diff / 1000000.0);

    return interval_ms;
}

void saveTensorTXT(const torch::Tensor &tensor, const std::string &filename)
{
// 检查输入是否为二维张量
    if (tensor.dim() != 2 || tensor.size(0) != 4 || tensor.size(1) != 4) {
        throw std::invalid_argument("输入必须是4x4的二维张量");
    }

    // 确保张量内存连续（防止非连续内存导致访问错误）
    auto tensor_contig = tensor.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (!torch::isfinite(tensor_contig).all().item<bool>())
        throw std::runtime_error("Cannot save a non-finite pose: " + filename);

    // 打开文件流
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    // 设置输出流的浮点精度为固定6位小数
    file << std::setprecision(std::numeric_limits<float>::max_digits10);

    // 获取数据指针（假设数据类型为float）
    auto data_ptr = tensor_contig.data_ptr<float>();

    // 按行写入数据
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            // 计算一维索引并写入值
            file << data_ptr[row * 4 + col];
            if (col < 3) file << " "; // 列间加空格
        }
        file << "\n"; // 每行结束后换行
    }

    file.close();
    if (!file)
        throw std::runtime_error("Failed while writing pose: " + filename);
}

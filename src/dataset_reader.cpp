#include <tinyply.h>

#include "dataset_reader.h"
#include "cv_utils.h"
#include "file_utils.h"
#include "tensor_math.h"
#include <cmath>
#include <set>
#include <map>

namespace fs = std::filesystem;
using namespace torch::indexing;
using json = nlohmann::json;

void Points::readPly(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    tinyply::PlyFile ply_file;
    ply_file.parse_header(file);

    std::shared_ptr<tinyply::PlyData> raw_xyzs, raw_colors, raw_normals;

    try
    {
        raw_xyzs = ply_file.request_properties_from_element("vertex", {"x", "y", "z"});
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Failed to read vertex positions: " + std::string(e.what()));
    }

    try
    {
        raw_colors = ply_file.request_properties_from_element("vertex", {"red", "green", "blue"});
    }
    catch (const std::exception &)
    {
        // Color data not available, ignore
    }

    try
    {
        raw_normals = ply_file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
    }
    catch (const std::exception &)
    {
        // Normal data not available, ignore
    }

    ply_file.read(file);
    std::cout << "load points num: " << raw_xyzs->count << std::endl;
    // Convert raw_xyzs to torch tensor
    xyz = torch::from_blob(raw_xyzs->buffer.get(),
                           {static_cast<long>(raw_xyzs->count), 3}, torch::kFloat)
              .clone();

    // Convert raw_colors to torch tensor if available
    if (raw_colors)
    {
        rgb = torch::from_blob(raw_colors->buffer.get(),
                               {static_cast<long>(raw_colors->count), 3}, torch::kUInt8)
                  .to(torch::kFloat32)
                  .div(255.0)
                  .clone();
    }
    else
    {
        rgb = torch::Tensor();
    }

    // Convert raw_normals to torch tensor if available
    if (raw_normals)
    {
        normal = torch::from_blob(raw_normals->buffer.get(),
                                  {static_cast<long>(raw_normals->count), 3}, torch::kFloat)
                     .clone();
    }
    else
    {
        normal = torch::Tensor();
    }
}

torch::Tensor Camera::loadImage(float downscale_factor)
{
    if (image.defined())
        throw std::runtime_error("Camera::loadImage called while image data is already resident");
    if (!std::isfinite(downscale_factor) || downscale_factor < 1.0f)
        throw std::runtime_error("READER.downscale_factor must be finite and at least 1.0");

    cv::Mat cImg = imreadRGB(imgFilePath);
    if (downscale_factor > 1.0f)
    {
        float scaleFactor = 1.0f / downscale_factor;
        cv::resize(cImg, cImg, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
        if (!load_geometry_initialized)
        {
            fx *= scaleFactor;
            fy *= scaleFactor;
            cx *= scaleFactor;
            cy *= scaleFactor;
        }
    }
    torch::Tensor res = imageToTensor(cImg);

    K = torch::tensor({{fx, 0.0f, cx},
                       {0.0f, fy, cy},
                       {0.0f, 0.0f, 1.0f}},
                      torch::kFloat32);
    // Update parameters
    height = res.size(0);
    width = res.size(1);
    fx = K[0][0].item<float>();
    fy = K[1][1].item<float>();
    cx = K[0][2].item<float>();
    cy = K[1][2].item<float>();
    load_geometry_initialized = true;
    return res;
}

torch::Tensor Camera::loadDepth(float downscale_factor, float depth_scale)
{
    if (depth.defined())
        throw std::runtime_error("Camera::loadDepth called while depth data is already resident");
    if (!std::isfinite(downscale_factor) || downscale_factor < 1.0f)
        throw std::runtime_error("READER.downscale_factor must be finite and at least 1.0");
    if (!std::isfinite(depth_scale) || depth_scale <= 0.0f)
        throw std::runtime_error("READER.depth_scale must be finite and greater than zero");
    cv::Mat cDepth = cv::imread(depthFilePath, cv::IMREAD_UNCHANGED);
    if (cDepth.empty())
        throw std::runtime_error(
            "Cannot read depth image: " + depthFilePath +
            ". Check that the configured depth path and codec are valid.");

    // 将深度图像转换为浮点型
    cDepth.convertTo(cDepth, CV_32F);
    cDepth /= depth_scale;

    if (downscale_factor > 1.0f)
    {
        float scaleFactor = 1.0f / downscale_factor;
        cv::resize(cDepth, cDepth, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
    }
    torch::Tensor res = depthToTensor(cDepth);
    return res;
}

std::string Camera::getFrameID(std::string prefix, std::string suffix) const
{
    if (!frame_id_str.empty())
        return frame_id_str;
    // 找到前缀的位置
    size_t startPos = imgFilePath.find(prefix);
    if (startPos != std::string::npos)
    {
        // 移动到前缀之后的位置
        startPos += prefix.length();
        // 找到后缀的位置
        size_t endPos = imgFilePath.find(suffix, startPos);
        if (endPos != std::string::npos)
        {
            // 提取出所需的子字符串
            std::string frameNumber = imgFilePath.substr(startPos, endPos - startPos);
            return frameNumber;
        }
        else
        {
            std::cerr << "suffix invalid!" << std::endl;
        }
    }
    else
    {
        std::cerr << "prefix invalid!" << std::endl;
    }
    return "error";
}

// TODO: rename variables
TensorDict Camera::getFrameMaps() const
{
    torch::Tensor depth_map = depth.to(torch::kCUDA);
    torch::Tensor color_map = image.to(torch::kCUDA);
    torch::Tensor intrinsics = K.to(torch::kCUDA);
    torch::Tensor extrinsics = c2w_slam.to(torch::kCUDA);

    auto options = torch::TensorOptions().device(depth_map.device()).dtype(torch::kFloat32);

    int H = depth_map.size(0);
    int W = depth_map.size(1);

    auto cam_coords = computeVertexMap(depth_map, intrinsics);

    // 1. 重塑顶点图为 (H*W, 3)
    auto vertices_c = cam_coords.reshape({-1, 3});

    // 2. 将顶点转换为齐次坐标，添加第四个分量 1
    auto ones = torch::ones({vertices_c.size(0), 1}, vertices_c.options());
    auto vertices_c_homogeneous = torch::cat({vertices_c, ones}, 1);

    // 3. 应用外参矩阵
    auto vertices_w_homogeneous = torch::matmul(vertices_c_homogeneous, extrinsics.t());

    // 4. 转换回非齐次坐标
    auto vertices_world = vertices_w_homogeneous.slice(1, 0, 3) /
                          vertices_w_homogeneous.slice(1, 3, 4);

    // 5. 重塑回原始形状
    auto vertex_map = vertices_world.reshape({H, W, 3});

    auto normal_map = computeNormalMap(vertex_map);

    // // 计算法向图
    // auto normal_map = torch::zeros({height, width, 3}, options);

    // for (int y = 1; y < height - 1; ++y)
    // {
    //     for (int x = 1; x < width - 1; ++x)
    //     {
    //         auto dzdx = (vertex_map[y][x + 1][2] - vertex_map[y][x - 1][2]) / 2;
    //         auto dzdy = (vertex_map[y + 1][x][2] - vertex_map[y - 1][x][2]) / 2;
    //         auto normal = torch::stack({-dzdx, -dzdy, torch::ones({1}, options)}, 0);
    //         normal = normal / normal.norm();
    //         normal_map[y][x] = normal;
    //     }
    // }
    TensorDict frame_maps;
    frame_maps["color_map"] = color_map;
    frame_maps["vertex_map"] = vertex_map;
    frame_maps["normal_map"] = normal_map;

    return frame_maps;
}
DatasetReader::DatasetReader(const YAML::Node &config)
{
    input_dir = config["input_dir"].as<std::string>();
    this->config = config;

    // get camera params
    const auto intrinsics = config["intrinsics"].as<std::vector<float>>();
    const auto image_shape = config["image_shape"].as<std::vector<int>>();
    if (intrinsics.size() != 4)
        throw std::runtime_error("READER.intrinsics must contain [fx, fy, cx, cy]");
    if (image_shape.size() != 2)
        throw std::runtime_error("READER.image_shape must contain [width, height]");
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
    width = image_shape[0];
    height = image_shape[1];
    if (!std::isfinite(fx) || !std::isfinite(fy) || fx <= 0.0f || fy <= 0.0f)
        throw std::runtime_error("READER focal lengths must be finite and greater than zero");
    if (!std::isfinite(cx) || !std::isfinite(cy))
        throw std::runtime_error("READER principal point must be finite");
    if (width <= 0 || height <= 0)
        throw std::runtime_error("READER.image_shape dimensions must be greater than zero");
    if (config["image_prefix"])
        image_prefix = config["image_prefix"].as<std::string>();
    if (config["image_extension"])
        image_extension = config["image_extension"].as<std::string>();
    if (config["depth_prefix"])
        depth_prefix = config["depth_prefix"].as<std::string>();
    if (config["depth_extension"])
        depth_extension = config["depth_extension"].as<std::string>();
    if (config["pose_prefix"])
        pose_prefix = config["pose_prefix"].as<std::string>();
    if (config["pose_extension"])
        pose_extension = config["pose_extension"].as<std::string>();
    auto normalize_extension = [](std::string &extension) {
        if (!extension.empty() && extension[0] != '.')
            extension = "." + extension;
    };
    normalize_extension(image_extension);
    normalize_extension(depth_extension);
    normalize_extension(pose_extension);
    std::string pcd_name = config["pcd_name"].as<std::string>();
    downscale_factor = config["downscale_factor"].as<float>();
    if (!std::isfinite(downscale_factor) || downscale_factor < 1.0f)
        throw std::runtime_error("READER.downscale_factor must be finite and at least 1.0");
    scene_scale = 1.1f * config["scene_scale"].as<float>();
    if (!std::isfinite(scene_scale) || scene_scale <= 0.0f)
        throw std::runtime_error("READER.scene_scale must be finite and positive");
    const std::string pose_source = config["pose_source"]
        ? config["pose_source"].as<std::string>() : "ground_truth";
    if (pose_source != "ground_truth" && pose_source != "identity")
        throw std::runtime_error("READER.pose_source must be ground_truth or identity");
    scene_centor = torch::zeros(3);
    fs::path dataRoot(input_dir);
    // load init points
    fs::path pointsPath = dataRoot / pcd_name;
    if (pose_source == "ground_truth" && fs::exists(pointsPath))
    {
        // 在初始点云文件存在的时候，读取初始点云
        std::cout << "read init scene point cloud." << std::endl;
        scene_points.readPly(pointsPath.string());
    }
    else
    {
        std::cout << "no init scene point cloud." << std::endl;
    }

    // check depth valid
    fs::path depthPath = dataRoot / config["depth_path"].as<std::string>();
    if (fs::exists(depthPath))
    {
        std::cout << "load data with depth!" << std::endl;
        has_depth = true;
    }
    else
    {
        std::cout << "load data without depth!" << std::endl;
        has_depth = false;
    }
}

std::vector<Camera> DatasetReader::getAllCams()
{
    auto all_cams = train_vec;
    if (config["test_split_interval"].as<int>() > 0)
        all_cams.insert(all_cams.end(), val_vec.begin(), val_vec.end());
    return all_cams;
}

void DatasetReader::read()
{
    // ProgressBar
    // Hide cursor
    show_console_cursor(false);
    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::PostfixText{"Read cameras"},
        option::ForegroundColor{Color::green},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

    // load path
    fs::path dataRoot(input_dir);
    fs::path imagePath = dataRoot / config["image_path"].as<std::string>();
    fs::path posePath = dataRoot / config["pose_path"].as<std::string>();
    fs::path depthPath = dataRoot / config["depth_path"].as<std::string>();
    // load id
    int start_frame = config["start_frame"].as<int>();
    int end_frame = config["end_frame"].as<int>();
    int frame_step = config["frame_step"].as<int>();
    int test_split_interval = config["test_split_interval"].as<int>();
    const bool preload_images = !config["preload_images"] || config["preload_images"].as<bool>();
    const float depth_scale = config["depth_scale"].as<float>();

    if (frame_step <= 0)
        throw std::runtime_error("READER.frame_step must be greater than zero");
    if (test_split_interval == 0)
        throw std::runtime_error("READER.test_split_interval must be -1 (no split) or a positive integer");

    if (end_frame <= 0)
    {
        int image_count = 0;
        for (const auto &entry : fs::directory_iterator(imagePath))
        {
            if (entry.is_regular_file() && entry.path().extension() == image_extension)
            {
                image_count++;
            }
        }
        end_frame = image_count > 0 ? start_frame + (image_count - 1) * frame_step : start_frame - 1;
    }

    if (end_frame < start_frame)
        throw std::runtime_error("READER.end_frame resolved before READER.start_frame");
    int read_count = 0;
    int total_frame = (end_frame - start_frame) / frame_step + 1;
    torch::Tensor init_c2w;
    for (int frame_id = start_frame; frame_id <= end_frame; frame_id += frame_step)
    {

        std::string filename = idToFilename(frame_id);
        std::string pose_filename = (posePath / (pose_prefix + filename + pose_extension)).string();
        std::string img_filename = (imagePath / (image_prefix + filename + image_extension)).string();
        std::string depth_filename = (depthPath / (depth_prefix + filename + depth_extension)).string();
        if (!fs::exists(img_filename))
            throw std::runtime_error(img_filename + " does not exist");
        if (has_depth && !fs::exists(depth_filename))
            throw std::runtime_error(depth_filename + " does not exist");
        torch::Tensor c2w_tensor = torch::eye(4, torch::kFloat32);
        const bool ground_truth = !config["pose_source"] ||
            config["pose_source"].as<std::string>() == "ground_truth";
        if (ground_truth)
        {
            const MatrixX loaded = readMatrixFromTXT(pose_filename);
            if (loaded.rows() != 4 || loaded.cols() != 4 || !loaded.allFinite())
                throw std::runtime_error("Invalid 4x4 camera pose: " + pose_filename);
            const Matrix4 c2w_mat = loaded;
            c2w_tensor = matrixToTensor(c2w_mat);
        }
        torch::Tensor ref_c2w_pose;
        if (read_count == 0)
        {
            init_c2w = c2w_tensor;
        }
        ref_c2w_pose = torch::matmul(poseInv(init_c2w), c2w_tensor);
        Camera cam(width, height, fx, fy, cx, cy, has_depth, ref_c2w_pose, img_filename, depth_filename);

        cam.c2w_slam = ref_c2w_pose.clone();
        cam.frame_id_str = filename;
        cam.load_downscale_factor = downscale_factor;
        cam.load_depth_scale = depth_scale;
        cam.release_after_frame = !preload_images;
        if (preload_images)
            cam.image = cam.loadImage(downscale_factor);
        cam.id = read_count;
        if (preload_images && has_depth)
            cam.depth = cam.loadDepth(downscale_factor, depth_scale);

        // cam.toGPU();

        if (test_split_interval < 0)
        {
            // 不区分训练、测试数据集
            train_vec.push_back(cam);
            val_vec.push_back(cam);
        }
        else
        {
            // 按照比例区分训练/测试数据集
            if (read_count % test_split_interval == 0)
                val_vec.push_back(cam);
            else
            {
                train_vec.push_back(cam);
            }
        }
        read_count++;
        bar.set_option(option::PostfixText{"Reading camera: " + std::to_string(read_count) + "/" + std::to_string(total_frame)});
        bar.set_progress(100 * read_count / total_frame);
    }
    show_console_cursor(true);
    if (train_vec.empty())
        throw std::runtime_error(
            "READER.test_split_interval produced an empty training set; use -1 or an interval greater than 1");
    std::cout << "train set num: " << train_vec.size() << std::endl;
    std::cout << "val set num: " << val_vec.size() << std::endl;
}

void DatasetReader::updateSceneGeo()
{
    if (config["pose_source"] && config["pose_source"].as<std::string>() == "identity")
    {
        std::cout << "[POSE SOURCE] estimated; configured scene scale: " << scene_scale << std::endl;
        return;
    }
    const auto all_cams = getAllCams();
    if (all_cams.size() <= 1)
    {
        // 当只有一个相机的时候，无法估算场景几何，直接返回
        scene_scale = 1.0f;
        scene_centor = torch::zeros({3});
        return;
    }
    torch::Tensor sum = torch::zeros({3});
    for (const auto &cam : all_cams)
    {
        torch::Tensor cam_location = cam.c2w.index({Slice(0, 3), 3});
        sum += cam_location;
    }

    scene_centor = sum / all_cams.size();
    float max_dist = 0.0;
    for (const auto &cam : all_cams)
    {
        torch::Tensor cam_location = cam.c2w.index({Slice(0, 3), 3});
        float dist = torch::norm(cam_location - scene_centor).item<float>();
        if (dist > max_dist)
        {
            max_dist = dist;
        }
    }
    scene_scale = 1.1f * max_dist;
    std::cout << "scene centor: " << scene_centor[0].item<float>() << "\t" << scene_centor[1].item<float>() << "\t" << scene_centor[2].item<float>() << std::endl;
    std::cout << "scene scale: " << scene_scale << std::endl;
}

void DatasetReader::savePose(const std::string &save_dir)
{
    fs::create_directories(save_dir);
    // std::cout << save_dir << std::endl;
    for (size_t i = 0; i < train_vec.size(); i++)
    {

        torch::Tensor slam_pose = train_vec[i].c2w_slam;
        // std::cout << slam_pose << std::endl;
        std::string pose_file_name = "frame" + train_vec[i].getFrameID() + ".txt";
        std::string save_file_name = fs::path(save_dir) / pose_file_name;
        saveTensorTXT(slam_pose, save_file_name);
    }
}

bool DatasetReader::loadSlamPose(const std::string &pose_dir)
{
    if (!fs::exists(pose_dir))
        return false;

    if (!fs::is_directory(pose_dir))
        throw std::runtime_error("Saved SLAM trajectory is not a directory: " + pose_dir);
    std::map<std::string, torch::Tensor> loaded;
    for (const auto &cam : getAllCams())
    {
        const std::string id = cam.getFrameID();
        const fs::path path = fs::path(pose_dir) / ("frame" + id + ".txt");
        if (!fs::is_regular_file(path))
            throw std::runtime_error("Incomplete saved SLAM trajectory: missing " + path.string());
        const MatrixX matrix = readMatrixFromTXT(path.string());
        if (matrix.rows() != 4 || matrix.cols() != 4 || !matrix.allFinite())
            throw std::runtime_error("Invalid saved 4x4 SLAM pose: " + path.string());
        const Matrix4 pose = matrix;
        loaded.emplace(id, matrixToTensor(pose));
    }
    // Commit only after every required pose has been validated.
    for (auto &cam : train_vec)
        cam.c2w_slam = loaded.at(cam.getFrameID()).clone();
    for (auto &cam : val_vec)
        cam.c2w_slam = loaded.at(cam.getFrameID()).clone();
    std::cout << "Loaded " << loaded.size() << " saved SLAM poses from: " << pose_dir << std::endl;
    return !loaded.empty();
}

void saveCameras(const std::vector<Camera> &input_cameras, const std::string &filename)
{
    json j = json::array();
    // std::cout << "save Camera num: " << input_cameras.size() << std::endl;
    for (size_t i = 0; i < input_cameras.size(); i++)
    {
        const Camera &cam = input_cameras[i];

        json camera = json::object();
        camera["id"] = i;
        camera["frame_id"] = cam.getFrameID();
        camera["img_name"] = fs::path(cam.imgFilePath).filename().string();
        camera["width"] = cam.width;
        camera["height"] = cam.height;
        camera["fx"] = cam.fx;
        camera["fy"] = cam.fy;

        torch::Tensor R = cam.c2w_slam.index({Slice(None, 3), Slice(None, 3)});
        torch::Tensor T = cam.c2w_slam.index({Slice(None, 3), Slice(3, 4)}).squeeze();

        std::vector<float> position(3);
        std::vector<std::vector<float>> rotation(3, std::vector<float>(3));
        for (int i = 0; i < 3; i++)
        {
            position[i] = T[i].item<float>();
            for (int j = 0; j < 3; j++)
            {
                rotation[i][j] = R[i][j].item<float>();
            }
        }

        camera["position"] = position;
        camera["rotation"] = rotation;
        j.push_back(camera);
    }

    std::ofstream of(filename);
    if (!of)
        throw std::runtime_error("Could not write cameras: " + filename);
    of << j;
    of.close();
    if (!of)
        throw std::runtime_error("Failed while writing cameras: " + filename);

    std::cout << "Wrote camera json to: " << filename << std::endl;
}

#include "pipeline.h"

#include <cmath>
#include <stdexcept>

namespace fs = std::filesystem;

namespace
{
bool pathContains(const fs::path &parent, const fs::path &child)
{
    const fs::path relative = child.lexically_relative(parent);
    if (relative.empty())
        return false;
    const auto first = relative.begin();
    return first == relative.end() || *first != "..";
}

std::string workspaceOutputPath(const std::string &workspace_dir,
                                const YAML::Node &config,
                                const char *key)
{
    if (!config[key])
        throw std::runtime_error(std::string("Missing PIPE.") + key);
    const fs::path configured = config[key].as<std::string>();
    if (configured.empty())
        throw std::runtime_error(std::string("PIPE.") + key + " must not be empty");

    const fs::path workspace = fs::weakly_canonical(fs::absolute(workspace_dir));
    const fs::path relative = configured.is_absolute() ? configured.relative_path() : configured;
    const fs::path output = fs::weakly_canonical(workspace / relative);
    if (output == workspace || !pathContains(workspace, output))
        throw std::runtime_error(
            std::string("PIPE.") + key + " must resolve to a child of workspace_dir");
    return output.string();
}
}

void Pipeline::loadConfig(const YAML::Node &config, const std::string &workspace_dir, bool is_train)
{
    // std::cout << "loading training pipeline parameters...";
    this->config = config;
    max_iterations = config["max_iterations"].as<int>();
    random_seed = config["random_seed"] ? config["random_seed"].as<int>() : 42;
    rng.seed(uint32_t(random_seed));
    torch::manual_seed(random_seed);
    selected_cam_idx = config["selected_cam_idx"].as<int>();
    enable_densify = config["enable_densify"].as<bool>();
    eval_after_train = config["eval_after_train"].as<bool>();
    save_after_train = config["save_after_train"].as<bool>();
    eval_split = config["eval_split"] ? config["eval_split"].as<std::string>() : "train";
    if (eval_split != "train" && eval_split != "val" && eval_split != "all")
        throw std::runtime_error("PIPE.eval_split must be one of: train, val, all");
    eval_stride = config["eval_stride"] ? config["eval_stride"].as<int>() : 1;
    eval_max_frames = config["eval_max_frames"] ? config["eval_max_frames"].as<int>() : 0;
    eval_start_offset = config["eval_start_offset"] ? config["eval_start_offset"].as<int>() : 0;
    if (eval_stride < 1)
        throw std::runtime_error("PIPE.eval_stride must be >= 1");
    if (eval_max_frames < 0)
        throw std::runtime_error("PIPE.eval_max_frames must be >= 0");
    if (eval_start_offset < 0)
        throw std::runtime_error("PIPE.eval_start_offset must be >= 0");
    eval_frame_indices.clear();
    if (config["eval_frame_indices"])
    {
        eval_frame_indices = config["eval_frame_indices"].as<std::vector<int>>();
        if (eval_frame_indices.empty())
            throw std::runtime_error("PIPE.eval_frame_indices must not be empty when provided");
        if (eval_stride != 1 || eval_max_frames != 0 || eval_start_offset != 0)
            throw std::runtime_error("PIPE.eval_frame_indices cannot be combined with stride/offset/max sampling");
        int previous = -1;
        for (const int index : eval_frame_indices)
        {
            if (index <= previous)
                throw std::runtime_error("PIPE.eval_frame_indices must be sorted, unique and nonnegative");
            previous = index;
        }
    }
    eval_image_extension = config["eval_image_extension"]
                               ? config["eval_image_extension"].as<std::string>() : ".jpg";
    if (eval_image_extension != ".jpg" && eval_image_extension != ".png")
        throw std::runtime_error("PIPE.eval_image_extension must be .jpg or .png");
    eval_write_auxiliary_images = config["eval_write_auxiliary_images"]
                                     ? config["eval_write_auxiliary_images"].as<bool>() : true;

    // output settings
    log_iter = config["log_iter"].as<int>();
    if (log_iter == -1)
        log_iter = max_iterations - 1;
    this->workspace_dir = workspace_dir;
    model_path = workspaceOutputPath(workspace_dir, config, "model_path");
    log_path = workspaceOutputPath(workspace_dir, config, "log_path");
    eval_path = workspaceOutputPath(workspace_dir, config, "eval_path");

#ifdef GPS_SLAM_ENABLE_TENSORBOARD
    tb_path = (fs::weakly_canonical(fs::absolute(workspace_dir)) / "tensorboard").string();
#endif
    weight_configs = config["weight_configs"];
    const float ssim_weight = weight_configs && weight_configs["ssim_weight"]
                                  ? weight_configs["ssim_weight"].as<float>()
                                  : 0.0f;
    const float depth_weight = weight_configs && weight_configs["depth_weight"]
                                   ? weight_configs["depth_weight"].as<float>()
                                   : 0.0f;
    if (!std::isfinite(ssim_weight) || ssim_weight < 0.0f || ssim_weight > 1.0f)
        throw std::runtime_error("PIPE.weight_configs.ssim_weight must be finite and in [0, 1]");
    if (!std::isfinite(depth_weight) || depth_weight < 0.0f)
        throw std::runtime_error("PIPE.weight_configs.depth_weight must be finite and >= 0");
    vis_configs = config["vis_configs"];
    if (is_train)
    {
        createDirectory(model_path, true);
        createDirectory(log_path, true);
        createDirectory(eval_path, true);
#ifdef GPS_SLAM_ENABLE_TENSORBOARD
        createDirectory(tb_path, true);
        GOOGLE_PROTOBUF_VERIFY_VERSION;
        tb_logger = std::make_unique<TensorBoardLogger>(
            (fs::path(tb_path) / "tfevents.pb").string());
#endif
    }
}

void Pipeline::writeEvalImage(const std::string &filename, const cv::Mat &image) const
{
    const fs::path path(filename);
    const bool scored_rgb = path.parent_path().filename() == "render" &&
                            path.stem().extension() == ".color";
    if (!eval_write_auxiliary_images && !scored_rgb)
        return;
    if (image.empty() || !cv::imwrite(filename, image))
        throw std::runtime_error("Failed to write evaluation image: " + filename);
}

cv::Mat Pipeline::evalRgbImage(const torch::Tensor &rgb) const
{
    if (rgb.dim() != 3 || rgb.size(2) != 3 || !torch::isfinite(rgb).all().item<bool>())
        throw std::runtime_error("Evaluation RGB must be a finite HxWx3 tensor");
    // Both common renderers use clamp, multiply by 255 and truncate to uint8.
    return tensorToImage(torch::clamp(rgb, 0, 1));
}

void Pipeline::save(RawGaussianModel &model, std::vector<Camera> cams)
{
    if (model.getGaussianNum() == 0)
    {
        std::cout << "empty model" << std::endl;
        return;
    }
    std::cout << "save model start" << std::endl;
    std::string save_path = model_path;
    createDirectory(fs::path(save_path) / "point_cloud", true);

    saveCfgArgs((fs::path(save_path) / "cfg_args").string(), model.getMaxSH());
    model.saveParamsPly((fs::path(save_path) / "point_cloud" / "point_cloud.ply").string());
    model.saveParamsTensor((fs::path(save_path) / "model.pt").string());
    saveCameras(cams,
                (fs::path(save_path) / "cameras.json").string());
}

// 根据curr iter保存结果
void Pipeline::logResults(TensorDict &render_res, TensorDict &log_res, const Camera &cam, std::vector<std::string> names, std::string log_mode)
{
    torch::NoGradGuard noGrad;
    logScalars(log_res, curr_iter);
    logScalars(render_res, curr_iter);

    float color_error_max = vis_configs["color_error_max"].as<float>();
    float depth_error_max = vis_configs["depth_error_max"].as<float>();
    float depth_vis_max = vis_configs["depth_vis_max"].as<float>();
    float depth_vis_min = vis_configs["depth_vis_min"].as<float>();
    float alpha_vis_max = vis_configs["alpha_vis_max"].as<float>();
    bool log_image = false;
    if (log_mode == "only scalar")
        return;
    else if (log_mode == "auto")
        log_image = curr_iter % log_iter == 0 || curr_iter + 1 == max_iterations;
    else if (log_mode == "scaler and image")
        log_image = true;
    else
        std::cout << "UNSUPPORT LOG MODE!!!!!!!!!!!!" << std::endl;
    if (log_image)
    {
        std::cout << "log results, curr iteration: " << curr_iter << std::endl;
        for (std::string name : names)
        {
            std::string frame_file_name = "frame" + cam.getFrameID() + "_iter" + std::to_string(curr_iter);
            if (name == "rgb")
            {
                torch::Tensor rendered_rgb = torch::clamp(render_res["rgb"], 0, 1);
                torch::Tensor gt_rgb = cam.image;
                cv::Mat rgb_image = colorCompImg(rendered_rgb, gt_rgb, 0, color_error_max);
                cv::imwrite((fs::path(log_path) / (frame_file_name + ".color.jpg")).string(),
                            rgb_image);
            }
            if (name == "alpha")
            {
                torch::Tensor rendered_alpha = render_res["alpha"];
                std::cout << rendered_alpha.mean() << std::endl;
                std::cout << rendered_alpha.max() << std::endl;
                std::cout << rendered_alpha.min() << std::endl;

                cv::Mat alpha_image = tensorToJetMat(rendered_alpha, 0, alpha_vis_max, false);
                cv::imwrite((fs::path(log_path) /
                             (frame_file_name + ".alpha.jpg"))
                                .string(),
                            alpha_image);
            }
            if (name == "depth")
            {
                torch::Tensor rendered_depth = render_res["depth"];
                if (cam.has_depth)
                {
                    torch::Tensor gt_depth = cam.depth;
                    cv::Mat depth_image = depthCompImg(rendered_depth, gt_depth, depth_vis_min, depth_vis_max, 0, depth_error_max);
                    cv::imwrite((fs::path(log_path) /
                                 (frame_file_name + ".depth.jpg"))
                                    .string(),
                                depth_image);
                }
                else
                {
                    cv::Mat depth_image = tensorToJetMat(rendered_depth, depth_vis_min, depth_vis_max, true);
                    cv::imwrite((fs::path(log_path) /
                                 (frame_file_name + ".depth.jpg"))
                                    .string(),
                                depth_image);
                }
            }
        }
    }
}

// 记录Dict中的标量到tensorboard里
void Pipeline::logScalars(TensorDict &scalars, int iter)
{
#ifdef GPS_SLAM_ENABLE_TENSORBOARD
    torch::NoGradGuard noGrad;
    if (!tb_logger)
        return;
    for (auto &pair : scalars)
    {
        std::string key = pair.first;
        torch::Tensor tensor = pair.second;
        if (tensor.numel() == 1)
        {
            auto dtype = tensor.dtype();
            if (dtype == torch::kFloat32)
                tb_logger->add_scalar(key, iter, tensor.item<float>());
            else if (dtype == torch::kInt32)
                tb_logger->add_scalar(key, iter, float(tensor.item<int>()));
            else
                std::cout << key << " unsupported data type!" << dtype << std::endl;
        }
        // else
        // {
        //     std::cout << key << " tensor numel is not 1!" << std::endl;
        // }
    }
#else
    (void)scalars;
    (void)iter;
#endif
}

// 按照原始3dgs的pipeline进行训练
void Pipeline::rawTrainCams(RawGaussianModel &model, std::vector<Camera> cams)
{
    assert(model.getRenderMethod() == "raw");
    // ProgressBar
    show_console_cursor(false);
    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::ForegroundColor{Color::green},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

    dist = std::uniform_int_distribution<std::size_t>(0, cams.size() - 1);
    model.setParamsDevice(model.device);
    model.setParamsRequireGrad();
    model.initOptimizers(max_iterations, scene_scale);
    RandomSelector data_loader(cams, {}, uint32_t(random_seed), true);
    // 开始优化
    for (; curr_iter < max_iterations; curr_iter++)
    {
        Camera cam;
        int cam_idx;
        // 随机选择优化的相机
        if (selected_cam_idx == -1)
        {
            auto sample = data_loader.getNext();
            cam = sample.value;
            cam_idx = sample.originalIndex;
        }
        else
        {
            cam = cams[selected_cam_idx];
            cam_idx = selected_cam_idx;
        }
        // 根据iter逐渐提升SH
        model.updateSH(curr_iter);
        // 渲染
        logStartTime();
        auto render_res = model.forward(cam);
        if (enable_densify)
        {
            render_res["means2d"].retain_grad();
        }
        TensorDict loss = model.computeLoss(render_res, cam, weight_configs);
        loss["total"].backward();
        logEndTime();

        model.optimizersStep();
        model.optimizersZeroGrad();
        if (enable_densify)
            model.stepPostBackward(render_res, cam, scene_scale, curr_iter);

        model.schedulersStep();
        // tensorboard记录结果
        TensorDict log_info = loss;
        auto duration = getLogTime();
        log_info["opt_state/opt_gs_num"] = torch::tensor(model.getGaussianNum()).to(torch::kInt32);
        log_info["opt_state/iter_time"] = torch::tensor(duration).to(torch::kFloat32);
        log_info["opt_state/opacity_mean"] = model.getRealOpacities().mean();
        logResults(render_res, log_info, cam, {"rgb"});

        // 更新progress bar的进度
        bar.set_option(option::PostfixText{"Training iter: " + std::to_string(curr_iter) + "/" + std::to_string(max_iterations) + " Loss: " + std::to_string(loss["total"].item<float>())});
        bar.set_progress(100 * curr_iter / max_iterations);
    }
}

// 按照ges-gs的pipeline进行训练
void Pipeline::gesTrainCams(RawGaussianModel &model, std::vector<Camera> cams)
{
    assert(model.getRenderMethod() == "ges");
    std::cout << "loading ref images...\n";
    // 读取mesh render的结果
    for (auto &cam : cams)
    {
        std::string mesh_rgb_name = config["mesh_rgb_path"].as<std::string>() + "/frame" + cam.getFrameID() + ".color.jpg";
        cv::Mat mesh_rgb_img = imreadRGB(mesh_rgb_name);
        torch::Tensor mesh_rgb_tensor = imageToTensor(mesh_rgb_img);

        std::string mesh_depth_name = config["mesh_depth_path"].as<std::string>() + "/frame" + cam.getFrameID() + ".depth.png";
        cv::Mat mesh_depth_img = imread(mesh_depth_name, cv::IMREAD_UNCHANGED);
        mesh_depth_img.convertTo(mesh_depth_img, CV_32F);

        mesh_depth_img /= 1000; // 固定保存深度单位为mm
        torch::Tensor mesh_depth_tensor = depthToTensor(mesh_depth_img);
        mesh_rgbs.emplace_back(mesh_rgb_tensor);
        mesh_depths.emplace_back(mesh_depth_tensor);
    }
    std::cout << "finish\n";
    // ProgressBar
    show_console_cursor(false);
    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::ForegroundColor{Color::green},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};
    dist = std::uniform_int_distribution<std::size_t>(0, cams.size() - 1);
    model.setParamsDevice(model.device);
    model.setParamsRequireGrad();
    model.initOptimizers(max_iterations, scene_scale);
    RandomSelector data_loader(cams, {}, uint32_t(random_seed ^ 0x517cc1b7), true);

    std::cout << "start training" << std::endl;
    // 开始优化
    for (; curr_iter < max_iterations; curr_iter++)
    {
        Camera cam;
        int cam_idx;
        // 随机选择优化的相机
        if (selected_cam_idx == -1)
        {
            auto sample = data_loader.getNext();
            cam = sample.value;
            cam_idx = sample.originalIndex;
        }
        else
        {
            cam = cams[selected_cam_idx];
            cam_idx = selected_cam_idx;
        }

        auto mesh_depth = mesh_depths[cam_idx].to(model.device);
        auto mesh_rgb = mesh_rgbs[cam_idx].to(model.device);
        auto cam_depth = cam.depth.to(model.device);
        auto cam_rgb = cam.image.to(model.device);
        // 根据iter逐渐提升SH
        model.updateSH(curr_iter);

        // 渲染
        logStartTime();
        auto render_res = model.forward(cam, mesh_depth, mesh_rgb);
        // auto render_res = model.forward(cam, cam_depth, cam_rgb);
        TensorDict loss = model.computeLoss(render_res, cam, weight_configs);
        loss["total"].backward();
        logEndTime();

        model.optimizersStep();
        model.optimizersZeroGrad();
        model.schedulersStep();
        // tensorboard记录结果
        TensorDict log_info = loss;
        auto duration = getLogTime();
        log_info["opt_state/opt_gs_num"] = torch::tensor(model.getGaussianNum()).to(torch::kInt32);
        log_info["opt_state/iter_time"] = torch::tensor(duration).to(torch::kFloat32);
        log_info["opt_state/opacity_mean"] = model.getRealOpacities().mean();
        logResults(render_res, log_info, cam, {"rgb"});

        // 更新progress bar的进度
        bar.set_option(option::PostfixText{"Training iter: " + std::to_string(curr_iter) + "/" + std::to_string(max_iterations) + " Loss: " + std::to_string(loss["total"].item<float>())});
        bar.set_progress(100 * curr_iter / max_iterations);
    }
}

void Pipeline::renderEvalImgs(RawGaussianModel &model, const std::vector<Camera> &cams, std::vector<std::string> names)
{
    torch::NoGradGuard noGrad;
    // ProgressBar
    show_console_cursor(false);
    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::PostfixText{"Eval cameras"},
        option::ForegroundColor{Color::green},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

    float color_error_max = vis_configs["color_error_max"].as<float>();
    float depth_error_max = vis_configs["depth_error_max"].as<float>();
    float depth_vis_max = vis_configs["depth_vis_max"].as<float>();
    fs::create_directories(fs::path(eval_path) / "gt");
    fs::create_directories(fs::path(eval_path) / "render");
    fs::create_directories(fs::path(eval_path) / "comp");
    int read_count = 0;

    for (Camera cam : cams)
    {
        cam.ensureLoaded();
        TensorDict render_res;
        if (model.getRenderMethod() == "ges")
        {
            auto mesh_depth = mesh_depths[read_count].to(model.device);
            auto mesh_rgb = mesh_rgbs[read_count].to(model.device);
            auto cam_depth = cam.depth.to(model.device);
            render_res = model.forward(cam, mesh_depth, mesh_rgb);
            // render_res = model.forward(cam, cam.depth, mesh_rgb);
            // std::cout << cam.id << std::endl;
            // std::cout << model.getExposure()[cam.id] << std::endl;
            // TensorDict loss = model.computeLoss(render_res, cam, weight_configs);
        }
        else if (model.getRenderMethod() == "raw")
            render_res = model.forward(cam);
        std::string frame_file_name = "frame" + cam.getFrameID() + "_iter" + std::to_string(curr_iter);
        for (std::string name : names)
        {
            if (name == "rgb")
            {
                torch::Tensor rendered_rgb = torch::clamp(render_res["rgb"], 0, 1);
                torch::Tensor gt_rgb = cam.image;
                // save gt image
                cv::Mat gt_img = tensorToImage(gt_rgb);
                writeEvalImage((fs::path(eval_path) / "gt" / (frame_file_name + ".color" + eval_image_extension)).string(),
                            gt_img);
                // save render image
                cv::Mat rendered_img = evalRgbImage(render_res["rgb"]);
                writeEvalImage((fs::path(eval_path) / "render" / (frame_file_name + ".color" + eval_image_extension)).string(),
                            rendered_img);
                // save compare image
                cv::Mat comp_image = colorCompImg(rendered_rgb, gt_rgb, 0, color_error_max);
                writeEvalImage((fs::path(eval_path) / "comp" / (frame_file_name + ".color" + eval_image_extension)).string(),
                            comp_image);
            }

            if (name == "alpha")
            {
                torch::Tensor rendered_alpha = render_res["alpha"];
                cv::Mat alpha_image = tensorToJetMat(rendered_alpha, 0, 1, false);
                writeEvalImage((fs::path(eval_path) / "render" / (frame_file_name + ".alpha.jpg")).string(),
                            alpha_image);
            }

            if (name == "depth")
            {
                torch::Tensor rendered_depth = render_res["depth"];
                cv::Mat rendered_img = tensorToJetMat(rendered_depth, 0, depth_vis_max, true);
                writeEvalImage((fs::path(eval_path) / "render" / (frame_file_name + ".depth.jpg")).string(),
                            rendered_img);
                if (cam.has_depth)
                {
                    torch::Tensor gt_depth = cam.depth;
                    // cv::Mat gt_depth_img = tensorToJetMat(gt_depth, 0, depth_vis_max, true);
                    // writeEvalImage((fs::path(eval_path) / "gt" / (frame_file_name + ".depth.jpg")).string(),
                    //             rendered_img);
                    // save compare image
                    cv::Mat comp_depth = depthCompImg(rendered_depth, gt_depth, 0, depth_vis_max, 0, depth_error_max);
                    writeEvalImage((fs::path(eval_path) / "comp" / (frame_file_name + ".depth.jpg")).string(),
                                comp_depth);
                }
            }
        }
        read_count++;
        bar.set_option(option::PostfixText{"Eval camera: " + frame_file_name});
        bar.set_progress(100 * read_count / cams.size());
    }
    show_console_cursor(true);
}

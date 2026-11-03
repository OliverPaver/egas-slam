#include "slam_pipeline.h"
#include <iomanip>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;
json matrixJson(const torch::Tensor &value)
{
    auto cpu = value.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    return std::vector<float>(cpu.data_ptr<float>(), cpu.data_ptr<float>() + cpu.numel());
}
void writeJson(const fs::path &path, const json &value)
{
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream stream(temporary);
    stream << value.dump(2) << '\n';
    stream.close();
    if (!stream) throw std::runtime_error("Cannot write demo metadata " + path.string());
    fs::rename(temporary, path);
}
void writeImage(const fs::path &path, const cv::Mat &image)
{
    if (!cv::imwrite(path.string(), image, {cv::IMWRITE_JPEG_QUALITY, 95}))
        throw std::runtime_error("Cannot write demonstration image");
}
}

void SLAMPipeline::captureDemoTransform(int owner, const torch::Tensor &before,
    const torch::Tensor &after, const std::string &reason)
{
    if ((before - after).abs().max().item<float>() <= 1e-6f) return;
    const fs::path root = fs::path(workspace_dir) / "demo";
    fs::create_directories(root);
    json record = {{"frame", curr_dataset_frame_id}, {"owner", owner},
        {"before_global_to_local", matrixJson(before)},
        {"after_global_to_local", matrixJson(after)}, {"reason", reason},
        {"event", "applied_owner_transform"}, {"index", demo_transform_count++}};
    std::ofstream stream(root / "transforms.jsonl", std::ios::app);
    stream << record.dump() << '\n';
    if (!stream) throw std::runtime_error("Cannot write demo transform event");
}

void SLAMPipeline::captureDemoFrame(const Camera &source, bool terminal, bool force)
{
    const auto options = config["demo_capture"];
    if (!options || !options["enabled"].as<bool>(false)) return;
    if (!use_gaussian_submaps || gaussian_submap_coordinate_frame != "local")
        throw std::runtime_error("Demo capture requires owner-local Gaussian submaps");
    const int interval = options["interval"].as<int>(10);
    if (interval < 1) throw std::runtime_error("Demo capture interval must be positive");
    const int frame = curr_dataset_frame_id;
    uint64_t epochs = 0;
    for (const auto &submap : gaussian_submaps) epochs += submap.transform_epoch;
    if (!force && frame % interval != 0 && demo_last_owner == source.owner_submap_id &&
        epochs == demo_last_epoch_sum) return;
    torch::NoGradGuard no_grad;
    auto *engine = dynamic_cast<ITMMultiEngine<ITMVoxel, ITMVoxelIndex> *>(tsdf_engine->getMainEngine());
    if (!engine) throw std::runtime_error("Demo capture requires the multi-map engine");
    const fs::path root = fs::path(workspace_dir) / "demo";
    fs::create_directories(root / "assets");
    std::ostringstream name;
    name << std::setfill('0') << std::setw(6) << frame << (terminal ? "_terminal" : "_online");
    const std::string stem = name.str();
    const auto active = tsdf_engine->getActiveSubmapIds();
    json output = {{"schema", 1}, {"mesh_display_cell_metres", 0.01}, {"mesh_format", "MSH1 indexed position/RGB float32; uint32 triangles"}, {"frame", frame}, {"stage", terminal ? "terminal" : "online"},
        {"scene", options["scene"].as<std::string>("scene")},
        {"dataset", options["dataset"].as<std::string>("dataset")},
        {"owner", source.owner_submap_id}, {"camera_to_world_raw", matrixJson(source.c2w_slam)},
        {"camera_to_owner", matrixJson(source.c2w_local)},
        {"width", source.width}, {"height", source.height},
        {"intrinsics", {source.fx, source.fy, source.cx, source.cy}},
        {"raw_tracking_status", tsdf_engine->getLastRawTrackingResult()},
        {"applied_transform_count", demo_transform_count},
        {"pending_transforms", pose_graph_pending_transform_updates.size()},
        {"resident_sdf_maps", tsdf_engine->getResidentSubmapCount()},
        {"sdf_maps", tsdf_engine->getSubmapCount()},
        {"submaps", json::array()}, {"loop_constraints", json::array()}};
    for (auto &s : gaussian_submaps)
    {
        if (s.retired || s.sdf_submap_id < 0 || s.sdf_submap_id >= tsdf_engine->getSubmapCount()) continue;
        const bool resident = tsdf_engine->isSubmapResident(s.sdf_submap_id);
        const bool updating = std::find(active.begin(), active.end(), s.sdf_submap_id) != active.end();
        if (resident && (updating || !demo_mesh_files.count(s.id) || force || demo_last_owner != source.owner_submap_id))
        {
            const std::string file = "assets/" + stem + "_s" + std::to_string(s.id) + ".mesh.bin";
            if (engine->CaptureResidentMesh(s.sdf_submap_id, (root / file).string()))
            {
                demo_mesh_files[s.id] = file;
                demo_mesh_frames[s.id] = frame;
            }
        }
        const int n = s.model.getGaussianNum();
        if (n > 0 && (!s.frozen || !demo_gaussian_files.count(s.id) || force))
        {
            // Representation view only: deterministic subsampling, actual scales
            // and quaternions. The native RGB rendering uses every owner Gaussian.
            const int step = std::max(1, (n + 7999) / 8000);
            const auto take = [step](const torch::Tensor &t) {
                return t.detach().slice(0, 0, t.size(0), step).to(torch::kCPU).to(torch::kFloat32).contiguous();
            };
            auto means = take(s.model.getRealMeans());
            auto scales = take(s.model.getRealScales());
            auto quats = take(s.model.getQuats());
            auto opacity = take(s.model.getRealOpacities()).reshape({-1, 1});
            auto rgb = (take(s.model.getFeaturesDc()) * 0.28209479177387814f + 0.5f).clamp(0, 1);
            auto packed = torch::cat({means, scales, quats, opacity, rgb}, 1).contiguous();
            if (!torch::isfinite(packed).all().item<bool>()) throw std::runtime_error("Non-finite demo Gaussian");
            const std::string file = "assets/" + stem + "_s" + std::to_string(s.id) + ".gaussians.bin";
            std::ofstream out(root / file, std::ios::binary);
            out.write(reinterpret_cast<const char *>(packed.data_ptr<float>()), packed.numel() * sizeof(float));
            if (!out) throw std::runtime_error("Cannot write demonstration Gaussians");
            demo_gaussian_files[s.id] = file;
        }
        output["submaps"].push_back({{"id", s.id}, {"sdf_id", s.sdf_submap_id},
            {"local_to_global", matrixJson(s.local_to_global)},
            {"sdf_local_to_global", matrixJson(infiMatrix4ToTensor(tsdf_engine->getGlobalToLocal(s.sdf_submap_id)).inverse())},
            {"epoch", s.transform_epoch}, {"frozen", s.frozen}, {"candidate", s.candidate_warming},
            {"finishing", s.freeze_pending}, {"resident", resident}, {"gaussian_count", n},
            {"mesh", demo_mesh_files.count(s.id) ? demo_mesh_files[s.id] : ""},
            {"mesh_frame", demo_mesh_frames.count(s.id) ? demo_mesh_frames[s.id] : -1},
            {"gaussians", demo_gaussian_files.count(s.id) ? demo_gaussian_files[s.id] : ""}});
    }
    for (const auto &c : hybrid_pose_graph_loop_constraints)
        output["loop_constraints"].push_back({{"from", c.from_hybrid_id}, {"to", c.to_hybrid_id},
            {"from_frame", c.verification_from_frame_id}, {"to_frame", c.verification_to_frame_id}});

    // Independent local raycast scratch: no map loads, render residency changes,
    // optimiser calls, saved-map processing or writes into tracking buffers.
    const int owner_idx = findGaussianSubmapBySdfId(source.owner_submap_id);
    if (owner_idx < 0) throw std::runtime_error("Demo camera owner missing");
    ITMUChar4Image colour(Vector2i(source.width, source.height), MEMORYDEVICE_CUDA);
    ITMFloat4Image vertices(Vector2i(source.width, source.height), MEMORYDEVICE_CUDA);
    ITMIntrinsics intrinsics;
    intrinsics.SetFrom(source.width, source.height, source.fx, source.fy, source.cx, source.cy);
    std::unique_ptr<ORUtils::Matrix4<float>> local_camera(tensorToInfiMatrix4(source.c2w_local));
    if (!engine->CaptureResidentView(source.owner_submap_id, *local_camera, intrinsics, colour, vertices))
        throw std::runtime_error("Current demo owner is not resident");
    auto base = ITMUChar4ImageToTensor(&colour);
    auto local_vertices = ITMUFloat4ImageToTensor(&vertices).slice(2, 0, 3) * tsdf_engine->getVoxelSize();
    auto camera_vertices = verticesTransform(local_vertices.contiguous(), poseInv(source.c2w_local).to(device));
    auto depth = camera_vertices.slice(2, 2, 3).contiguous();
    depth.masked_fill_(local_vertices.eq(0).all(2).unsqueeze(-1), 0);
    Camera local = source;
    local.c2w_slam = source.c2w_local;
    auto &model = gaussian_submaps[owner_idx].model;
    torch::Tensor hybrid = base;
    if (model.getGaussianNum() > 0)
        hybrid = model.forward(local, depth, base)["rgb"].clamp(0, 1);
    const std::string prefix = "assets/" + stem;
    const cv::Mat input = cv::imread(source.imgFilePath);
    if (input.empty() || input.cols != source.width || input.rows != source.height)
        throw std::runtime_error("Demo RGB input geometry mismatch");
    writeImage(root / (prefix + "_input.jpg"), input);
    writeImage(root / (prefix + "_hybrid.jpg"), tensorToImage(hybrid));
    writeImage(root / (prefix + "_sdf.jpg"), tensorToImage(base));
    output["input"] = prefix + "_input.jpg";
    output["hybrid"] = prefix + "_hybrid.jpg";
    output["sdf"] = prefix + "_sdf.jpg";
    output["render_scope"] = "native owner-local SDF/Gaussian hybrid, all owner Gaussians";
    writeJson(root / (stem + ".json"), output);
    // Append only once every referenced asset has been completely closed.
    std::ofstream timeline(root / "frames.jsonl", std::ios::app);
    timeline << json({{"file", stem + ".json"}, {"frame", frame},
                      {"stage", output["stage"]}}).dump() << '\n';
    if (!timeline) throw std::runtime_error("Cannot publish demonstration frame");
    writeJson(root / "latest.json", output);
    demo_last_owner = source.owner_submap_id;
    demo_last_epoch_sum = epochs;
    std::cout << "[DEMO] Captured " << stem << " submaps=" << output["submaps"].size() << std::endl;
}

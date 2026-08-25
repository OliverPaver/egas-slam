#include "dataset_reader.h"
#include "slam_pipeline.h"
#include "InfiniTAM_tools.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>

static bool g_bypass_libtorch_atexit = false;

static std::vector<Camera> evalSplitCameras(DatasetReader &data_reader, const std::string &eval_split)
{
    if (eval_split == "train")
        return data_reader.train_vec;
    if (eval_split == "val")
    {
        if (data_reader.val_vec.empty())
            throw std::runtime_error("PIPE.eval_split is 'val', but READER produced no validation cameras");
        return data_reader.val_vec;
    }
    if (eval_split == "all")
        return data_reader.getAllCams();
    throw std::runtime_error("Unsupported PIPE.eval_split: " + eval_split);
}

static std::vector<Camera> selectEvalCameras(DatasetReader &data_reader, const Pipeline &pipe)
{
    const std::vector<Camera> split_cams = evalSplitCameras(data_reader, pipe.eval_split);
    if (split_cams.empty())
        throw std::runtime_error("Selected PIPE.eval_split produced no evaluation cameras");

    std::vector<Camera> selected;
    selected.reserve(split_cams.size());
    for (size_t idx = static_cast<size_t>(pipe.eval_start_offset);
         idx < split_cams.size();
         idx += static_cast<size_t>(pipe.eval_stride))
    {
        selected.push_back(split_cams[idx]);
        if (pipe.eval_max_frames > 0 && selected.size() >= static_cast<size_t>(pipe.eval_max_frames))
            break;
    }

    if (selected.empty())
    {
        throw std::runtime_error(
            "PIPE eval sampling selected no cameras: split=" + pipe.eval_split +
            ", split_size=" + std::to_string(split_cams.size()) +
            ", eval_start_offset=" + std::to_string(pipe.eval_start_offset) +
            ", eval_stride=" + std::to_string(pipe.eval_stride) +
            ", eval_max_frames=" + std::to_string(pipe.eval_max_frames));
    }

    std::cout << "[EVAL] split=" << pipe.eval_split
              << " source_cameras=" << split_cams.size()
              << " selected_cameras=" << selected.size()
              << " eval_start_offset=" << pipe.eval_start_offset
              << " eval_stride=" << pipe.eval_stride
              << " eval_max_frames=" << pipe.eval_max_frames << std::endl;
    return selected;
}

static void writeEvalSamplingManifest(const Pipeline &pipe, const std::vector<Camera> &eval_cams)
{
    std::filesystem::create_directories(pipe.eval_path);
    const std::filesystem::path manifest_path = std::filesystem::path(pipe.eval_path) / "eval_sampling.json";
    std::ofstream out(manifest_path);
    if (!out)
        throw std::runtime_error("Could not write eval sampling manifest: " + manifest_path.string());

    out << "{\n";
    out << "  \"eval_split\": \"" << pipe.eval_split << "\",\n";
    out << "  \"eval_start_offset\": " << pipe.eval_start_offset << ",\n";
    out << "  \"eval_stride\": " << pipe.eval_stride << ",\n";
    out << "  \"eval_max_frames\": " << pipe.eval_max_frames << ",\n";
    out << "  \"selected_camera_count\": " << eval_cams.size() << ",\n";
    out << "  \"frame_ids\": [";
    for (size_t idx = 0; idx < eval_cams.size(); ++idx)
    {
        if (idx > 0)
            out << ", ";
        out << "\"" << eval_cams[idx].getFrameID() << "\"";
    }
    out << "]\n";
    out << "}\n";
    out.close();
    if (!out)
        throw std::runtime_error("Failed while writing eval sampling manifest: " + manifest_path.string());
}

int runTrainer(int argc, char *argv[])
{
    if (argc != 2)
        throw std::runtime_error("Usage: slam_trainer <config.yaml>");
    std::cout << "ours trainer demo!" << std::endl;
    const char *config_filename = argv[1];
    YAML::Node config = YAML::LoadFile(config_filename);
    g_bypass_libtorch_atexit = config["PIPE"] && config["PIPE"]["bypass_libtorch_atexit"]
                                   ? config["PIPE"]["bypass_libtorch_atexit"].as<bool>()
                                   : false;
    std::string workspace_dir = config["workspace_dir"].as<std::string>();
    std::string work_mode = config["work_mode"].as<std::string>();
    validateTsdfConfig(config["PIPE"]["TSDF"]);

    // setup cout precision
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);

    // setup cuda device
    const std::string devId = config["dev_id"].as<std::string>();
    setenv("CUDA_VISIBLE_DEVICES", devId.c_str(), 1);

    std::cout << "======= load camera data ======" << std::endl;
    DatasetReader data_reader(config["READER"]);
    data_reader.read();
    data_reader.updateSceneGeo();
    if (work_mode == "train" || work_mode == "recon")
        createWorkSpace(config_filename);

    std::cout << "======= Convert InfiniTAM data ======" << std::endl;
    TsdfEngineFacade *tsdf_engine = createTsdfEngine(data_reader, config["PIPE"]["TSDF"], workspace_dir);
    SLAMPipeline pipe;
    pipe.setTsdfEngine(tsdf_engine);
    pipe.work_mode = work_mode;
    pipe.device_id = config["dev_id"].as<int>();

    if (work_mode == "train" || work_mode == "recon")
    {
        std::cout << "======= setup Gaussian model ======" << std::endl;
        SLAMGaussianModel model;
        model.loadConfig(config["MODEL"]);

        std::cout << "======= setup training pipe ======" << std::endl;
        pipe.scene_scale = data_reader.scene_scale;
        pipe.loadConfig(config["PIPE"], workspace_dir, true);

        std::cout << "======= start training ======" << std::endl;
        pipe.SLAMTrainCams(model, data_reader.train_vec);
        std::cout << "train finish" << std::endl;
        if (pipe.isGaussianSubmapsEnabled())
        {
            std::cout << "submap num: " << pipe.getSubmapCount() << std::endl;
            std::cout << "submap keyframe num: " << pipe.getTotalSubmapKeyframeNum() << std::endl;
            std::cout << "submap gs num: " << pipe.getTotalSubmapGaussianNum() << std::endl;
        }
        else
        {
            std::cout << "keyframe num: " << pipe.keyframe_cam_list.size() << std::endl;
        }
        if (pipe.save_after_train)
        {
            std::cout << "======= save model, mesh and tsdf ======" << std::endl;
            if (pipe.isGaussianSubmapsEnabled())
                pipe.saveGaussianSubmaps(data_reader.getAllCams());
            else
                pipe.save(model, data_reader.getAllCams());
            pipe.saveEngine();
            pipe.saveMesh();
        }
        data_reader.savePose(pipe.eval_path + "/pose");
        if (pipe.hasCorrectedPosesAvailable())
            pipe.saveCorrectedPoses(data_reader.train_vec, pipe.eval_path + "/corrected_pose");

        if (pipe.eval_after_train)
        {
            std::cout << "======= render eval images ======" << std::endl;
            auto eval_cams = selectEvalCameras(data_reader, pipe);
            writeEvalSamplingManifest(pipe, eval_cams);
            std::cout << "eval split: " << pipe.eval_split << ", camera num: " << eval_cams.size() << std::endl;
            if (pipe.isGaussianSubmapsEnabled())
                pipe.renderEvalImgsSubmaps(eval_cams, {"rgb"});
            else
                pipe.renderEvalImgs(model, eval_cams, {"rgb"});
        }
        pipe.finalizeRuntimeDiagnostics();
    }

    else if (work_mode == "eval")
    {
        std::cout << "======= setup Gaussian model ======" << std::endl;
        SLAMGaussianModel model;
        model.loadConfig(config["MODEL"]);
        std::cout << "======= setup eval pipe ======" << std::endl;
        pipe.scene_scale = data_reader.scene_scale;
        pipe.loadConfig(config["PIPE"], workspace_dir, false);
        if (pipe.isGaussianSubmapsEnabled() &&
            !data_reader.loadSlamPose(pipe.eval_path + "/corrected_pose") &&
            !data_reader.loadSlamPose(workspace_dir + "/val/corrected_pose") &&
            !data_reader.loadSlamPose(pipe.eval_path + "/pose"))
            data_reader.loadSlamPose(workspace_dir + "/val/pose");
        else if (!pipe.isGaussianSubmapsEnabled() &&
                 !data_reader.loadSlamPose(pipe.eval_path + "/pose"))
            data_reader.loadSlamPose(workspace_dir + "/val/pose");
        pipe.loadEngine();
        std::cout << "======= start eval ======" << std::endl;
        if (pipe.isGaussianSubmapsEnabled())
        {
            pipe.loadGaussianSubmaps(model.getConfig());
            auto eval_cams = selectEvalCameras(data_reader, pipe);
            writeEvalSamplingManifest(pipe, eval_cams);
            std::cout << "eval split: " << pipe.eval_split << ", camera num: " << eval_cams.size() << std::endl;
            pipe.renderEvalImgsSubmaps(eval_cams, {"rgb"});
        }
        else
        {
            const bool eval_sdf_only = config["PIPE"]["eval_sdf_only"]
                ? config["PIPE"]["eval_sdf_only"].as<bool>()
                : false;
            const std::filesystem::path model_file = std::filesystem::path(workspace_dir) / "gs_model" / "model.pt";
            if (!eval_sdf_only)
            {
                if (!std::filesystem::exists(model_file))
                    throw std::runtime_error("Missing Gaussian model for eval: " + model_file.string() +
                                             ". Set PIPE.eval_sdf_only=true for TSDF-only checkpoint evaluation.");
                model.loadParamsTensor(model_file.string());
            }
            else
            {
                std::cout << "[EVAL] PIPE.eval_sdf_only=true; rendering TSDF raycast outputs without Gaussian model" << std::endl;
            }
            auto eval_cams = selectEvalCameras(data_reader, pipe);
            writeEvalSamplingManifest(pipe, eval_cams);
            std::cout << "eval split: " << pipe.eval_split << ", camera num: " << eval_cams.size() << std::endl;
            pipe.renderEvalImgs(model, eval_cams, {"rgb"});
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    int exit_code = 0;
    try
    {
        exit_code = runTrainer(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FATAL] " << e.what() << std::endl;
        exit_code = 1;
    }
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
    if (g_bypass_libtorch_atexit)
        std::_Exit(exit_code);
    return exit_code;
}

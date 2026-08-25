#include "pipeline.h"
#include "slam_gs_model.h"
#include "InfiniTAM_tools.h"
#include <deque>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

namespace MiniSlamGraph
{
class GraphEdgeSE3;
}

struct PoseGraphOrbFeaturePacket;

struct PoseGraphBowHistogramPacket
{
    std::map<int, int> words;
    int total = 0;
    int latest_frame_id = -1;
    int observation_count = 0;
    int word_bits = 0;
    int max_observations = 0;
    int loop_observation_max = 0;
    int keyframe_observation_max = 0;
    bool include_keyframes = false;
};

struct PoseGraphOrbWarmRequest
{
    Camera local_cam;
    std::string cache_key;
    std::string reason;
    int submap_idx = -1;
    int frame_id = -1;
};

struct GaussianSubmap
{
    int id = -1;
    int sdf_submap_id = -1;
    int start_frame_id = 0;
    int end_frame_id = -1;
    int start_sequence_id = 0;
    int end_sequence_id = -1;
    bool frozen = false;
    bool freeze_pending = false;
    bool candidate_warming = false;
    bool outgoing_handoff = false;
    bool retired = false;
    bool gaussian_resident = true;
    uint64_t last_gaussian_use = 0;
    uint64_t transform_epoch = 0;
    int candidate_transition_rejects = 0;
    int candidate_transition_last_reject_frame = -1;
    int candidate_transition_last_inliers = -1;
    float candidate_transition_last_visible = -1.0f;
    int candidate_transition_last_blocks = -1;
    int handoff_start_frame_id = -1;
    int handoff_until_frame_id = -1;
    int render_handoff_until_frame_id = -1;
    int handoff_opt_passes = 0;
    int graduation_opt_passes = 0;
    bool graduation_complete = false;
    int budget_last_prune_frame_id = -1;
    int budget_pressure_prunes = 0;
    int budget_cap_hits = 0;
    int last_budget_cap = -1;
    int cross_boundary_observations = 0;
    int cross_boundary_rejections = 0;
    int cross_boundary_last_frame_id = -1;
    std::vector<int> graduation_readiness_frame_ids;
    float graduation_readiness_last_delta = std::numeric_limits<float>::quiet_NaN();
    float graduation_readiness_last_sdf_l1 = std::numeric_limits<float>::quiet_NaN();
    float graduation_readiness_last_hybrid_l1 = std::numeric_limits<float>::quiet_NaN();
    int graduation_readiness_last_valid = 0;
    int frozen_refinement_passes = 0;
    int frozen_refinement_last_frame_id = -1;
    std::string close_reason;
    torch::Tensor origin_c2w;
    torch::Tensor global_to_local;
    torch::Tensor local_to_global;
    SLAMGaussianModel model;

    std::deque<Camera> localframe_cam_window;
    std::deque<TensorDict> localframe_raycast_window;
    std::map<std::string, std::vector<float>> keyframe_loss_dict;
    std::vector<Camera> keyframe_cam_list;
    std::map<std::string, Camera> keyframe_cam_dict;
    std::vector<Camera> training_observation_cam_list;
    std::vector<Camera> loop_observation_cam_list;
    std::vector<Camera> opt_cam_list;
    std::vector<TensorDict> opt_raycast_list;
};

struct GaussianOptimizationJob
{
    bool active = false;
    int submap_idx = -1;
    int trigger_sequence_id = -1;
    int trigger_frame_id = -1;
    int total_iterations = 0;
    int completed_iterations = 0;
    size_t trigger_raycast_index = 0;
    SLAMGaussianModel *model = nullptr;
    Camera trigger_camera;
    Camera trigger_global_camera;
    std::vector<Camera> cameras;
    std::vector<TensorDict> raycasts;
    std::vector<size_t> sample_indices;
};

struct HybridPoseGraphNode
{
    int hybrid_id = -1;
    int sdf_submap_id = -1;
    torch::Tensor local_to_global;
    uint64_t transform_epoch = 0;
};

struct PendingPoseGraphTransformUpdate
{
    int hybrid_id = -1;
    int sdf_submap_id = -1;
    torch::Tensor optimized_global_to_local;
    torch::Tensor optimized_local_to_global;
    std::string apply_mode;
    std::string source;
    int source_frame_id = -1;
};

struct HybridPoseGraphConstraint
{
    int from_hybrid_id = -1;
    int to_hybrid_id = -1;
    torch::Tensor from_local_to_to_local;
    std::string type;
    std::string source;
    float translation_distance = 0.0f;
    float rotation_degrees = 0.0f;
    float verification_translation_distance = 0.0f;
    float verification_rotation_degrees = 0.0f;
    float verification_depth_overlap = 0.0f;
    float verification_mean_depth_error = 0.0f;
    int verification_icp_inliers = 0;
    float verification_icp_rmse = 0.0f;
    float verification_icp_translation_delta = 0.0f;
    float verification_icp_rotation_delta_degrees = 0.0f;
    float verification_icp_allowed_translation_delta = 0.0f;
    float verification_icp_allowed_rotation_delta_degrees = 0.0f;
    float verification_icp_allowed_score = 0.0f;
    int verification_support_observations = 0;
    int verification_consistent_observations = 0;
    float verification_max_observation_translation_delta = 0.0f;
    float verification_max_observation_rotation_delta_degrees = 0.0f;
    int verification_from_frame_id = -1;
    int verification_to_frame_id = -1;
    int appearance_matches = 0;
    int appearance_quadrants = 0;
    float bow_score = 0.0f;
    torch::Tensor estimated_from_local_to_to_local;
    float score = 0.0f;
};

class SLAMPipeline : public Pipeline
{
public:
    ~SLAMPipeline();

    void loadConfig(const YAML::Node &config, const std::string &workspace_dir, bool is_train);

    void SLAMTrainCams(SLAMGaussianModel &model, std::vector<Camera> &cams);

    void finalizeRuntimeDiagnostics();

    void localOptimize(SLAMGaussianModel &model, int iter_override = -1);

    float evaluateGaussianOptimizationLoss(SLAMGaussianModel &model,
                                           const std::vector<Camera> &cams,
                                           const std::vector<TensorDict> &raycasts,
                                           int max_cameras = 0);
    float evaluateGaussianReadinessDelta(SLAMGaussianModel &model,
                                         const std::vector<Camera> &cams,
                                         const std::vector<TensorDict> &raycasts,
                                         int max_cameras = 0,
                                         float *sdf_l1_out = nullptr,
                                         float *hybrid_l1_out = nullptr,
                                         int *valid_count_out = nullptr);

    void removeRedundantGs(SLAMGaussianModel &model);

    void updateFrameList();

    void localFrameRaycast();

    void initNewGaussians(SLAMGaussianModel &model, TensorDict &raycast_maps);

    void setTsdfEngine(TsdfEngineFacade *tsdf_engine)
    {
        this->tsdf_engine = tsdf_engine;
        this->main_engine = tsdf_engine->getMainEngine();
        raycast_color_raw_temp = new ITMUChar4Image(tsdf_engine->getDepthSize(), true, true);
        raycast_vertex_raw_temp = new ITMFloat4Image(tsdf_engine->getDepthSize(), true, true);
        voxel_size = tsdf_engine->getVoxelSize();
    }

    void saveMesh()
    {
        if (config["TSDF"]["saved_mesh"].as<std::string>() == "")
            return;
        tsdf_engine->saveMesh(workspace_dir + "/" + config["TSDF"]["saved_mesh"].as<std::string>());
    }

    void saveEngine()
    {
        if (config["TSDF"]["saved_engine"].as<std::string>() == "")
            return;
        tsdf_engine->save(workspace_dir + "/" + config["TSDF"]["saved_engine"].as<std::string>());
        const YAML::Node multi_cfg = config["TSDF"]["multi"];
        const bool keep_runtime_pages = multi_cfg && multi_cfg["keep_runtime_pages_after_save"]
            ? multi_cfg["keep_runtime_pages_after_save"].as<bool>()
            : true;
        if (tsdf_engine->isMulti() && !keep_runtime_pages && eval_after_train)
        {
            std::cout << "[TSDF] Keeping runtime TSDF page snapshots because eval_after_train is enabled" << std::endl;
            return;
        }
		if (tsdf_engine->isMulti() && !keep_runtime_pages)
			cleanupRuntimePages("checkpoint");
    }

    void cleanupRuntimePages(const char *reason)
    {
        if (tsdf_engine == nullptr || !tsdf_engine->isMulti() || !config || !config["TSDF"])
            return;
        const YAML::Node multi_cfg = config["TSDF"]["multi"];
        const bool keep_runtime_pages = multi_cfg && multi_cfg["keep_runtime_pages_after_save"]
            ? multi_cfg["keep_runtime_pages_after_save"].as<bool>()
            : true;
        if (keep_runtime_pages)
            return;
        try
        {
            tsdf_engine->synchronizePaging();
        }
        catch (const std::exception &error)
        {
            std::cerr << "[TSDF] Runtime page synchronization failed during " << reason
                      << ": " << error.what() << std::endl;
            return;
        }
        const std::filesystem::path runtime_pages =
            std::filesystem::path(workspace_dir) / "tsdf_pages";
        std::error_code error;
        const uintmax_t removed = std::filesystem::remove_all(runtime_pages, error);
        if (error)
        {
            std::cerr << "[TSDF] Failed to remove runtime page snapshots during " << reason
                      << ": " << runtime_pages << " error=" << error.message() << std::endl;
            return;
        }
        if (removed > 0)
            std::cout << "[TSDF] Removed runtime TSDF page snapshots during " << reason
                      << ": " << runtime_pages << " entries=" << removed << std::endl;
    }

    void loadEngine()
    {
        tsdf_engine->load(workspace_dir + "/" + config["TSDF"]["saved_engine"].as<std::string>());
    }

    // 输入任意dataset内的相机，使用main engine里预先存好的cam pose进行raycast，返回深度图和颜色图
    TensorDict runRaycastByCam(const Camera &cam, bool use_cam_depth = true, bool combined_global = false,
                               const std::vector<int> &residency_pins = {});
    TensorDict runRaycastByCamWithTsdfSubmaps(const Camera &cam, bool use_cam_depth,
                                              const std::vector<int> &render_submaps,
                                              bool trim_after,
                                              const std::vector<int> &extra_residency_pins = {},
                                              bool trim_before = true);

    void renderEvalImgs(SLAMGaussianModel &model, const std::vector<Camera> &cams, std::vector<std::string> names);

    void renderEvalImgsSubmaps(const std::vector<Camera> &cams, std::vector<std::string> names);

    void saveGaussianSubmaps(std::vector<Camera> cams);

    void saveCorrectedPoses(const std::vector<Camera> &cams, const std::string &save_dir) const;

    void loadGaussianSubmaps(const YAML::Node &model_config);

    void buildMergedGaussianModel(SLAMGaussianModel &merged_model);

    void buildSelectedGaussianModel(SLAMGaussianModel &merged_model, const std::vector<int> &submap_indices);

    void keyFrameRaycast(SLAMGaussianModel &model);

    void checkKeyFrameError(SLAMGaussianModel &model);

    bool isGaussianSubmapsEnabled() const
    {
        return use_gaussian_submaps;
    }

    bool hasCorrectedPosesAvailable() const;

    int getTotalSubmapGaussianNum() const;

    int getTotalSubmapKeyframeNum() const;

    int getSubmapCount() const
    {
        return gaussian_submaps.size();
    }

    // private:
    torch::Device device = torch::kCUDA;
    std::string work_mode = "train";
    int device_id = 0;

    TsdfEngineFacade *tsdf_engine = nullptr;
    ITMMainEngine *main_engine = nullptr;
    float voxel_size = 0.0f;
    ITMUChar4Image *raycast_color_raw_temp = nullptr;
    ITMFloat4Image *raycast_vertex_raw_temp = nullptr;
    cv::Mat raycast_color_mat, raycast_vertex_mat;

    int curr_frame_id = -1;
    int localframe_cam_window_length, localframe_cam_window_interval;
    int local_opt_iters, local_opt_interval;
    std::string gaussian_schedule_mode = "immediate";
    int gaussian_schedule_iterations_per_frame = 1;
    GaussianOptimizationJob gaussian_optimization_job;
    bool gaussian_optimization_job_executing = false;
    bool export_online_end_trajectory = false;
    int keyframe_select_max = 2;
    Camera curr_cam;

    std::deque<Camera> localframe_cam_window;
    std::deque<TensorDict> localframe_raycast_window;
    std::map<std::string, std::vector<float>> keyframe_loss_dict;
    std::vector<Camera> keyframe_cam_list;
    std::map<std::string, Camera> keyframe_cam_dict;
    std::vector<Camera> opt_cam_list;
    std::vector<TensorDict> opt_raycast_list;

    float keyframe_theta_thres;
    float keyframe_trans_thres;
    bool is_keyframe = false;
    bool log_slam_state;
    int submap_stats_interval_frames = 1;
    bool tsdf_require_raw_tracking_success = false;
    int tsdf_raw_tracking_failure_grace_frames = 0;

    float new_gs_sample_ratio;
    float empty_alpha_thres;
    float color_error_thres;

    bool use_gaussian_submaps = false;
    bool gaussian_submaps_debug = true;
    std::string gaussian_submap_ownership = "legacy";
    std::string gaussian_submap_coordinate_frame = "global";
    int active_submap_idx = -1;
    int curr_dataset_frame_id = -1;
    int last_dataset_frame_id = -1;
    int submap_max_frames = 150;
    int submap_overlap_local_frames = 2;
    int submap_overlap_keyframes = 2;
    int submap_max_gaussians = 80000;
    int submap_max_new_gaussians_per_frame = 0;
    int submap_bootstrap_max_new_gaussians_per_frame = 0;
    int submap_bootstrap_until_gaussians = 0;
    int submap_bootstrap_until_age_frames = 0;
    bool submap_gaussian_insert_diagnostics = false;
    int submap_gaussian_insert_min_age_frames = 0;
    int submap_gaussian_insert_min_sdf_blocks = 0;
    float submap_max_translation = 2.0f;
    float submap_max_rotation = 75.0f;
    bool submap_handoff_optimize = true;
    bool submap_candidate_warming = true;
    bool submap_boundary_observation_handoff = true;
    bool submap_handoff_add_gaussians_to_outgoing = false;
    std::string submap_gaussian_insert_selection = "random";
    int submap_gaussian_insert_tile_size = 32;
    int submap_gaussian_insert_tile_max_pixels = 0;
    int submap_gaussian_insert_adaptive_tile_min_candidate_pixels = 1024;
    int submap_gaussian_insert_adaptive_tile_min_target_insertions = 256;
    int submap_gaussian_insert_adaptive_tile_min_submap_gaussians = 0;
    int submap_gaussian_insert_adaptive_tile_min_observed_frames = 0;
    std::string submap_keyframe_selection_policy = "random";
    int submap_candidate_warm_interval = 1;
    bool submap_retire_unaccepted_candidates = true;
    bool submap_candidate_reject_retire = false;
    int submap_candidate_reject_retire_min_rejects = 8;
    int submap_candidate_reject_retire_min_observed_frames = 16;
    int submap_candidate_reject_retire_min_gaussians = 1;
    int submap_candidate_reject_retire_max_inliers = 1;
    int submap_candidate_reject_retire_cooldown_frames = 30;
    int submap_candidate_gaussian_insert_min_frames = 0;
    float submap_candidate_immature_new_gs_sample_ratio = -1.0f;
    int submap_candidate_immature_max_new_gaussians_per_frame = 0;
    bool submap_candidate_extend_while_overlap = false;
    int submap_candidate_max_warm_frames = 0;
    int submap_candidate_handoff_max_opt_passes = -1;
    int submap_candidate_handoff_opt_iters = -1;
    int submap_handoff_frames = 30;
    int submap_handoff_opt_iters = 3;
    int submap_handoff_max_opt_passes = 4;
    bool submap_graduation_optimize = false;
    int submap_graduation_opt_iters = 0;
    int submap_graduation_keyframes = 8;
    int submap_graduation_max_opt_passes = 1;
    std::string submap_graduation_keyframe_policy = "loss_temporal_mix";
    std::string submap_graduation_observation_policy = "temporal_spread";
    bool submap_graduation_adaptive = false;
    int submap_graduation_adaptive_loss_sample_cameras = 6;
    float submap_graduation_adaptive_target_loss = 0.0f;
    float submap_graduation_adaptive_min_abs_drop = 0.0001f;
    float submap_graduation_adaptive_min_rel_drop = 0.005f;
    bool submap_graduation_readiness_enable = false;
    int submap_graduation_readiness_sample_cameras = 5;
    float submap_graduation_readiness_skip_gain = 0.0015f;
    float submap_graduation_readiness_min_abs_drop = 0.0001f;
    int submap_graduation_readiness_skip_min_observed_frames = 0;
    int submap_graduation_readiness_skip_min_keyframes = 0;
    int submap_training_observation_max_per_submap = 0;
    int submap_training_observation_interval_frames = 10;
    bool submap_quality_finalize_enable = false;
    bool submap_quality_finalize_include_frozen = true;
    bool submap_quality_finalize_include_active = true;
    bool submap_quality_finalize_resident_only = false;
    int submap_quality_finalize_max_submaps = 2;
    int submap_quality_finalize_max_observed_frames = 256;
    int submap_quality_finalize_min_keyframes = 8;
    float submap_quality_finalize_max_hybrid_l1 = 0.0f;
    bool submap_quality_finalize_insert_enable = false;
    bool submap_quality_finalize_insert_include_frozen = false;
    int submap_quality_finalize_insert_min_sdf_submaps = 2;
    int submap_quality_finalize_insert_frames = 2;
    int submap_quality_finalize_insert_max_new_gaussians_per_frame = 1500;
    bool submap_quality_finalize_insert_acceptance_enable = true;
    int submap_quality_finalize_insert_acceptance_cameras = 5;
    float submap_quality_finalize_insert_min_readiness_delta = 0.0f;
    bool submap_frozen_refinement_enable = false;
    int submap_frozen_refinement_interval_frames = 30;
    int submap_frozen_refinement_opt_iters = 1;
    int submap_frozen_refinement_keyframes = 6;
    int submap_frozen_refinement_max_passes_per_submap = 1;
    int submap_frozen_refinement_max_submaps_per_frame = 1;
    int submap_frozen_refinement_min_new_observations = 1;
    bool submap_frozen_refinement_readiness_guard_enable = false;
    bool submap_frozen_refinement_readiness_use_cached_graduation = true;
    int submap_frozen_refinement_readiness_sample_cameras = 5;
    float submap_frozen_refinement_readiness_max_pre_gain = 0.0f;
    int submap_frozen_refinement_readiness_min_valid_cameras = 1;
    bool submap_frozen_refinement_acceptance_enable = false;
    float submap_frozen_refinement_acceptance_min_delta = 0.0f;
    std::string submap_frozen_refinement_observation_policy = "temporal_spread";
    int submap_frozen_refinement_observation_selection_max_eval = 0;
    bool submap_cross_boundary_refinement_enable = false;
    int submap_cross_boundary_observation_interval_frames = 10;
    int submap_cross_boundary_max_submaps_per_frame = 1;
    int submap_cross_boundary_min_owner_age_frames = 0;
    int submap_cross_boundary_min_neighbor_gaussians = 512;
    bool submap_cross_boundary_frozen_only = true;
    bool submap_cross_boundary_resident_only = true;
    bool submap_cross_boundary_sequential_only = true;
    bool submap_cross_boundary_depth_gate = true;
    float submap_cross_boundary_min_projected_ratio = 0.02f;
    float submap_cross_boundary_min_depth_consistent_ratio = 0.50f;
    std::string submap_cross_boundary_depth_tolerance_mode = "scaled_max";
    float submap_cross_boundary_depth_abs_thres = 0.05f;
    float submap_cross_boundary_depth_floor = 0.015f;
    float submap_cross_boundary_depth_voxel_scale = 0.75f;
    float submap_cross_boundary_depth_quadratic_scale = 0.005f;
    float submap_candidate_new_gs_sample_ratio = -1.0f;
    int submap_candidate_max_new_gaussians_per_frame = 0;
    int submap_render_neighbor_count = 1;
    std::string submap_render_selection_policy = "temporal";
    int submap_render_max_visible_submaps = 0;
    int submap_render_max_visible_gaussians = 0;
    float submap_render_bounds_inflate = 0.05f;
    float submap_render_bounds_min_projected_area = 0.0f;
    int submap_render_selection_hysteresis_frames = 0;
    float submap_render_selection_hysteresis_area_bonus = 0.05f;
    int submap_render_selection_continuity_frame_scale = 250;
    float submap_render_selection_continuity_min_factor = 0.05f;
    float submap_render_selection_adjacent_bonus = 0.35f;
    float submap_render_selection_active_overlap_bonus = 0.25f;
    bool submap_cross_submap_novelty_gate = false;
    std::string submap_cross_novelty_reference_policy = "legacy_previous";
    std::string submap_cross_novelty_depth_tolerance_mode = "legacy_additive";
    float submap_cross_novelty_alpha_thres = 0.7f;
    float submap_cross_novelty_depth_abs_thres = 0.03f;
    float submap_cross_novelty_depth_rel_thres = 0.01f;
    float submap_cross_novelty_depth_floor = 0.015f;
    float submap_cross_novelty_depth_voxel_scale = 0.75f;
    float submap_cross_novelty_depth_quadratic_scale = 0.005f;
    float submap_cross_novelty_photometric_thres = 0.2f;
    int submap_cross_novelty_max_reference_submaps = 0;
    bool submap_cross_novelty_previous_only = true;
    bool submap_cross_novelty_resident_only = true;
    int submap_cross_novelty_erode_kernel = 3;
    int submap_render_frame_margin = 20;
    int submap_render_past_frame_margin = -1;
    int submap_render_future_frame_margin = -1;
    int submap_render_max_neighbor_submaps = 1;
    int submap_render_min_owner_age_frames = 0;
    int submap_render_min_ready_frames = 16;
    int submap_render_owner_min_ready_frames = -1;
    int submap_render_neighbor_min_ready_frames = -1;
    int submap_render_min_neighbor_gaussians = 0;
    std::string submap_gaussian_budget_policy = "fixed";
    int submap_gaussian_budget_base = 8000;
    float submap_gaussian_budget_blocks_density = 0.40f;
    int submap_gaussian_budget_min = 8000;
    int submap_gaussian_budget_max = 90000;
    float submap_gaussian_budget_pressure_ratio = 0.90f;
    int submap_gaussian_budget_prune_interval_frames = 10;
    bool submap_render_nonsequential_overlaps = false;
    int tsdf_max_render_maps = 4;
    std::string tsdf_render_compositor = "nearest_depth";
    std::string tsdf_render_order_policy = "inferred";
    bool tsdf_render_fill_requires_boundary = false;
    int tsdf_render_fill_boundary_kernel = 5;
    int tsdf_render_fill_boundary_min_valid = 1;
    float tsdf_render_fill_boundary_depth_thres = 0.05f;
    bool tsdf_render_from_residency_pins = false;
    bool tsdf_render_pins_from_gaussian_selection = false;
    int submap_render_cache_limit = 8;
    bool submap_page_frozen_gaussians = false;
    int submap_resident_gaussian_submaps = 3;
    bool tsdf_render_pin_primary_submap = true;
    uint64_t gaussian_residency_clock = 0;
    int last_handoff_optimize_frame = -1;
    std::map<int, int> submap_candidate_reject_retire_cooldown_until;
    float submap_render_owner_alpha_fill_thres = 0.25f;
    bool submap_render_fill_require_sdf_hole = true;
    float submap_render_fill_blend_weight = 0.2f;
    int submap_render_adaptive_kernel = 5;
    float submap_render_adaptive_alpha_scale = 2.0f;
    float submap_render_adaptive_min_weight = 0.02f;
    float submap_render_adaptive_boundary_feather = 0.5f;
    float submap_render_adaptive_depth_thres = 0.05f;
    bool submap_render_hole_requires_boundary = false;
    int submap_render_hole_boundary_kernel = 5;
    int submap_render_hole_boundary_min_valid = 1;
    float submap_render_hole_boundary_depth_thres = 0.05f;
    float submap_render_blend_alpha_thres = 0.1f;
    float submap_render_blend_depth_thres = 0.05f;
    float submap_render_blend_max_weight = 0.2f;
    float submap_render_blend_weight_scale = 0.2f;
    float submap_render_neighbor_alpha_fill_thres = 0.05f;
    bool submap_render_include_handoff_submaps = false;
    bool submap_render_single_submap_passthrough = true;
    bool submap_render_single_submap_passthrough_requires_single_logical_map = false;
    bool submap_render_use_local_sdf_reference = false;
    bool submap_render_recompose_with_global_sdf_base = false;
    bool submap_render_overlap_depth_gate = false;
    std::string submap_render_overlap_depth_tolerance_mode = "fixed";
    float submap_render_overlap_depth_abs_thres = 0.05f;
    float submap_render_overlap_depth_floor = 0.015f;
    float submap_render_overlap_depth_voxel_scale = 0.75f;
    float submap_render_overlap_depth_quadratic_scale = 0.005f;
    float submap_render_overlap_alpha_thres = 0.05f;
    bool submap_render_overlap_diagnostics = false;
    std::string submap_render_policy = "handoff_aware_fill";
    bool tsdf_render_prefetch = false;
    int tsdf_render_prefetch_lookahead_frames = 1;
    int tsdf_render_prefetch_lookahead_max_maps = 0;
    bool submap_save_global_merged_model = true;
    bool pose_graph_identity_check = false;
    bool pose_graph_synthetic_check = false;
    bool pose_graph_overlap_roundtrip_check = false;
    float pose_graph_overlap_roundtrip_tolerance = 1e-4f;
    bool pose_graph_render_identity_check = false;
    bool pose_graph_render_synthetic_check = false;
    int pose_graph_synthetic_submap_id = -1;
    std::vector<float> pose_graph_synthetic_translation = {0.01f, 0.0f, 0.0f};
    bool pose_graph_candidate_detection = false;
    bool pose_graph_optimize = false;
    bool pose_graph_dry_run = false;
    bool pose_graph_dry_run_include_loop_candidates = false;
    bool pose_graph_dry_run_use_estimated_loop_transforms = true;
    float pose_graph_dry_run_max_sequential_translation_residual = 1e-4f;
    float pose_graph_dry_run_max_sequential_rotation_residual_deg = 0.1f;
    float pose_graph_dry_run_max_loop_translation_residual = 0.30f;
    float pose_graph_dry_run_max_loop_rotation_residual_deg = 30.0f;
    bool pose_graph_solve_only = false;
    bool pose_graph_solve_only_include_loop_candidates = false;
    bool pose_graph_solve_only_use_estimated_loop_transforms = true;
    float pose_graph_solve_only_max_translation_delta = 0.30f;
    float pose_graph_solve_only_max_rotation_delta_deg = 30.0f;
    float pose_graph_sequential_constraint_weight = 1.0f;
    float pose_graph_loop_constraint_weight = 1.0f;
    std::string pose_graph_information_policy = "scalar";
    std::string pose_graph_loop_robust_kernel = "none";
    float pose_graph_loop_robust_scale = 1.0f;
    float pose_graph_loop_information_min_confidence = 0.25f;
    float pose_graph_loop_information_max_confidence = 8.0f;
    std::string pose_graph_anchor_policy = "first";
    std::string pose_graph_apply_mode = "disabled";
    bool pose_graph_apply_use_estimated_loop_transforms = true;
    float pose_graph_apply_correction_scale = 1.0f;
    bool pose_graph_apply_scale_backoff = false;
    float pose_graph_apply_scale_backoff_min_scale = 0.25f;
    float pose_graph_apply_scale_backoff_factor = 0.5f;
    int pose_graph_apply_scale_backoff_max_trials = 3;
    float pose_graph_apply_max_translation_delta = 0.30f;
    float pose_graph_apply_max_rotation_delta_deg = 30.0f;
    float pose_graph_apply_max_sequential_translation_residual = 0.008f;
    float pose_graph_apply_max_sequential_rotation_residual_deg = 0.75f;
    float pose_graph_apply_min_translation_delta = 0.0f;
    int pose_graph_apply_min_loop_constraints = 4;
    int pose_graph_apply_min_loop_submap_span = 4;
    int pose_graph_apply_min_loop_consistent_observations = 2;
    int pose_graph_apply_min_loop_icp_inliers = 16;
    float pose_graph_apply_max_loop_score = 0.5f;
    std::string pose_graph_apply_support_policy = "strict";
    int pose_graph_apply_adaptive_min_loop_constraints = 1;
    int pose_graph_apply_adaptive_min_loop_submap_span = 2;
    int pose_graph_apply_adaptive_min_loop_consistent_observations = 3;
    int pose_graph_apply_adaptive_min_loop_icp_inliers = 48;
    float pose_graph_apply_adaptive_max_loop_score = 0.4f;
    int pose_graph_apply_high_confidence_min_loop_constraints = 1;
    int pose_graph_apply_high_confidence_min_loop_submap_span = 10;
    int pose_graph_apply_high_confidence_min_loop_consistent_observations = 3;
    int pose_graph_apply_high_confidence_min_loop_icp_inliers = 64;
    float pose_graph_apply_high_confidence_max_loop_score = 0.5f;
    float pose_graph_apply_high_confidence_max_icp_rmse = 0.025f;
    float pose_graph_apply_max_frame_relative_translation_change = 0.05f;
    float pose_graph_apply_max_frame_relative_rotation_change_deg = 5.0f;
    std::string pose_graph_apply_loop_selection_policy = "greedy_sequential_gate";
    int pose_graph_apply_loop_selection_max_candidates = 16;
    int pose_graph_apply_pcm_min_clique_size = 2;
    int pose_graph_apply_pcm_max_trial_cliques = 32;
    int pose_graph_apply_pcm_event_cluster_radius = 1;
    int pose_graph_apply_pcm_min_distinct_events = 2;
    int pose_graph_apply_pcm_min_long_span = 0;
    bool pose_graph_apply_pcm_prefer_long_span = false;
    bool pose_graph_apply_pcm_scale_threshold_by_cycle_length = true;
    float pose_graph_apply_pcm_translation_threshold = 0.02f;
    float pose_graph_apply_pcm_rotation_threshold_deg = 2.0f;
    bool pose_graph_apply_skip_repeated_loop_pairs = true;
    int pose_graph_apply_repeat_min_new_nodes = 1;
    bool pose_graph_apply_observation_gate = true;
    bool pose_graph_apply_observation_gate_in_selection = false;
    int pose_graph_apply_observation_gate_max_samples = 4;
    int pose_graph_apply_observation_gate_min_samples = 3;
    std::string pose_graph_apply_observation_gate_sample_policy = "resident_recent";
    int pose_graph_apply_observation_max_render_maps_per_sample = 0;
    int pose_graph_apply_observation_max_pinned_maps = 0;
    bool pose_graph_apply_observation_prefetch_loop_endpoints = false;
    int pose_graph_apply_observation_prefetch_max_maps = 0;
    int pose_graph_apply_observation_prefetch_pin_ttl_frames = 0;
    bool pose_graph_apply_observation_defer_nonresident = false;
    int pose_graph_apply_observation_defer_retry_frames = 120;
    bool pose_graph_apply_observation_pin_deferred_maps = true;
    bool pose_graph_apply_observation_defer_stops_scale_backoff = true;
    bool pose_graph_apply_observation_terminal_sync_page_in = false;
    bool pose_graph_apply_observation_skip_resident_trim = false;
    std::string pose_graph_apply_observation_gate_tolerance_policy = "either";
    float pose_graph_apply_observation_max_color_mse_ratio = 1.02f;
    float pose_graph_apply_observation_max_color_mse_increase = 0.0005f;
    float pose_graph_apply_observation_max_depth_rmse_ratio = 1.05f;
    float pose_graph_apply_observation_max_depth_rmse_increase = 0.01f;
    float pose_graph_apply_observation_depth_rmse_ratio_floor = 0.01f;
    float pose_graph_apply_observation_max_depth_rmse_absolute_increase = 0.05f;
    float pose_graph_apply_observation_max_valid_fraction_drop = 0.30f;
    int pose_graph_apply_max_transform_updates_per_frame = 0;
    bool pose_graph_apply_health_gate = false;
    float pose_graph_apply_health_max_raw_failed_ratio = 0.02f;
    float pose_graph_apply_health_max_raw_non_good_ratio = 0.10f;
    int pose_graph_apply_health_max_raw_failed_streak = 5;
    int pose_graph_apply_health_min_frames = 30;
    int pose_graph_health_raw_frames = 0;
    int pose_graph_health_raw_failed_frames = 0;
    int pose_graph_health_raw_non_good_frames = 0;
    int pose_graph_health_raw_failed_streak = 0;
    int pose_graph_health_max_raw_failed_streak_observed = 0;
    bool pose_graph_online_updates = false;
    int pose_graph_candidate_interval_frames = 30;
    bool pose_graph_run_on_submap_events = true;
    int pose_graph_online_min_submaps = 4;
    int pose_graph_online_min_submap_growth = 2;
    int pose_graph_max_updates_per_run = 0;
    int pose_graph_last_update_frame = -1;
    int pose_graph_last_update_submap_count = 0;
    int pose_graph_online_update_count = 0;
    std::vector<int> pose_graph_deferred_observation_pins;
    std::map<int, int> pose_graph_loop_endpoint_pin_expire_frame;
    int pose_graph_deferred_observation_retry_after_frame = -1;
    int pose_graph_deferred_observation_expire_frame = -1;
    int pose_graph_candidate_scan_cursor = 0;
    bool pose_graph_persistent_loop_pool = true;
    int pose_graph_verified_loop_pool_max = 64;
    bool pose_graph_skip_verified_loop_pairs = false;
    bool pose_graph_correction_applied = false;
    int pose_graph_loop_min_separation = 3;
    int pose_graph_max_scanned_pairs_per_check = 0;
    bool pose_graph_final_candidate_scan = true;
    bool pose_graph_terminal_online_scan = false;
    int pose_graph_final_max_scanned_pairs = 0;
    bool pose_graph_final_rank_all_candidates = true;
    bool pose_graph_incremental_candidate_scan = true;
    int pose_graph_max_candidate_pairs_per_check = 16;
    int pose_graph_candidate_accept_target_per_check = 0;
    int pose_graph_loop_max_candidates = 16;
    std::string pose_graph_loop_candidate_export_policy = "score";
    float pose_graph_loop_candidate_export_span_reserve_fraction = 0.0f;
    int pose_graph_loop_candidate_export_min_span = 0;
    int pose_graph_loop_observation_interval_frames = 10;
    int pose_graph_loop_observation_max_per_submap = 8;
    bool pose_graph_include_keyframes_with_loop_observations = true;
    int pose_graph_keyframe_observation_max_per_submap = 8;
    float pose_graph_loop_max_distance = 0.75f;
    float pose_graph_loop_max_rotation_deg = 60.0f;
    std::string pose_graph_candidate_verification = "keyframe_pose";
    std::string pose_graph_candidate_order = "temporal";
    int pose_graph_span_reserve_pairs = 0;
    float pose_graph_span_reserve_fraction = 0.25f;
    int pose_graph_appearance_max_observation_pairs = 8;
    int pose_graph_appearance_min_orb_matches = 0;
    bool pose_graph_orb_feature_cache = true;
    int pose_graph_orb_feature_cache_limit = 256;
    bool pose_graph_warm_loop_observations = true;
    int pose_graph_warm_max_per_frame = 2;
    bool pose_graph_warm_skip_update_frames = true;
    bool pose_graph_warm_async = true;
    int pose_graph_warm_queue_limit = 32;
    double pose_graph_warm_orb_total_ms = 0.0;
    double pose_graph_warm_orb_detect_ms = 0.0;
    int pose_graph_warm_orb_requests = 0;
    int pose_graph_warm_orb_cache_hits = 0;
    int pose_graph_warm_orb_cache_misses = 0;
    int pose_graph_warm_orb_cache_evictions = 0;
    int pose_graph_warm_orb_extract_calls = 0;
    int pose_graph_warm_orb_frame_count = 0;
    int pose_graph_warm_orb_queued = 0;
    int pose_graph_warm_orb_dropped = 0;
    int pose_graph_warm_orb_deduplicated = 0;
    int pose_graph_warm_orb_completed = 0;
    int pose_graph_warm_orb_queue_peak = 0;
    int pose_graph_orb_max_features = 1000;
    float pose_graph_orb_ratio = 0.75f;
    std::string pose_graph_orb_match_policy = "mutual_ratio";
    int pose_graph_orb_min_quadrants = 3;
    int pose_graph_orb_depth_search_radius = 0;
    bool pose_graph_orb_pnp_fallback = false;
    int pose_graph_orb_pnp_iterations = 100;
    float pose_graph_orb_pnp_reprojection_error = 4.0f;
    float pose_graph_orb_pnp_confidence = 0.99f;
    int pose_graph_bow_word_bits = 12;
    int pose_graph_bow_max_observations_per_submap = 8;
    float pose_graph_bow_min_score = 0.0f;
    bool pose_graph_bow_defer_appearance = true;
    bool pose_graph_bow_histogram_cache = true;
    float pose_graph_verify_max_keyframe_distance = 0.75f;
    float pose_graph_verify_max_keyframe_rotation_deg = 60.0f;
    float pose_graph_verify_max_depth_error = 0.08f;
    float pose_graph_verify_min_depth_overlap = 0.10f;
    int pose_graph_verify_depth_sample_stride = 16;
    int pose_graph_verify_icp_min_inliers = 64;
    float pose_graph_verify_icp_max_rmse = 0.05f;
    float pose_graph_verify_icp_max_translation_delta = 0.25f;
    float pose_graph_verify_icp_max_rotation_delta_deg = 20.0f;
    bool pose_graph_verify_icp_ransac = true;
    int pose_graph_verify_icp_ransac_iterations = 128;
    float pose_graph_verify_icp_ransac_threshold = 0.04f;
    float pose_graph_verify_icp_ransac_min_inlier_ratio = 0.25f;
    bool pose_graph_verify_icp_weighted_kabsch = false;
    bool pose_graph_verify_icp_depth_adaptive_ransac = false;
    float pose_graph_verify_icp_depth_noise_scale = 0.002f;
    float pose_graph_verify_icp_depth_adaptive_threshold_scale = 3.0f;
    float pose_graph_verify_icp_max_score = 0.5f;
    bool pose_graph_verify_icp_drift_aware_delta = false;
    float pose_graph_verify_icp_drift_reference_span = 10.0f;
    float pose_graph_verify_icp_drift_max_scale = 3.0f;
    float pose_graph_verify_icp_drift_score_margin = 0.0f;
    bool pose_graph_verify_orb_aggregate_pairs = false;
    int pose_graph_verify_orb_aggregate_max_pairs = 8;
    int pose_graph_verify_max_observation_pairs = 32;
    std::string pose_graph_verify_pair_selection = "pose";
    int pose_graph_verify_min_observations = 2;
    float pose_graph_verify_observation_max_translation_delta = 0.05f;
    float pose_graph_verify_observation_max_rotation_delta_deg = 5.0f;
    YAML::Node gaussian_model_config;
    std::vector<GaussianSubmap> gaussian_submaps;
    mutable std::map<int, int> submap_render_last_selected_frame;
    mutable int last_render_selection_candidate_count = 0;
    mutable int last_render_selection_truncated_submaps = 0;
    mutable int last_render_selection_truncated_gaussians = 0;
    mutable int last_render_selection_selected_gaussians = 0;
    mutable int last_render_selection_budget_gaussians = 0;
    mutable int last_render_selection_owner_oversized = 0;
    mutable int last_render_selection_sticky_count = 0;
    mutable int last_render_depth_gate_enabled = 0;
    std::map<int, int> frame_owner_submap;
    std::map<int, std::vector<int>> frame_overlap_submaps;
    std::vector<HybridPoseGraphNode> hybrid_pose_graph_nodes;
    std::vector<HybridPoseGraphConstraint> hybrid_pose_graph_sequential_constraints;
    std::vector<HybridPoseGraphConstraint> hybrid_pose_graph_loop_constraints;
    std::vector<HybridPoseGraphConstraint> hybrid_pose_graph_verified_loop_pool;
    std::map<std::pair<int, int>, int> pose_graph_applied_loop_pair_node_count;
    std::map<int, std::vector<Camera>> sdf_loop_observation_cam_lists;
    std::map<int, std::vector<int>> frame_loop_observation_submaps;
    std::map<int, torch::Tensor> frame_local_pose;
    std::map<std::string, std::shared_ptr<PoseGraphOrbFeaturePacket>> pose_graph_orb_feature_packets;
    std::deque<std::string> pose_graph_orb_feature_cache_order;
    std::mutex pose_graph_orb_feature_cache_mutex;
    std::thread pose_graph_orb_warm_worker;
    std::mutex pose_graph_orb_warm_mutex;
    std::condition_variable pose_graph_orb_warm_cv;
    std::deque<PoseGraphOrbWarmRequest> pose_graph_orb_warm_queue;
    std::set<std::string> pose_graph_orb_warm_queued_keys;
    bool pose_graph_orb_warm_worker_running = false;
    bool pose_graph_orb_warm_worker_stop = false;
    std::map<int, PoseGraphBowHistogramPacket> pose_graph_bow_histogram_packets;
    std::deque<PendingPoseGraphTransformUpdate> pose_graph_pending_transform_updates;
    int pose_graph_pending_transform_total = 0;
    int pose_graph_pending_transform_applied = 0;
    int last_logged_tsdf_primary = -1;
    int last_logged_tsdf_count = 0;
    std::ofstream submap_events_log;
    std::ofstream frame_assignments_log;
    std::ofstream submap_stats_log;

    void initializeGaussianSubmaps(const YAML::Node &model_config);

    void maybeCreateGaussianSubmap();

    void createGaussianSubmap(const Camera &start_cam, bool initial, const std::string &reason = "initial");

    int findGaussianSubmapBySdfId(int sdf_submap_id) const;

    Camera cameraInSubmapFrame(const Camera &cam, int sdf_submap_id) const;

    void appendBoundaryObservation(int submap_idx, const Camera &global_or_local_cam, const std::string &reason);

    void appendLoopObservation(int submap_idx, const Camera &global_or_local_cam, const std::string &reason);

    bool appendTrainingObservation(int submap_idx,
                                   const Camera &global_or_local_cam,
                                   const std::string &reason,
                                   bool force);

    Camera poseGraphCameraInSdfSubmapFrame(const Camera &cam, int sdf_submap_id) const;

    void appendSdfLoopObservation(int sdf_submap_id, const Camera &global_or_local_cam, const std::string &reason);

    void warmPoseGraphLoopObservation(int submap_idx, const Camera &local_cam, const std::string &reason);

    void startPoseGraphOrbWarmWorker();

    void stopPoseGraphOrbWarmWorker();

    void poseGraphOrbWarmWorkerLoop();

    bool enqueuePoseGraphOrbWarmRequest(int submap_idx, const Camera &local_cam, const std::string &reason);

    void recordPoseGraphLoopObservations(const std::string &reason);

    void warmCandidateGaussianSubmaps();

    bool extendCandidateGaussianSubmapIfActive(int submap_idx, const std::string &reason);

    void retireInvalidGaussianSubmapOwners(const std::string &context);

    void retireCandidateGaussianSubmap(GaussianSubmap &submap, const std::string &reason);

    void processTsdfTransitionEvents(const std::vector<ITMActiveMapManager::TransitionEvent> &events);

    void syncGaussianSubmapTransformsFromTsdf(const std::string &context);

    void addBoundaryObservationsToPendingSubmaps();

    void includeHandoffSubmapsForRendering(Camera &cam, const std::string &reason);

    void optimizeHandoffSubmaps();

    bool graduateGaussianSubmap(int submap_idx, const std::string &reason);

    void appendVisibleFrozenTrainingObservations(const Camera &global_cam, const std::string &reason);

    void appendCrossBoundaryTrainingObservations(const Camera &global_cam, const std::string &reason);

    bool refineFrozenGaussianSubmap(int submap_idx, const std::string &reason);

    void refineVisibleFrozenGaussianSubmaps(const Camera &global_cam, const std::string &reason);

    void finalizeExpiredHandoffSubmaps();

    void validateHybridSubmaps(const std::string &context) const;

    void applySubmapGlobalToLocalTransform(int sdf_submap_id, const torch::Tensor &global_to_local, const std::string &reason);

    void applyPoseGraphTransformUpdate(const PendingPoseGraphTransformUpdate &update,
                                       const std::string &reason);

    int processPendingPoseGraphTransformUpdates(int frame_id,
                                                int max_updates,
                                                const std::string &reason);

    void flushPendingPoseGraphTransformUpdates(const std::string &reason);

    void runPoseGraphReadinessChecks();

    void runPoseGraphRenderIdentityCheck(const std::vector<Camera> &cams);

    void runPoseGraphRenderSyntheticCheck(const std::vector<Camera> &cams);

    void collectPoseGraphLoopCandidates(nlohmann::json &manifest, const std::filesystem::path &staging_root);

    bool poseGraphRuntimeEnabled() const;

    bool localTsdfSubmapGraphAvailable() const;

    bool localGaussianSubmapGraphAvailable() const;

    int resolvePoseGraphFixedNodeId(
        const std::map<int, torch::Tensor> &original_local_to_global,
        const std::map<int, int> &hybrid_id_by_sdf_id,
        const std::string &context) const;

    void configurePoseGraphEdge(MiniSlamGraph::GraphEdgeSE3 *edge,
                                const HybridPoseGraphConstraint &constraint,
                                const std::string &type,
                                double scalar_weight) const;

    double poseGraphLoopInformationConfidence(const HybridPoseGraphConstraint &constraint) const;

    void buildLivePoseGraphScaffold(nlohmann::json &manifest, const std::string &source, int frame_id);

    void runLivePoseGraphUpdate(const std::filesystem::path &diagnostic_root,
                                const std::string &source,
                                int frame_id);

    void maybeRunOnlinePoseGraphUpdate(int frame_id);

    void armDeferredObservationRetry(const std::vector<int> &sdf_submap_ids, int frame_id);

    void clearDeferredObservationResidencyPins();

    void rememberLoopEndpointResidencyPins(const std::vector<int> &sdf_submap_ids,
                                           int frame_id,
                                           const std::string &reason);

    void refreshPoseGraphExtraResidencyPins(int frame_id);

    void refreshPoseGraphExtraResidencyPins(int frame_id,
                                            const std::vector<int> &transient_pins);

    bool tryRunDeferredObservationRetry(int frame_id);

    void appendPoseGraphUpdateSummary(const std::string &source,
                                      int frame_id,
                                      const std::filesystem::path &diagnostic_root,
                                      const nlohmann::json &manifest) const;

    void runPoseGraphDryRun(const std::filesystem::path &diagnostic_root, nlohmann::json *manifest);

    void runPoseGraphSolveOnly(const std::filesystem::path &diagnostic_root, nlohmann::json *manifest);

    void runPoseGraphApply(const std::filesystem::path &diagnostic_root, nlohmann::json *manifest);

    void ensureGaussianSubmapResident(int submap_idx, const std::string &reason);

    void releaseFrozenGaussianSubmap(int submap_idx, const std::string &reason);

    void enforceGaussianResidency(const std::vector<int> &keep_indices, const std::string &reason);

    bool gaussianSubmapNeedsQualityFinalize(const GaussianSubmap &submap) const;

    bool topUpGaussianSubmapForQualityFinalize(int submap_idx, const std::string &reason);

    bool finalizeGaussianSubmapQuality(int submap_idx, const std::string &reason);

    void finalizeGaussianSubmapQualityAtSequenceEnd();

    void startGaussianOptimizationJob(SLAMGaussianModel &model,
                                      const Camera &global_camera);

    int runGaussianOptimizationJobSlice(int max_iterations);

    void finishGaussianOptimizationJob(bool run_followup);

    void drainGaussianOptimizationJob(const std::string &reason);

    void writeOnlineEndTrajectory(const std::vector<Camera> &cams) const;

    void writeRuntimeResourceSummary(double sequence_wall_ms,
                                     int final_gaussian_count) const;

    std::vector<int> selectSubmapsForCamera(const Camera &cam) const;
    std::vector<int> selectSubmapsForCameraByBounds(const Camera &cam,
                                                    int excluded_submap_idx = -1,
                                                    bool include_owner = true) const;

    void logSubmapStats(int sequence_id, int frame_id, const std::string &stage);

    int resolvedRenderPastFrameMargin() const;

    int resolvedRenderFutureFrameMargin() const;

    bool gaussianSubmapWithinRenderWindow(const GaussianSubmap &submap, int frame_id) const;

    bool gaussianSubmapReadyForInsertion(int &sequence_age, int &sdf_blocks, std::string &reason) const;

    int gaussianSubmapObservedFrames(const GaussianSubmap &submap) const;

    bool candidateGaussianInsertionMature(const GaussianSubmap &submap) const;

    bool candidateGaussianInsertionWarmupAllowed(const GaussianSubmap &submap) const;

    bool gaussianSubmapReadyForRender(int submap_idx, int frame_id, bool is_owner) const;

    bool gaussianSubmapUsableForRender(int submap_idx, int frame_id, bool is_owner) const;

    std::vector<int> selectTsdfRenderSubmapsForCamera(const Camera &cam) const;

    std::vector<int> boundedTsdfResidencyPins(const std::vector<int> &pins,
                                              int max_count = -1) const;

    std::vector<int> selectGaussianSubmapKeyframeIndices(const GaussianSubmap &submap,
                                                         int max_count,
                                                         const std::string &policy) const;

    int gaussianSubmapBudgetCap(const GaussianSubmap &submap, int sdf_blocks) const;

    TensorDict renderLocalGaussianSubmaps(const Camera &global_cam,
                                          const std::vector<int> &submap_indices,
                                          const torch::Tensor &raycast_depth,
                                          const torch::Tensor &raycast_color,
                                          const std::vector<int> &tsdf_residency_pins = {});

    int cameraFrameId(const Camera &cam) const;

    std::string poseGraphCameraCacheKey(const Camera &cam, const std::string &kind) const;

    std::shared_ptr<PoseGraphOrbFeaturePacket> getOrCreatePoseGraphOrbFeaturePacket(
        const Camera &cam,
        int &cache_hits,
        int &cache_misses,
        int &cache_evictions,
        double &image_load_ms,
        double &feature_image_ms,
        double &feature_detect_ms,
        double &feature_total_ms,
        int &extract_calls);

    int inferOwnerSdfSubmapForFrame(int frame_id, int fallback_owner = -1) const;

    std::vector<int> inferOverlapSdfSubmapsForFrame(int frame_id, int owner_sdf_id) const;

    torch::Tensor correctedGlobalPoseForCamera(const Camera &cam) const;

    GaussianSubmap &activeSubmap();

    SLAMGaussianModel &activeGaussianModel(SLAMGaussianModel &fallback_model);

    std::deque<Camera> &activeLocalFrameWindow();

    std::deque<TensorDict> &activeLocalFrameRaycastWindow();

    std::map<std::string, std::vector<float>> &activeKeyframeLossDict();

    std::vector<Camera> &activeKeyframeList();

    std::map<std::string, Camera> &activeKeyframeDict();

    std::vector<Camera> &activeOptCamList();

    std::vector<TensorDict> &activeOptRaycastList();
};

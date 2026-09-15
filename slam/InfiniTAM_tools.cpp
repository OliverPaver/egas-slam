#include "InfiniTAM_tools.h"
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <mutex>
#include <thread>

namespace
{
namespace fs = std::filesystem;

template <typename T>
T configValue(const YAML::Node &node, const char *key, const T &default_value)
{
    return node && node[key] ? node[key].as<T>() : default_value;
}

const char *compiledHashProfile()
{
    return GPS_SLAM_COMPILED_HASH_PROFILE;
}

int residentMapCount(const YAML::Node &multi_cfg)
{
    if (!multi_cfg || !multi_cfg["resident_gpu_maps"])
        return 3;
    const std::string value = multi_cfg["resident_gpu_maps"].as<std::string>();
    if (value != "auto")
    {
        try
        {
            size_t parsed = 0;
            const int count = std::stoi(value, &parsed);
            if (parsed != value.size() || count < 1)
                throw std::runtime_error("resident map count must be positive");
            return count;
        }
        catch (const std::exception &)
        {
            throw std::runtime_error(
                "PIPE.TSDF.multi.resident_gpu_maps must be 'auto' or a positive integer");
        }
    }

    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        return 3;
    const size_t reserve_mb = configValue<size_t>(multi_cfg, "gpu_reserve_mb", 3000);
    const size_t free_mb = free_bytes / (1024 * 1024);
    const size_t map_budget_mb = free_mb > reserve_mb ? free_mb - reserve_mb : 0;
    const std::string profile = compiledHashProfile();
    const size_t map_allowance_mb =
        profile == "compact_64k" ? 512 :
        profile == "medium_128k" ? 1536 :
        4096;
    // The allowance includes render/tracking state and temporary snapshots.
    // Compact maps are measured at roughly 260 MB resident in the validation
    // diagnostics, so 512 MB keeps headroom without forcing avoidable paging
    // stalls on 8 GB GPUs. Medium maps still need more room per resident map.
    return std::clamp<int>(int(map_budget_mb / map_allowance_mb), 2, 8);
}

ITMLibSettings::FailureMode trackingFailureMode(const std::string &mode)
{
    if (mode == "legacy_ignore") return ITMLibSettings::FAILUREMODE_IGNORE;
    if (mode == "contain_failed") return ITMLibSettings::FAILUREMODE_CONTAIN_FAILED;
    if (mode == "stop_integration") return ITMLibSettings::FAILUREMODE_STOP_INTEGRATION;
    if (mode == "relocalise") return ITMLibSettings::FAILUREMODE_RELOCALISE;
    throw std::runtime_error(
        "Unknown PIPE.TSDF.tracking_failure_mode=" + mode +
        "; expected legacy_ignore, contain_failed, stop_integration, or relocalise");
}

struct OnDemandPrefetchState
{
    std::vector<Camera> *cams;
    bool enabled;

    std::thread worker;
    std::mutex mutex;
    std::condition_variable request_cv;
    std::condition_variable completed_cv;
    bool stop = false;
    bool request_pending = false;
    bool request_running = false;
    int requested_frame = -1;
    int completed_frame = -1;
    std::exception_ptr worker_error;

    OnDemandPrefetchState(std::vector<Camera> *cams, bool enabled) : cams(cams), enabled(enabled)
    {
        if (enabled)
            worker = std::thread([this]() { workerLoop(); });
    }

    ~OnDemandPrefetchState()
    {
        if (!worker.joinable())
            return;
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
            request_pending = false;
        }
        request_cv.notify_one();
        worker.join();
    }

    void workerLoop()
    {
        while (true)
        {
            int frame_idx = -1;
            {
                std::unique_lock<std::mutex> lock(mutex);
                request_cv.wait(lock, [this]() { return stop || request_pending; });
                if (stop)
                    return;
                frame_idx = requested_frame;
                request_pending = false;
                request_running = true;
            }

            std::exception_ptr error;
            try
            {
                (*cams)[frame_idx].ensureLoaded();
            }
            catch (...)
            {
                error = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                completed_frame = frame_idx;
                request_running = false;
                if (error)
                    worker_error = error;
            }
            completed_cv.notify_all();
        }
    }

    void prepare(int frame_idx)
    {
        if (!enabled)
        {
            (*cams)[frame_idx].ensureLoaded();
            return;
        }

        std::exception_ptr error;
        bool prefetched = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            if ((request_pending || request_running) && requested_frame == frame_idx)
                completed_cv.wait(lock, [this]() {
                    return worker_error || (!request_pending && !request_running);
                });
            else if (request_pending || request_running)
                completed_cv.wait(lock, [this]() {
                    return worker_error || (!request_pending && !request_running);
                });
            error = worker_error;
            const Camera &camera = (*cams)[frame_idx];
            prefetched = completed_frame == frame_idx && camera.image.defined() &&
                         (!camera.has_depth || camera.depth.defined());
        }
        if (error)
            std::rethrow_exception(error);
        if (!prefetched)
            (*cams)[frame_idx].ensureLoaded();
    }

    void prefetch(int frame_idx)
    {
        if (!enabled)
            return;
        if (frame_idx < 0 || frame_idx >= int(cams->size()))
            return;

        std::unique_lock<std::mutex> lock(mutex);
        completed_cv.wait(lock, [this]() {
            return worker_error || (!request_pending && !request_running);
        });
        if (worker_error)
            std::rethrow_exception(worker_error);
        requested_frame = frame_idx;
        completed_frame = -1;
        request_pending = true;
        lock.unlock();
        request_cv.notify_one();
    }
};
}

void validateTsdfConfig(const YAML::Node &config)
{
    const std::string input_mode = configValue<std::string>(config, "input_mode", "preload");
    if (input_mode != "preload" && input_mode != "on_demand")
        throw std::runtime_error("Unknown PIPE.TSDF.input_mode=" + input_mode + "; expected preload or on_demand");
    const std::string tensor_bridge = configValue<std::string>(config, "input_tensor_bridge", "direct");
    if (tensor_bridge != "direct" && tensor_bridge != "opencv_legacy")
        throw std::runtime_error(
            "Unknown PIPE.TSDF.input_tensor_bridge=" + tensor_bridge +
            "; expected direct or opencv_legacy");
    if (configValue<bool>(config, "input_tensor_bridge_validate", false) &&
        (input_mode != "on_demand" || tensor_bridge != "direct"))
        throw std::runtime_error(
            "PIPE.TSDF.input_tensor_bridge_validate=true requires "
            "input_mode=on_demand and input_tensor_bridge=direct");
    const std::string backend = configValue<std::string>(config, "engine", "basic");
    if (backend != "basic" && backend != "multi")
        throw std::runtime_error("Unknown PIPE.TSDF.engine: " + backend);
    if (config["expected_hash_profile"] &&
        config["expected_hash_profile"].as<std::string>() != compiledHashProfile())
        throw std::runtime_error("PIPE.TSDF.expected_hash_profile does not match the compiled binary");
    if (configValue<bool>(config, "basic_capacity_diagnostics", false) && backend != "basic")
        throw std::runtime_error("basic_capacity_diagnostics requires PIPE.TSDF.engine=basic");
    const std::string failure_mode =
        configValue<std::string>(config, "tracking_failure_mode", "legacy_ignore");
    (void)trackingFailureMode(failure_mode);
    if (backend == "multi" && failure_mode == "relocalise")
        throw std::runtime_error(
            "PIPE.TSDF.tracking_failure_mode=relocalise is not supported by the multi engine; "
            "use contain_failed or stop_integration for contained operation");
    if (config["tracker_config"] && config["tracker_config"].as<std::string>().empty())
        throw std::runtime_error("PIPE.TSDF.tracker_config must be non-empty when specified");
    const std::string tracking_profile =
        configValue<std::string>(config, "tracking_profile", "legacy");
    if (tracking_profile != "legacy" &&
        tracking_profile != "efficient" &&
        tracking_profile != "robust_rpe")
        throw std::runtime_error(
            "Unknown PIPE.TSDF.tracking_profile=" + tracking_profile +
            "; expected legacy, efficient, or robust_rpe");
    if (config["tracker_config"] && tracking_profile != "legacy")
        throw std::runtime_error(
            "PIPE.TSDF.tracker_config and a non-legacy tracking_profile are mutually exclusive");
    if (configValue<int>(config, "raw_tracking_failure_grace_frames", 0) < 0)
        throw std::runtime_error("PIPE.TSDF.raw_tracking_failure_grace_frames must be non-negative");
    const int basic_debug_start = configValue<int>(config, "basic_debug_tracking_start_frame", -1);
    const int basic_debug_end = configValue<int>(config, "basic_debug_tracking_end_frame", -1);
    if ((basic_debug_start >= 0 || basic_debug_end >= 0) &&
        (basic_debug_start < 0 || basic_debug_end < basic_debug_start))
        throw std::runtime_error(
            "PIPE.TSDF.basic_debug_tracking_start_frame/end_frame must be unset or a valid inclusive range");
    if (backend == "multi")
    {
#ifdef GPS_SLAM_CAPACITY_CONTROL
        throw std::runtime_error("Capacity-control binaries support only PIPE.TSDF.engine=basic");
#endif
        const YAML::Node multi_cfg = config["multi"];
        const std::string boundary_policy = configValue<std::string>(multi_cfg, "boundary_policy", "visibility");
        if (boundary_policy != "visibility" && boundary_policy != "capacity")
            throw std::runtime_error(
                "Unsupported PIPE.TSDF.multi.boundary_policy=" + boundary_policy +
                "; supported values are visibility and capacity");
        const std::string sequential_transform_policy =
            configValue<std::string>(multi_cfg, "sequential_transform_policy", "validated_overlap");
        if (sequential_transform_policy != "validated_overlap" &&
            sequential_transform_policy != "overlap" &&
            sequential_transform_policy != "initial")
            throw std::runtime_error(
                "Unsupported PIPE.TSDF.multi.sequential_transform_policy=" + sequential_transform_policy +
                "; supported values are validated_overlap, overlap, and initial");
        if (configValue<float>(multi_cfg, "sequential_transform_max_translation", 0.03f) < 0.0f)
            throw std::runtime_error("PIPE.TSDF.multi.sequential_transform_max_translation must be non-negative");
        if (configValue<float>(multi_cfg, "sequential_transform_max_rotation_deg", 3.0f) < 0.0f)
            throw std::runtime_error("PIPE.TSDF.multi.sequential_transform_max_rotation_deg must be non-negative");
        if (configValue<int>(multi_cfg, "invalid_transform_restart_attempts", 0) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.invalid_transform_restart_attempts must be non-negative");
        const std::string handoff_pose_policy =
            configValue<std::string>(multi_cfg, "handoff_pose_policy", "candidate");
        if (handoff_pose_policy != "preserve_global" &&
            handoff_pose_policy != "candidate" &&
            handoff_pose_policy != "validated_candidate")
            throw std::runtime_error(
                "Unsupported PIPE.TSDF.multi.handoff_pose_policy=" + handoff_pose_policy +
                "; supported values are preserve_global, candidate, and validated_candidate");
        const std::string new_map_handoff_policy =
            configValue<std::string>(multi_cfg, "new_map_handoff_policy", "visibility");
        if (new_map_handoff_policy != "visibility" &&
            new_map_handoff_policy != "accepted" &&
            new_map_handoff_policy != "validated_visibility")
            throw std::runtime_error(
                "Unsupported PIPE.TSDF.multi.new_map_handoff_policy=" + new_map_handoff_policy +
                "; supported values are visibility, accepted, and validated_visibility");
        const float new_map_handoff_min_visible_ratio =
            configValue<float>(multi_cfg, "new_map_handoff_min_visible_ratio", 0.0f);
        if (new_map_handoff_min_visible_ratio < 0.0f || new_map_handoff_min_visible_ratio > 1.0f)
            throw std::runtime_error("PIPE.TSDF.multi.new_map_handoff_min_visible_ratio must be in [0, 1]");
        const float new_map_handoff_min_visible_margin =
            configValue<float>(multi_cfg, "new_map_handoff_min_visible_margin", 0.0f);
	        if (new_map_handoff_min_visible_margin < 0.0f || new_map_handoff_min_visible_margin > 1.0f)
	            throw std::runtime_error("PIPE.TSDF.multi.new_map_handoff_min_visible_margin must be in [0, 1]");
	        if (configValue<int>(multi_cfg, "new_map_handoff_min_accepts", 1) < 0)
	            throw std::runtime_error("PIPE.TSDF.multi.new_map_handoff_min_accepts must be non-negative");
	        if (configValue<int>(multi_cfg, "new_map_handoff_emergency_min_accepts", -1) < -1)
	            throw std::runtime_error("PIPE.TSDF.multi.new_map_handoff_emergency_min_accepts must be -1 or non-negative");
	        const std::string candidate_recovery_policy =
	            configValue<std::string>(multi_cfg, "candidate_recovery_policy", "disabled");
        if (candidate_recovery_policy != "disabled" &&
            candidate_recovery_policy != "keep_if_visible")
            throw std::runtime_error(
                "Unsupported PIPE.TSDF.multi.candidate_recovery_policy=" + candidate_recovery_policy +
                "; supported values are disabled and keep_if_visible");
        const float candidate_recovery_min_visible_ratio =
            configValue<float>(multi_cfg, "candidate_recovery_min_visible_ratio", 0.3f);
        if (candidate_recovery_min_visible_ratio < 0.0f || candidate_recovery_min_visible_ratio > 1.0f)
            throw std::runtime_error("PIPE.TSDF.multi.candidate_recovery_min_visible_ratio must be in [0, 1]");
        if (configValue<int>(multi_cfg, "candidate_recovery_min_blocks", 1000) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.candidate_recovery_min_blocks must be non-negative");
        if (configValue<int>(multi_cfg, "candidate_recovery_max_attempts", 2) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.candidate_recovery_max_attempts must be non-negative");
        const std::string primary_failure_handoff_policy =
            configValue<std::string>(multi_cfg, "primary_failure_handoff_policy", "disabled");
        if (primary_failure_handoff_policy != "disabled" &&
            primary_failure_handoff_policy != "promote_visible_candidate")
            throw std::runtime_error(
                "Unsupported PIPE.TSDF.multi.primary_failure_handoff_policy=" + primary_failure_handoff_policy +
                "; supported values are disabled and promote_visible_candidate");
        const bool require_handoff_pose_continuity = configValue<bool>(
            multi_cfg, "primary_failure_handoff_require_pose_continuity", false);
        if (primary_failure_handoff_policy == "promote_visible_candidate" &&
            require_handoff_pose_continuity &&
            handoff_pose_policy != "preserve_global" &&
            handoff_pose_policy != "validated_candidate")
            throw std::runtime_error(
                "PIPE.TSDF.multi.primary_failure_handoff_require_pose_continuity=true requires "
                "handoff_pose_policy=preserve_global or validated_candidate");
        if (configValue<int>(multi_cfg, "primary_failure_handoff_min_raw_failed_frames", 30) < 1)
            throw std::runtime_error("PIPE.TSDF.multi.primary_failure_handoff_min_raw_failed_frames must be at least 1");
        const std::string primary_failure_handoff_raw_gate =
            configValue<std::string>(multi_cfg, "primary_failure_handoff_raw_gate", "failed_only");
        if (primary_failure_handoff_raw_gate != "failed_only" &&
            primary_failure_handoff_raw_gate != "failed_or_weak")
            throw std::runtime_error(
                "Unsupported PIPE.TSDF.multi.primary_failure_handoff_raw_gate=" + primary_failure_handoff_raw_gate +
                "; supported values are failed_only and failed_or_weak");
        if (configValue<int>(multi_cfg, "primary_failure_handoff_min_raw_weak_frames", 0) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.primary_failure_handoff_min_raw_weak_frames must be non-negative");
        const float primary_failure_handoff_min_candidate_visible_ratio =
            configValue<float>(multi_cfg, "primary_failure_handoff_min_candidate_visible_ratio", 0.5f);
        if (primary_failure_handoff_min_candidate_visible_ratio < 0.0f ||
            primary_failure_handoff_min_candidate_visible_ratio > 1.0f)
            throw std::runtime_error("PIPE.TSDF.multi.primary_failure_handoff_min_candidate_visible_ratio must be in [0, 1]");
        if (configValue<int>(multi_cfg, "primary_failure_handoff_min_candidate_blocks", 1000) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.primary_failure_handoff_min_candidate_blocks must be non-negative");
        if (configValue<int>(multi_cfg, "primary_failure_handoff_min_candidate_good_frames", 10) < 1)
            throw std::runtime_error("PIPE.TSDF.multi.primary_failure_handoff_min_candidate_good_frames must be at least 1");
        if (configValue<int>(multi_cfg, "primary_failure_handoff_min_overlap_frames", 0) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.primary_failure_handoff_min_overlap_frames must be non-negative");
        const std::string tracking_boundary_policy =
            configValue<std::string>(multi_cfg, "tracking_boundary_policy", "disabled");
        if (tracking_boundary_policy != "disabled" && tracking_boundary_policy != "raw_weak_streak")
            throw std::runtime_error(
                "Unsupported PIPE.TSDF.multi.tracking_boundary_policy=" + tracking_boundary_policy +
                "; supported values are disabled and raw_weak_streak");
        const std::string tracking_boundary_handoff_policy =
            configValue<std::string>(multi_cfg, "tracking_boundary_handoff_policy", "weak_or_failed");
        if (tracking_boundary_handoff_policy != "weak_or_failed" &&
            tracking_boundary_handoff_policy != "failed_only" &&
            tracking_boundary_handoff_policy != "never")
            throw std::runtime_error(
                "Unsupported PIPE.TSDF.multi.tracking_boundary_handoff_policy=" + tracking_boundary_handoff_policy +
                "; supported values are weak_or_failed, failed_only, and never");
        if (configValue<int>(multi_cfg, "raw_weak_boundary_frames", 3) < 1)
            throw std::runtime_error("PIPE.TSDF.multi.raw_weak_boundary_frames must be at least 1");
        if (configValue<int>(multi_cfg, "raw_failed_boundary_frames", 1) < 1)
            throw std::runtime_error("PIPE.TSDF.multi.raw_failed_boundary_frames must be at least 1");
        if (configValue<int>(multi_cfg, "tracking_boundary_min_primary_frames", 20) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.tracking_boundary_min_primary_frames must be non-negative");
        if (configValue<int>(multi_cfg, "tracking_boundary_cooldown_frames", 20) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.tracking_boundary_cooldown_frames must be non-negative");
        if (configValue<int>(multi_cfg, "emergency_overlap_frames", 0) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.emergency_overlap_frames must be non-negative");
        if (configValue<int>(multi_cfg, "emergency_overlap_frames", 0) > configValue<int>(multi_cfg, "overlap_frames", 40))
            throw std::runtime_error("PIPE.TSDF.multi.emergency_overlap_frames cannot exceed overlap_frames");
        if (configValue<int>(multi_cfg, "max_render_maps", 4) < 1)
            throw std::runtime_error("PIPE.TSDF.multi.max_render_maps must be at least 1");
        const float max_block_occupancy = configValue<float>(multi_cfg, "max_block_occupancy", 0.8f);
        const float new_map_block_occupancy = configValue<float>(multi_cfg, "new_map_block_occupancy", 0.7f);
        if (max_block_occupancy < 0.0f || max_block_occupancy > 1.0f)
            throw std::runtime_error("PIPE.TSDF.multi.max_block_occupancy must be in [0, 1]");
        if (new_map_block_occupancy < 0.0f || new_map_block_occupancy > 1.0f)
            throw std::runtime_error("PIPE.TSDF.multi.new_map_block_occupancy must be in [0, 1]");
        if (max_block_occupancy > 0.0f && new_map_block_occupancy > max_block_occupancy)
            throw std::runtime_error("PIPE.TSDF.multi.new_map_block_occupancy must be <= max_block_occupancy");
        if (configValue<int>(multi_cfg, "fusion_reserve_blocks", 0) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.fusion_reserve_blocks must be non-negative");
        if (configValue<int>(multi_cfg, "occupancy_growth_lookahead_frames", 0) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.occupancy_growth_lookahead_frames must be non-negative");
        if (configValue<int>(multi_cfg, "occupancy_growth_min_primary_frames", 20) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.occupancy_growth_min_primary_frames must be non-negative");
        if (configValue<int>(multi_cfg, "occupancy_growth_sample_frames", 5) < 1)
            throw std::runtime_error("PIPE.TSDF.multi.occupancy_growth_sample_frames must be at least 1");
        const std::string occupancy_threshold_policy =
            configValue<std::string>(multi_cfg, "occupancy_threshold_policy", "fixed");
        if (occupancy_threshold_policy != "fixed" && occupancy_threshold_policy != "adaptive_longspan")
            throw std::runtime_error(
                "Unsupported PIPE.TSDF.multi.occupancy_threshold_policy=" + occupancy_threshold_policy +
                "; supported values are fixed and adaptive_longspan");
        const float adaptive_new_map_occupancy =
            configValue<float>(multi_cfg, "adaptive_new_map_block_occupancy", new_map_block_occupancy);
        const float adaptive_max_occupancy =
            configValue<float>(multi_cfg, "adaptive_max_block_occupancy", max_block_occupancy);
        if (adaptive_new_map_occupancy < 0.0f || adaptive_new_map_occupancy > 1.0f)
            throw std::runtime_error("PIPE.TSDF.multi.adaptive_new_map_block_occupancy must be in [0, 1]");
        if (adaptive_max_occupancy < 0.0f || adaptive_max_occupancy > 1.0f)
            throw std::runtime_error("PIPE.TSDF.multi.adaptive_max_block_occupancy must be in [0, 1]");
        if (adaptive_max_occupancy > 0.0f && adaptive_new_map_occupancy > adaptive_max_occupancy)
            throw std::runtime_error("PIPE.TSDF.multi.adaptive_new_map_block_occupancy must be <= adaptive_max_block_occupancy");
        if (configValue<int>(multi_cfg, "adaptive_occupancy_min_primary_frames", 0) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.adaptive_occupancy_min_primary_frames must be non-negative");
        if (configValue<int>(multi_cfg, "adaptive_occupancy_min_processed_frames", 0) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.adaptive_occupancy_min_processed_frames must be non-negative");
        if (configValue<int>(multi_cfg, "adaptive_occupancy_max_raw_weak_streak", 0) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.adaptive_occupancy_max_raw_weak_streak must be non-negative");
        if (configValue<int>(multi_cfg, "adaptive_occupancy_max_raw_failed_streak", 0) < 0)
            throw std::runtime_error("PIPE.TSDF.multi.adaptive_occupancy_max_raw_failed_streak must be non-negative");
        (void)configValue<bool>(multi_cfg, "skip_primary_fusion_on_capacity_handoff", false);
        (void)configValue<bool>(multi_cfg, "allow_visibility_handoff", true);
        if (configValue<bool>(multi_cfg, "paging", false))
        {
            if (residentMapCount(multi_cfg) < 2)
                throw std::runtime_error("PIPE.TSDF.multi.resident_gpu_maps must be at least 2 for primary/candidate overlap");
            if (std::string(compiledHashProfile()) == "legacy_large")
                throw std::runtime_error("PIPE.TSDF.multi.paging requires compact_64k or medium_128k; legacy_large maps are too large for bounded paging");
            const std::string paging_storage = configValue<std::string>(multi_cfg, "paging_storage", "disk");
            if (paging_storage != "disk" && paging_storage != "cpu")
                throw std::runtime_error("PIPE.TSDF.multi.paging_storage must be 'disk' or 'cpu'");
            if (configValue<int>(multi_cfg, "max_pending_page_snapshots", 1) < 0)
                throw std::runtime_error("PIPE.TSDF.multi.max_pending_page_snapshots must be non-negative");
            if (configValue<int>(multi_cfg, "max_pending_page_snapshot_mb", 0) < 0)
                throw std::runtime_error("PIPE.TSDF.multi.max_pending_page_snapshot_mb must be non-negative");
            if (configValue<int>(multi_cfg, "max_pending_page_ins", 1) < 0)
                throw std::runtime_error("PIPE.TSDF.multi.max_pending_page_ins must be non-negative");
            if (configValue<int>(multi_cfg, "max_pending_page_in_mb", 0) < 0)
                throw std::runtime_error("PIPE.TSDF.multi.max_pending_page_in_mb must be non-negative");
            if (configValue<int>(multi_cfg, "cpu_page_cache_limit_mb", 0) < 0)
                throw std::runtime_error("PIPE.TSDF.multi.cpu_page_cache_limit_mb must be non-negative");
        }
        const std::string compiled_hash_profile = compiledHashProfile();
        const std::string hash_profile = configValue<std::string>(multi_cfg, "hash_profile", compiled_hash_profile);
        if (hash_profile != compiled_hash_profile)
            throw std::runtime_error("PIPE.TSDF.multi.hash_profile=" + hash_profile +
                                     " does not match this binary's build-time profile " + compiled_hash_profile);
    }
}

TsdfEngineFacade::TsdfEngineFacade(CLIEngine *cli_engine, ITMMainEngine *main_engine, const std::string &backend)
    : cli_engine(cli_engine), main_engine(main_engine), backend(backend)
{
    basic_engine = backend == "basic" ? static_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(main_engine) : nullptr;
    multi_engine = backend == "multi" ? static_cast<ITMMultiEngine<ITMVoxel, ITMVoxelIndex> *>(main_engine) : nullptr;
    if (!basic_engine && !multi_engine) throw std::runtime_error("Unsupported TSDF engine type");
}

bool TsdfEngineFacade::processFrame()
{
    const bool processed = cli_engine->ProcessFrame();
    if (processed)
    {
        last_tracking_result = cli_engine->GetLastTrackingResult();
		if (multi_engine)
		{
			last_raw_tracking_result = multi_engine->getLastPrimaryRawTrackingResult();
			last_policy_tracking_result = multi_engine->getLastPrimaryPolicyTrackingResult();
			last_pose_rollback = multi_engine->getLastPrimaryPoseRollback();
			last_fusion_suppressed = multi_engine->getLastPrimaryFusionSuppressed();
		}
		else
		{
			last_raw_tracking_result = basic_engine->getLastRawTrackingResult();
			last_policy_tracking_result = basic_engine->getLastPolicyTrackingResult();
			last_pose_rollback = basic_engine->getLastPoseRollback();
			last_fusion_suppressed = basic_engine->getLastFusionSuppressed();
		}
        current_frame_no++;
    }
    return processed;
}

float TsdfEngineFacade::getVoxelSize() const
{
    return basic_engine ? basic_engine->getVoxelSize() : multi_engine->getVoxelSize();
}

ORUtils::Matrix4<float> TsdfEngineFacade::getGlobalCameraPose() const
{
    return basic_engine ? basic_engine->GetTrackingState()->pose_d->GetInvM() : multi_engine->getGlobalCameraPose();
}

int TsdfEngineFacade::getPrimarySubmapId() const { return multi_engine ? multi_engine->findPrimaryLocalMapIdx() : 0; }
int TsdfEngineFacade::getSubmapCount() const { return multi_engine ? (int)multi_engine->getLocalMapCount() : 1; }
std::vector<int> TsdfEngineFacade::getActiveSubmapIds() const { return multi_engine ? multi_engine->getActiveLocalMapIndices() : std::vector<int>{0}; }
std::vector<ITMActiveMapManager::TransitionEvent> TsdfEngineFacade::consumeRecentTransitionEvents() const
{
    return multi_engine ? multi_engine->consumeRecentTransitionEvents()
                        : std::vector<ITMActiveMapManager::TransitionEvent>{};
}
ORUtils::Matrix4<float> TsdfEngineFacade::getGlobalToLocal(int submap_id) const
{
    if (multi_engine) return multi_engine->getLocalMapGlobalToLocal(submap_id).GetM();
    ORUtils::Matrix4<float> identity;
    identity.setIdentity();
    return identity;
}
void TsdfEngineFacade::setGlobalToLocal(int submap_id, const ORUtils::Matrix4<float> &global_to_local) const
{
    if (!multi_engine)
        throw std::runtime_error("TSDF submap transform updates require MultiEngine");
    if (submap_id < 0 || submap_id >= getSubmapCount())
        throw std::runtime_error("Invalid TSDF submap transform update ID: " + std::to_string(submap_id));
    multi_engine->setLocalMapGlobalToLocal(submap_id, ORUtils::SE3Pose(global_to_local));
}
int TsdfEngineFacade::getSubmapBlockCount(int submap_id) const { return multi_engine ? multi_engine->getLocalMapBlockCount(submap_id) : -1; }
int TsdfEngineFacade::getSubmapCapacity(int submap_id) const { return multi_engine ? multi_engine->getLocalMapCapacity(submap_id) : -1; }
TsdfSubmapBounds TsdfEngineFacade::getSubmapLocalBounds(int submap_id) const
{
    TsdfSubmapBounds bounds;
    if (!multi_engine)
        return bounds;
    bounds.valid = multi_engine->getLocalMapBlockBounds(submap_id, bounds.local_min, bounds.local_max);
    return bounds;
}
bool TsdfEngineFacade::isSubmapResident(int submap_id) const { return multi_engine ? multi_engine->isLocalMapResident(submap_id) : true; }
int TsdfEngineFacade::getResidentSubmapCount() const { return multi_engine ? multi_engine->getResidentLocalMapCount() : 1; }
bool TsdfEngineFacade::isPagingEnabled() const { return multi_engine && multi_engine->isPagingEnabled(); }

void TsdfEngineFacade::runRaycast(ORUtils::SE3Pose *global_pose, ITMIntrinsics *intrinsics, int local_map_id)
{
    if (basic_engine) basic_engine->runRaycast(global_pose, intrinsics);
    else multi_engine->runRaycast(global_pose, intrinsics, local_map_id);
}
void TsdfEngineFacade::prefetchRaycastMaps(const std::vector<int> &local_map_ids)
{
    if (multi_engine) multi_engine->prefetchRaycastMaps(local_map_ids);
}
bool TsdfEngineFacade::tryAdoptPrefetchedRaycastMaps(const std::vector<int> &local_map_ids)
{
    return multi_engine ? multi_engine->tryAdoptPrefetchedRaycastMaps(local_map_ids) : true;
}
void TsdfEngineFacade::setExtraResidencyPins(const std::vector<int> &local_map_ids)
{
    if (multi_engine) multi_engine->setExtraResidencyPins(local_map_ids);
}
void TsdfEngineFacade::trimResidency(int pinned_local_map_id)
{
    if (multi_engine) multi_engine->trimResidency(pinned_local_map_id);
}
void TsdfEngineFacade::trimResidency(const std::vector<int> &required_local_map_ids)
{
    if (multi_engine) multi_engine->trimResidency(required_local_map_ids);
}
void TsdfEngineFacade::synchronizePaging()
{
    if (multi_engine) multi_engine->synchronizePaging();
}

ITMUChar4Image *TsdfEngineFacade::getFreeImage() const { return basic_engine ? basic_engine->GetFreeImage() : multi_engine->GetFreeImage(); }
ITMFloat4Image *TsdfEngineFacade::getFreeVertex() const { return basic_engine ? basic_engine->GetFreeVertex() : multi_engine->GetFreeVertex(); }
void TsdfEngineFacade::saveMesh(const std::string &filename) const { main_engine->SaveSceneToMesh(filename.c_str()); }
void TsdfEngineFacade::save(const std::string &directory) const
{
    fs::path target(directory);
    if (target.filename().empty())
        target = target.parent_path();
    const fs::path staging = target.string() + ".tmp";
    const fs::path backup = target.string() + ".bak";
    fs::remove_all(staging);
    if (!fs::create_directories(staging))
        throw std::runtime_error("Could not create staged TSDF checkpoint directory: " + staging.string());
    const std::string staged_directory = staging.string() + "/";
    if (basic_engine) basic_engine->SaveToFile(staged_directory);
    else multi_engine->SaveToFile(staged_directory);

    const fs::path required_file = basic_engine ? staging / "Scene" / "hash.dat" : staging / "multi_engine.bin";
    if (!fs::exists(required_file))
        throw std::runtime_error("Staged TSDF checkpoint is incomplete: " + required_file.string());

    fs::remove_all(backup);
    if (fs::exists(target))
        fs::rename(target, backup);
    try
    {
        fs::rename(staging, target);
    }
    catch (...)
    {
        if (!fs::exists(target) && fs::exists(backup))
            fs::rename(backup, target);
        throw;
    }
    fs::remove_all(backup);
    std::cout << "[TSDF] Atomically committed checkpoint at " << target << std::endl;
}
void TsdfEngineFacade::load(const std::string &directory) const
{
    if (basic_engine) basic_engine->LoadFromFile(directory);
    else multi_engine->LoadFromFile(directory);
}

TsdfEngineFacade *createTsdfEngine(DatasetReader &data_reader, const YAML::Node &config, const std::string &workspace_dir)
{
    validateTsdfConfig(config);
    // 1. 建立InifiTAM Calib参数
    ITMRGBDCalib rgbd_calib;
    rgbd_calib.intrinsics_rgb.SetFrom(data_reader.width, data_reader.height, data_reader.fx, data_reader.fy, data_reader.cx, data_reader.cy);
    rgbd_calib.intrinsics_d = rgbd_calib.intrinsics_rgb;
    // 默认深度转换比例为1000, 且无偏移
    rgbd_calib.disparityCalib.SetStandard();

    // 2. 创建IfiniTAM images;
    int image_num = data_reader.train_vec.size();
    const std::string input_mode = configValue<std::string>(config, "input_mode", "preload");
    std::vector<ITMUChar4Image *> rgb_images;
    std::vector<ITMShortImage *> depth_images;
    if (input_mode == "preload")
    {
        if (!data_reader.train_vec.empty() && !data_reader.train_vec.front().image.defined())
            throw std::runtime_error("PIPE.TSDF.input_mode=preload requires READER.preload_images=true");
        rgb_images.resize(image_num);
        depth_images.resize(image_num);
    }
    const bool use_gt_pose = config["use_gt_pose"].as<bool>();
    std::vector<ORUtils::Matrix4<float> *> gt_c2w_poses;
    if (use_gt_pose)
        gt_c2w_poses.resize(image_num);

    show_console_cursor(false);
    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::PostfixText{"Convert InfiniTAM format"},
        option::ForegroundColor{Color::green},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

    for (int i = 0; i < image_num; i++)
    {
        if (input_mode == "preload")
        {
            cv::Mat rgb_image_cv = tensorToImage(data_reader.train_vec[i].image);
            rgb_images[i] = imageToITMUChar4Image(rgb_image_cv);

            cv::Mat depth_image_cv = tensorToDepth(data_reader.train_vec[i].depth);
            depth_images[i] = depthToITMUShortImage(depth_image_cv);
        }

        if (use_gt_pose)
            gt_c2w_poses[i] = tensorToInfiMatrix4(data_reader.train_vec[i].c2w);

        bar.set_option(option::PostfixText{(input_mode == "preload" ? "Convert camera: " : "Prepare camera metadata: ") + std::to_string(i) + "/" + std::to_string(image_num)});
        bar.set_progress(100 * i / image_num);
    }
    show_console_cursor(true);
    // 3. 创建main Engine
    ITMLibSettings *internalSettings = new ITMLibSettings();
    const std::string failure_mode =
        configValue<std::string>(config, "tracking_failure_mode", "legacy_ignore");
    internalSettings->behaviourOnFailure = trackingFailureMode(failure_mode);
    if (config["tracker_config"])
        internalSettings->SetTrackerConfig(config["tracker_config"].as<std::string>());
    const std::string tracking_profile =
        configValue<std::string>(config, "tracking_profile", "legacy");
    if (!config["tracker_config"] && tracking_profile == "efficient")
        internalSettings->SetTrackerConfig(
            "type=extended,levels=rrbb,useDepth=1,useColour=0,minstep=1e-4,"
            "outlierSpaceC=0.1,outlierSpaceF=0.004,numiterC=20,numiterF=20,"
            "tukeyCutOff=8,framesToSkip=20,framesToWeight=50,failureDec=20.0");
    else if (!config["tracker_config"] && tracking_profile == "robust_rpe")
        internalSettings->SetTrackerConfig(
            "type=extended,levels=rrbb,useDepth=1,useColour=0,minstep=1e-4,"
            "outlierSpaceC=0.1,outlierSpaceF=0.004,numiterC=20,numiterF=20,"
            "tukeyCutOff=8,framesToSkip=10,framesToWeight=50,failureDec=20.0");
    internalSettings->useBilateralFilter =
        configValue<bool>(config, "use_bilateral_filter", false);
    internalSettings->useApproximateRaycast =
        configValue<bool>(config, "use_approximate_raycast", false);
    internalSettings->sceneParams.voxelSize = config["voxel_size"].as<float>();
    internalSettings->sceneParams.mu = config["trunc_dist"].as<float>();
    internalSettings->sceneParams.viewFrustum_min = config["viewFrustum_min"].as<float>();
    internalSettings->sceneParams.viewFrustum_max = config["viewFrustum_max"].as<float>();
    std::cout << "[TSDF] scene_params voxel_size=" << internalSettings->sceneParams.voxelSize
              << " trunc_dist=" << internalSettings->sceneParams.mu
              << " viewFrustum_min=" << internalSettings->sceneParams.viewFrustum_min
              << " viewFrustum_max=" << internalSettings->sceneParams.viewFrustum_max
              << std::endl;
    std::cout << "[TSDF CONTAINMENT] tracking_failure_mode=" << failure_mode
              << " tracker_config_override=" << (config["tracker_config"] ? 1 : 0)
              << " semantics="
              << (failure_mode == "legacy_ignore" ? "legacy_failed_pose_may_fuse" :
                  failure_mode == "contain_failed" ? "rollback_failed_pose_and_skip_failed_fusion" :
                  failure_mode == "stop_integration" ? "rollback_failed_pose_and_skip_non_good_fusion" :
                  "relocalise_before_fusion")
              << std::endl;
    std::cout << "[TSDF TRACKING] use_bilateral_filter="
              << (internalSettings->useBilateralFilter ? 1 : 0)
              << " use_approximate_raycast="
              << (internalSettings->useApproximateRaycast ? 1 : 0)
              << " profile=" << (config["tracker_config"] ? "custom" : tracking_profile)
              << " tracker_config=\"" << internalSettings->trackerConfig << "\""
              << std::endl;
    const std::string backend = configValue<std::string>(config, "engine", "basic");
    const Vector2i image_size(data_reader.width, data_reader.height);
    ITMMainEngine *mainEngine = nullptr;
    if (backend == "basic")
    {
        std::cout << "[TSDF] engine=basic hash_profile=" << compiledHashProfile()
                  << " reject_nonfinite_tracking_pose=" << configValue<bool>(config, "reject_nonfinite_tracking_pose", false)
                  << " basic_debug_tracking_start_frame=" << configValue<int>(config, "basic_debug_tracking_start_frame", -1)
                  << " basic_debug_tracking_end_frame=" << configValue<int>(config, "basic_debug_tracking_end_frame", -1)
                  << " require_raw_tracking_success=" << configValue<bool>(config, "require_raw_tracking_success", false)
                  << " raw_tracking_failure_grace_frames=" << configValue<int>(config, "raw_tracking_failure_grace_frames", 0)
                  << std::endl;
        auto *basic = new ITMBasicEngine<ITMVoxel, ITMVoxelIndex>(internalSettings, rgbd_calib, image_size, image_size);
        basic->setRejectNonFiniteTrackingPose(configValue<bool>(config, "reject_nonfinite_tracking_pose", false));
        if (configValue<bool>(config, "basic_capacity_diagnostics", false))
            basic->configureCapacityDiagnostics((fs::path(workspace_dir) / "basic_capacity_diagnostics.csv").string());
        const int debug_start = configValue<int>(config, "basic_debug_tracking_start_frame", -1);
        const int debug_end = configValue<int>(config, "basic_debug_tracking_end_frame", -1);
        if (debug_start >= 0 && debug_end >= debug_start)
        {
            const fs::path debug_path = fs::path(workspace_dir) / "basic_tracking_diagnostics.csv";
            basic->configureTrackingDiagnostics(debug_start, debug_end, debug_path.string());
        }
        mainEngine = basic;
    }
    else if (backend == "multi")
    {
        const YAML::Node multi_cfg = config["multi"];
        auto *multi = new ITMMultiEngine<ITMVoxel, ITMVoxelIndex>(internalSettings, rgbd_calib, image_size, image_size);
        multi->setConsoleDiagnostics(configValue<bool>(multi_cfg, "console_diagnostics", false));
        mainEngine = multi;
        std::cout << "[TSDF] engine=multi force_single_submap=" << configValue<bool>(multi_cfg, "force_single_submap", false)
                  << " hash_profile=" << compiledHashProfile()
                  << " console_diagnostics=" << configValue<bool>(multi_cfg, "console_diagnostics", false)
                  << " require_raw_tracking_success=" << configValue<bool>(config, "require_raw_tracking_success", false)
                  << " raw_tracking_failure_grace_frames=" << configValue<int>(config, "raw_tracking_failure_grace_frames", 0)
                  << " loop_closure=" << configValue<bool>(multi_cfg, "loop_closure", false)
                  << " global_adjustment=" << configValue<bool>(multi_cfg, "global_adjustment", false)
                  << " boundary_policy=" << configValue<std::string>(multi_cfg, "boundary_policy", "visibility")
                  << " sequential_transform_policy=" << configValue<std::string>(multi_cfg, "sequential_transform_policy", "validated_overlap")
                  << " sequential_transform_max_translation=" << configValue<float>(multi_cfg, "sequential_transform_max_translation", 0.03f)
                  << " sequential_transform_max_rotation_deg=" << configValue<float>(multi_cfg, "sequential_transform_max_rotation_deg", 3.0f)
                  << " invalid_transform_restart_attempts=" << configValue<int>(multi_cfg, "invalid_transform_restart_attempts", 0)
                  << " handoff_pose_policy=" << configValue<std::string>(multi_cfg, "handoff_pose_policy", "candidate")
                  << " new_map_handoff_policy=" << configValue<std::string>(multi_cfg, "new_map_handoff_policy", "visibility")
                  << " new_map_handoff_min_visible_ratio=" << configValue<float>(multi_cfg, "new_map_handoff_min_visible_ratio", 0.0f)
                  << " new_map_handoff_min_visible_margin=" << configValue<float>(multi_cfg, "new_map_handoff_min_visible_margin", 0.0f)
                  << " new_map_handoff_min_accepts=" << configValue<int>(multi_cfg, "new_map_handoff_min_accepts", 1)
                  << " new_map_handoff_emergency_min_accepts=" << configValue<int>(multi_cfg, "new_map_handoff_emergency_min_accepts", -1)
                  << " candidate_recovery_policy=" << configValue<std::string>(multi_cfg, "candidate_recovery_policy", "disabled")
                  << " candidate_recovery_min_visible_ratio=" << configValue<float>(multi_cfg, "candidate_recovery_min_visible_ratio", 0.3f)
                  << " candidate_recovery_min_blocks=" << configValue<int>(multi_cfg, "candidate_recovery_min_blocks", 1000)
                  << " candidate_recovery_max_attempts=" << configValue<int>(multi_cfg, "candidate_recovery_max_attempts", 2)
                  << " primary_failure_handoff_policy=" << configValue<std::string>(multi_cfg, "primary_failure_handoff_policy", "disabled")
                  << " primary_failure_handoff_min_raw_failed_frames=" << configValue<int>(multi_cfg, "primary_failure_handoff_min_raw_failed_frames", 30)
                  << " primary_failure_handoff_raw_gate=" << configValue<std::string>(multi_cfg, "primary_failure_handoff_raw_gate", "failed_only")
                  << " primary_failure_handoff_min_raw_weak_frames=" << configValue<int>(multi_cfg, "primary_failure_handoff_min_raw_weak_frames", 0)
                  << " primary_failure_handoff_min_candidate_visible_ratio=" << configValue<float>(multi_cfg, "primary_failure_handoff_min_candidate_visible_ratio", 0.5f)
                  << " primary_failure_handoff_min_candidate_blocks=" << configValue<int>(multi_cfg, "primary_failure_handoff_min_candidate_blocks", 1000)
                  << " primary_failure_handoff_min_candidate_good_frames=" << configValue<int>(multi_cfg, "primary_failure_handoff_min_candidate_good_frames", 10)
                  << " primary_failure_handoff_min_overlap_frames=" << configValue<int>(multi_cfg, "primary_failure_handoff_min_overlap_frames", 0)
                  << " primary_failure_handoff_require_pose_continuity=" << configValue<bool>(multi_cfg, "primary_failure_handoff_require_pose_continuity", false)
                  << " tracking_boundary_policy=" << configValue<std::string>(multi_cfg, "tracking_boundary_policy", "disabled")
                  << " emergency_overlap_frames=" << configValue<int>(multi_cfg, "emergency_overlap_frames", 0)
                  << " raw_weak_boundary_frames=" << configValue<int>(multi_cfg, "raw_weak_boundary_frames", 3)
                  << " raw_failed_boundary_frames=" << configValue<int>(multi_cfg, "raw_failed_boundary_frames", 1)
                  << " tracking_boundary_min_primary_frames=" << configValue<int>(multi_cfg, "tracking_boundary_min_primary_frames", 20)
                  << " tracking_boundary_cooldown_frames=" << configValue<int>(multi_cfg, "tracking_boundary_cooldown_frames", 20)
                  << " tracking_boundary_force_handoff=" << configValue<bool>(multi_cfg, "tracking_boundary_force_handoff", true)
                  << " tracking_boundary_handoff_policy=" << configValue<std::string>(multi_cfg, "tracking_boundary_handoff_policy", "weak_or_failed")
                  << " new_map_visible_ratio=" << configValue<float>(multi_cfg, "new_map_visible_ratio", 0.2f)
                  << " new_map_block_occupancy=" << configValue<float>(multi_cfg, "new_map_block_occupancy", 0.7f)
                  << " max_block_occupancy=" << configValue<float>(multi_cfg, "max_block_occupancy", 0.8f)
                  << " fusion_reserve_blocks=" << configValue<int>(multi_cfg, "fusion_reserve_blocks", 0)
                  << " occupancy_growth_lookahead_frames=" << configValue<int>(multi_cfg, "occupancy_growth_lookahead_frames", 0)
                  << " occupancy_growth_min_primary_frames=" << configValue<int>(multi_cfg, "occupancy_growth_min_primary_frames", 20)
                  << " occupancy_growth_sample_frames=" << configValue<int>(multi_cfg, "occupancy_growth_sample_frames", 5)
                  << " occupancy_threshold_policy=" << configValue<std::string>(multi_cfg, "occupancy_threshold_policy", "fixed")
                  << " adaptive_new_map_block_occupancy=" << configValue<float>(multi_cfg, "adaptive_new_map_block_occupancy", configValue<float>(multi_cfg, "new_map_block_occupancy", 0.7f))
                  << " adaptive_max_block_occupancy=" << configValue<float>(multi_cfg, "adaptive_max_block_occupancy", configValue<float>(multi_cfg, "max_block_occupancy", 0.8f))
                  << " adaptive_occupancy_min_primary_frames=" << configValue<int>(multi_cfg, "adaptive_occupancy_min_primary_frames", 0)
                  << " adaptive_occupancy_min_processed_frames=" << configValue<int>(multi_cfg, "adaptive_occupancy_min_processed_frames", 0)
                  << " adaptive_occupancy_max_raw_weak_streak=" << configValue<int>(multi_cfg, "adaptive_occupancy_max_raw_weak_streak", 0)
                  << " adaptive_occupancy_max_raw_failed_streak=" << configValue<int>(multi_cfg, "adaptive_occupancy_max_raw_failed_streak", 0)
                  << " skip_primary_fusion_on_capacity_handoff=" << configValue<bool>(multi_cfg, "skip_primary_fusion_on_capacity_handoff", false)
                  << " allow_visibility_handoff=" << configValue<bool>(multi_cfg, "allow_visibility_handoff", true)
                  << " paging=" << configValue<bool>(multi_cfg, "paging", false)
                  << " async_page_out=" << configValue<bool>(multi_cfg, "async_page_out", false)
                  << " max_pending_page_snapshots=" << configValue<int>(multi_cfg, "max_pending_page_snapshots", 1)
                  << " max_pending_page_snapshot_mb=" << configValue<int>(multi_cfg, "max_pending_page_snapshot_mb", 0)
                  << " async_page_in=" << configValue<bool>(multi_cfg, "async_page_in", false)
                  << " max_pending_page_ins=" << configValue<int>(multi_cfg, "max_pending_page_ins", 1)
                  << " max_pending_page_in_mb=" << configValue<int>(multi_cfg, "max_pending_page_in_mb", 0)
                  << " cpu_page_cache_limit_mb=" << configValue<int>(multi_cfg, "cpu_page_cache_limit_mb", 0)
                  << " resident_gpu_maps=" << residentMapCount(multi_cfg)
                  << " paging_storage=" << configValue<std::string>(multi_cfg, "paging_storage", "disk") << std::endl;
    }
    else
        throw std::runtime_error("Unknown PIPE.TSDF.engine: " + backend);

    if (backend == "multi")
    {
        const YAML::Node multi_cfg = config["multi"];
        std::cout << "[TSDF] configuring MultiEngine policy" << std::endl;
        const std::string boundary_policy = configValue<std::string>(multi_cfg, "boundary_policy", "visibility");
        const float visible_ratio =
            boundary_policy == "capacity" ? 0.0f : configValue<float>(multi_cfg, "new_map_visible_ratio", 0.2f);
        static_cast<ITMMultiEngine<ITMVoxel, ITMVoxelIndex> *>(mainEngine)->configureHybridSubmaps(
            !configValue<bool>(multi_cfg, "force_single_submap", false),
            configValue<bool>(multi_cfg, "loop_closure", false),
            configValue<bool>(multi_cfg, "global_adjustment", false),
            visible_ratio,
            configValue<int>(multi_cfg, "overlap_frames", 40),
            configValue<int>(multi_cfg, "trial_frames", 60),
            configValue<int>(multi_cfg, "emergency_overlap_frames", 0),
            configValue<int>(multi_cfg, "sampled_original_blocks", 1000),
            configValue<int>(multi_cfg, "max_frames", 750),
            configValue<float>(multi_cfg, "max_block_occupancy", 0.8f),
            configValue<float>(multi_cfg, "new_map_block_occupancy", 0.7f),
            configValue<bool>(multi_cfg, "paging", false),
            residentMapCount(multi_cfg),
            (fs::path(workspace_dir) / "tsdf_pages").string(),
            configValue<std::string>(multi_cfg, "paging_storage", "disk"),
            configValue<int>(multi_cfg, "loop_min_submap_separation", 3),
            configValue<int>(multi_cfg, "loop_candidate_cooldown_frames", 30),
            configValue<bool>(multi_cfg, "async_page_out", false),
            configValue<int>(multi_cfg, "max_pending_page_snapshots", 1),
            static_cast<int64_t>(configValue<int>(multi_cfg, "max_pending_page_snapshot_mb", 0)) * 1024LL * 1024LL,
            configValue<bool>(multi_cfg, "async_page_in", false),
            configValue<int>(multi_cfg, "max_pending_page_ins", 1),
            static_cast<int64_t>(configValue<int>(multi_cfg, "max_pending_page_in_mb", 0)) * 1024LL * 1024LL,
            static_cast<int64_t>(configValue<int>(multi_cfg, "cpu_page_cache_limit_mb", 0)) * 1024LL * 1024LL,
            configValue<std::string>(multi_cfg, "sequential_transform_policy", "validated_overlap"),
            configValue<float>(multi_cfg, "sequential_transform_max_translation", 0.03f),
            configValue<float>(multi_cfg, "sequential_transform_max_rotation_deg", 3.0f),
            configValue<int>(multi_cfg, "invalid_transform_restart_attempts", 0),
            configValue<std::string>(multi_cfg, "handoff_pose_policy", "candidate"),
            configValue<std::string>(multi_cfg, "new_map_handoff_policy", "visibility"),
            configValue<float>(multi_cfg, "new_map_handoff_min_visible_ratio", 0.0f),
            configValue<float>(multi_cfg, "new_map_handoff_min_visible_margin", 0.0f),
            configValue<int>(multi_cfg, "new_map_handoff_min_accepts", 1),
            configValue<int>(multi_cfg, "new_map_handoff_emergency_min_accepts", -1),
            configValue<std::string>(multi_cfg, "candidate_recovery_policy", "disabled"),
            configValue<float>(multi_cfg, "candidate_recovery_min_visible_ratio", 0.3f),
            configValue<int>(multi_cfg, "candidate_recovery_min_blocks", 1000),
            configValue<int>(multi_cfg, "candidate_recovery_max_attempts", 2),
            configValue<std::string>(multi_cfg, "primary_failure_handoff_policy", "disabled"),
            configValue<int>(multi_cfg, "primary_failure_handoff_min_raw_failed_frames", 30),
            configValue<std::string>(multi_cfg, "primary_failure_handoff_raw_gate", "failed_only"),
            configValue<int>(multi_cfg, "primary_failure_handoff_min_raw_weak_frames", 0),
            configValue<float>(multi_cfg, "primary_failure_handoff_min_candidate_visible_ratio", 0.5f),
            configValue<int>(multi_cfg, "primary_failure_handoff_min_candidate_blocks", 1000),
            configValue<int>(multi_cfg, "primary_failure_handoff_min_candidate_good_frames", 10),
            configValue<int>(multi_cfg, "primary_failure_handoff_min_overlap_frames", 0),
            configValue<bool>(multi_cfg, "primary_failure_handoff_require_pose_continuity", false),
            configValue<std::string>(multi_cfg, "tracking_boundary_policy", "disabled"),
            configValue<int>(multi_cfg, "raw_weak_boundary_frames", 3),
            configValue<int>(multi_cfg, "raw_failed_boundary_frames", 1),
            configValue<int>(multi_cfg, "tracking_boundary_min_primary_frames", 20),
            configValue<int>(multi_cfg, "tracking_boundary_cooldown_frames", 20),
            configValue<bool>(multi_cfg, "tracking_boundary_force_handoff", true),
            configValue<std::string>(multi_cfg, "tracking_boundary_handoff_policy", "weak_or_failed"),
            configValue<int>(multi_cfg, "fusion_reserve_blocks", 0),
            configValue<int>(multi_cfg, "occupancy_growth_lookahead_frames", 0),
            configValue<int>(multi_cfg, "occupancy_growth_min_primary_frames", 20),
            configValue<int>(multi_cfg, "occupancy_growth_sample_frames", 5),
            configValue<bool>(multi_cfg, "skip_primary_fusion_on_capacity_handoff", false),
            configValue<bool>(multi_cfg, "allow_visibility_handoff", true),
            configValue<std::string>(multi_cfg, "occupancy_threshold_policy", "fixed"),
            configValue<float>(multi_cfg, "adaptive_new_map_block_occupancy", configValue<float>(multi_cfg, "new_map_block_occupancy", 0.7f)),
            configValue<float>(multi_cfg, "adaptive_max_block_occupancy", configValue<float>(multi_cfg, "max_block_occupancy", 0.8f)),
            configValue<int>(multi_cfg, "adaptive_occupancy_min_primary_frames", 0),
            configValue<int>(multi_cfg, "adaptive_occupancy_min_processed_frames", 0),
            configValue<int>(multi_cfg, "adaptive_occupancy_max_raw_weak_streak", 0),
            configValue<int>(multi_cfg, "adaptive_occupancy_max_raw_failed_streak", 0));
        std::cout << "[TSDF] MultiEngine policy configured" << std::endl;
    }
    if (use_gt_pose)
    {
        std::cout << "[TSDF] configuring GT poses" << std::endl;
        if (backend == "basic")
        {
            auto *basic = static_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(mainEngine);
            basic->turnOffTracking();
            basic->gtC2wPoses = gt_c2w_poses;
        }
        else
        {
            auto *multi = static_cast<ITMMultiEngine<ITMVoxel, ITMVoxelIndex> *>(mainEngine);
            multi->configureGroundTruthPoses(gt_c2w_poses);
        }
    }
    // 4. 创建CLI Engine
    std::cout << "[TSDF] initializing CLI input" << std::endl;
    CLIEngine *tsdf_engine = CLIEngine::Instance();
    if (input_mode == "preload")
        tsdf_engine->Initialise(rgb_images, depth_images, mainEngine);
    else
    {
        std::vector<Camera> *cams = &data_reader.train_vec;
        const bool input_prefetch = configValue<bool>(config, "input_prefetch", false);
        const std::string input_tensor_bridge =
            configValue<std::string>(config, "input_tensor_bridge", "direct");
        const bool input_tensor_bridge_validate =
            configValue<bool>(config, "input_tensor_bridge_validate", false);
        auto prefetch_state = std::make_shared<OnDemandPrefetchState>(cams, input_prefetch);
        tsdf_engine->InitialiseOnDemand(image_size, image_size, image_num,
            [cams, prefetch_state, input_tensor_bridge, input_tensor_bridge_validate](
                int frame_idx, ITMUChar4Image *rgb, ITMShortImage *depth)
            {
                if (frame_idx < 0 || frame_idx >= int(cams->size()))
                    return false;
                prefetch_state->prepare(frame_idx);
                if (input_tensor_bridge == "direct")
                {
                    imageTensorToITMUChar4Image((*cams)[frame_idx].image, *rgb);
                    depthTensorToITMUShortImage((*cams)[frame_idx].depth, *depth);
                    if (input_tensor_bridge_validate)
                    {
                        ITMUChar4Image legacy_rgb(rgb->noDims, true, false);
                        ITMShortImage legacy_depth(depth->noDims, true, false);
                        imageToITMUChar4Image(tensorToImage((*cams)[frame_idx].image), legacy_rgb);
                        depthToITMUShortImage(tensorToDepth((*cams)[frame_idx].depth), legacy_depth);

                        const auto *direct_rgb = rgb->GetData(MEMORYDEVICE_CPU);
                        const auto *reference_rgb = legacy_rgb.GetData(MEMORYDEVICE_CPU);
                        const auto *direct_depth = reinterpret_cast<const uint16_t *>(
                            depth->GetData(MEMORYDEVICE_CPU));
                        const auto *reference_depth = reinterpret_cast<const uint16_t *>(
                            legacy_depth.GetData(MEMORYDEVICE_CPU));
                        const int64_t pixel_count = int64_t(rgb->noDims.x) * rgb->noDims.y;
                        int64_t rgb_mismatches = 0;
                        int64_t depth_mismatches = 0;
                        int max_rgb_delta = 0;
                        int max_depth_delta = 0;
                        for (int64_t pixel = 0; pixel < pixel_count; ++pixel)
                        {
                            for (int channel = 0; channel < 4; ++channel)
                            {
                                const int delta = std::abs(
                                    int(direct_rgb[pixel].v[channel]) -
                                    int(reference_rgb[pixel].v[channel]));
                                if (delta != 0)
                                    ++rgb_mismatches;
                                max_rgb_delta = std::max(max_rgb_delta, delta);
                            }
                            const int depth_delta = std::abs(
                                int(direct_depth[pixel]) - int(reference_depth[pixel]));
                            if (depth_delta != 0)
                                ++depth_mismatches;
                            max_depth_delta = std::max(max_depth_delta, depth_delta);
                        }
                        if (rgb_mismatches != 0 || depth_mismatches != 0)
                            throw std::runtime_error(
                                "Direct TSDF tensor bridge differs from the legacy OpenCV bridge at frame " +
                                std::to_string(frame_idx) + ": rgb_mismatches=" +
                                std::to_string(rgb_mismatches) + " max_rgb_delta=" +
                                std::to_string(max_rgb_delta) + " depth_mismatches=" +
                                std::to_string(depth_mismatches) + " max_depth_delta=" +
                                std::to_string(max_depth_delta));
                        if (frame_idx == 0 || frame_idx + 1 == int(cams->size()))
                            std::cout << "[TSDF INPUT] direct_bridge_validation frame=" << frame_idx
                                      << " pixels=" << pixel_count
                                      << " rgb_mismatches=0 depth_mismatches=0" << std::endl;
                    }
                }
                else
                {
                    imageToITMUChar4Image(tensorToImage((*cams)[frame_idx].image), *rgb);
                    depthToITMUShortImage(tensorToDepth((*cams)[frame_idx].depth), *depth);
                }
                prefetch_state->prefetch(frame_idx + 1);
                return true;
            }, mainEngine);
    }
    std::cout << "[TSDF] CLI input initialized" << std::endl;
    std::cout << "[TSDF] engine=" << backend << " input_mode=" << input_mode
              << " input_prefetch=" << configValue<bool>(config, "input_prefetch", false)
              << " input_tensor_bridge=" << configValue<std::string>(config, "input_tensor_bridge", "direct")
              << " input_tensor_bridge_validate=" << configValue<bool>(config, "input_tensor_bridge_validate", false)
              << " frames=" << image_num << std::endl;
    return new TsdfEngineFacade(tsdf_engine, mainEngine, backend);
}

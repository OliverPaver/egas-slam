#pragma once

#include "TsdfFusion/CLIEngine.h"
#include "ITMLib/ITMLibDefines.h"
#include "ITMLib/Core/ITMBasicEngine.h"
#include "ITMLib/Core/ITMMultiEngine.h"

#include "dataset_reader.h"
#include "cv_utils.h"
#include "tensor_math.h"

using namespace InfiniTAM::Engine;
using namespace InputSource;
using namespace ITMLib;

struct TsdfSubmapBounds
{
    bool valid = false;
    Vector3f local_min = Vector3f(0.0f);
    Vector3f local_max = Vector3f(0.0f);
};

class TsdfEngineFacade
{
public:
    TsdfEngineFacade(CLIEngine *cli_engine, ITMMainEngine *main_engine, const std::string &backend);

    bool processFrame();
    bool lastTrackingSucceeded() const { return last_tracking_result != ITMTrackingState::TRACKING_FAILED; }
    int getLastTrackingResult() const { return last_tracking_result; }
    int getLastRawTrackingResult() const { return last_raw_tracking_result; }
	int getLastPolicyTrackingResult() const { return last_policy_tracking_result; }
	bool getLastPoseRollback() const { return last_pose_rollback; }
	bool getLastFusionSuppressed() const { return last_fusion_suppressed; }
    int currentFrameNo() const { return current_frame_no; }
    Vector2i getDepthSize() const { return cli_engine->GetDepthSize(); }
    ITMMainEngine *getMainEngine() const { return main_engine; }
    const std::string &getBackend() const { return backend; }
    bool isMulti() const { return multi_engine != nullptr; }
    float getVoxelSize() const;
    ORUtils::Matrix4<float> getGlobalCameraPose() const;
    int getPrimarySubmapId() const;
    int getSubmapCount() const;
    std::vector<int> getActiveSubmapIds() const;
    std::vector<ITMActiveMapManager::TransitionEvent> consumeRecentTransitionEvents() const;
    ORUtils::Matrix4<float> getGlobalToLocal(int submap_id) const;
    void setGlobalToLocal(int submap_id, const ORUtils::Matrix4<float> &global_to_local) const;
    int getSubmapBlockCount(int submap_id) const;
    int getSubmapCapacity(int submap_id) const;
    TsdfSubmapBounds getSubmapLocalBounds(int submap_id) const;
    bool isSubmapResident(int submap_id) const;
    int getResidentSubmapCount() const;
    bool isPagingEnabled() const;
    void runRaycast(ORUtils::SE3Pose *global_pose, ITMIntrinsics *intrinsics, int local_map_id = -1);
    void prefetchRaycastMaps(const std::vector<int> &local_map_ids);
    bool tryAdoptPrefetchedRaycastMaps(const std::vector<int> &local_map_ids);
    void setExtraResidencyPins(const std::vector<int> &local_map_ids);
    void trimResidency(int pinned_local_map_id = -1);
    void trimResidency(const std::vector<int> &required_local_map_ids);
    void synchronizePaging();
    ITMUChar4Image *getFreeImage() const;
    ITMFloat4Image *getFreeVertex() const;
    void saveMesh(const std::string &filename) const;
    void save(const std::string &directory) const;
    void load(const std::string &directory) const;

private:
    CLIEngine *cli_engine;
    ITMMainEngine *main_engine;
    ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *basic_engine;
    ITMMultiEngine<ITMVoxel, ITMVoxelIndex> *multi_engine;
    std::string backend;
    int current_frame_no = 0;
    int last_tracking_result = ITMTrackingState::TRACKING_GOOD;
    int last_raw_tracking_result = ITMTrackingState::TRACKING_GOOD;
	int last_policy_tracking_result = ITMTrackingState::TRACKING_GOOD;
	bool last_pose_rollback = false;
	bool last_fusion_suppressed = false;
};

TsdfEngineFacade *createTsdfEngine(DatasetReader &data_reader, const YAML::Node &config, const std::string &workspace_dir);
void validateTsdfConfig(const YAML::Node &config);

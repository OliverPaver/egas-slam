// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMMainEngine.h"
#include "ITMTrackingController.h"
#include "../Engines/LowLevel/Interface/ITMLowLevelEngine.h"
#include "../Engines/ViewBuilding/Interface/ITMViewBuilder.h"
#include "../Objects/Misc/ITMIMUCalibrator.h"
#include "../../FernRelocLib/Relocaliser.h"

#include "../Engines/MultiScene/ITMActiveMapManager.h"
#include "../Engines/MultiScene/ITMGlobalAdjustmentEngine.h"
#include "../Engines/Visualisation/Interface/ITMMultiVisualisationEngine.h"
#include "../Engines/Meshing/ITMMultiMeshingEngineFactory.h"

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace ITMLib
{
	/** \brief
	*/
	template <typename TVoxel, typename TIndex>
	class ITMMultiEngine : public ITMMainEngine
	{
	private:
		const ITMLibSettings *settings;

		ITMLowLevelEngine *lowLevelEngine;
		ITMVisualisationEngine<TVoxel, TIndex> *visualisationEngine;
		ITMMultiVisualisationEngine<TVoxel, TIndex> *multiVisualisationEngine;

		ITMMultiMeshingEngine<TVoxel, TIndex> *meshingEngine;

		ITMViewBuilder *viewBuilder;
		ITMTrackingController *trackingController;
		ITMTracker *tracker;
		ITMIMUCalibrator *imuCalibrator;
		ITMDenseMapper<TVoxel, TIndex> *denseMapper;

		FernRelocLib::Relocaliser<float> *relocaliser;

		ITMVoxelMapGraphManager<TVoxel, TIndex> *mapManager;
		ITMActiveMapManager *mActiveDataManager;
		ITMGlobalAdjustmentEngine *mGlobalAdjustmentEngine;
		bool mScheduleGlobalAdjustment;
		bool loopClosureEnabled;
		bool globalAdjustmentEnabled;
		bool trackingActive;
		bool consoleDiagnostics;
		bool allowNewLocalMapsConfigured;
		struct EffectiveOccupancyThresholds
		{
			float newMapOccupancy;
			float maxOccupancy;
			bool adaptiveActive;
		};
		int framesProcessed;
		int maxFramesPerPrimary;
		float newMapBlockOccupancy;
		float maxBlockOccupancy;
		std::string occupancyThresholdPolicy;
		float adaptiveNewMapBlockOccupancy;
		float adaptiveMaxBlockOccupancy;
		int adaptiveOccupancyMinPrimaryFrames;
		int adaptiveOccupancyMinProcessedFrames;
		int adaptiveOccupancyMaxRawWeakStreak;
		int adaptiveOccupancyMaxRawFailedStreak;
		int fusionReserveBlocks;
		int lastPrimaryMapIdx;
		int lastOccupancyRequestMapIdx;
		int lastOccupancyRequestFrame;
		int lastPrimaryOccupancyFrame;
		int lastPrimaryOccupancyBlocks;
		int occupancyGrowthLookaheadFrames;
		int occupancyGrowthMinPrimaryFrames;
		int occupancyGrowthSampleFrames;
		int lastCapacityDeferMapIdx;
		int lastCapacityDeferFrame;
		int lastPrimaryHandoffFusionSkipMapIdx;
		int lastPrimaryHandoffFusionSkipFrame;
		int capacityLimitedPrimaryMapIdx;
		bool skipPrimaryFusionOnCapacityHandoff;
		int primaryMapFrames;
		int lastPrimaryRawTrackingResult;
		int lastPrimaryPolicyTrackingResult;
		bool lastPrimaryPoseRollback;
		bool lastPrimaryFusionSuppressed;
		std::string trackingBoundaryPolicy;
		int rawWeakFrameStreak;
		int rawFailedFrameStreak;
		int rawWeakBoundaryFrames;
		int rawFailedBoundaryFrames;
		int trackingBoundaryMinPrimaryFrames;
		int trackingBoundaryCooldownFrames;
		int lastTrackingBoundaryRequestFrame;
		bool trackingBoundaryForceHandoff;
		std::string trackingBoundaryHandoffPolicy;
		bool pagingEnabled;
		bool asyncPageOutSnapshots;
		bool asyncPageInPrefetch;
		bool cpuPagingStorage;
		int maxResidentMaps;
			int maxPendingPageSnapshots;
			int64_t maxPendingPageSnapshotBytes;
			int maxPendingPageIns;
			int64_t maxPendingPageInBytes;
			int64_t maxCpuPageCacheBytes;
			int minLoopSubmapSeparation;
		int loopCandidateCooldownFrames;
		std::map<int, int> loopCandidateLastAttempt;
		std::vector<int> extraResidencyPins;
		std::string pagingDirectory;
		std::string pagingStorageMode;
		std::ofstream pagingLog;
		std::ofstream loopLog;

		Vector2i trackedImageSize;
		ITMRenderState *renderState_freeview;
		ITMRenderState *renderState_multiscene;
		int freeviewLocalMapIdx;
		EffectiveOccupancyThresholds getEffectiveOccupancyThresholds(int occupancySampleFrames) const;
		void enforceResidency(int pinnedLocalMapIdx = -1);
		void enforceResidency(const std::vector<int> &requiredMaps);
			void ensureLocalMapResident(int localMapIdx);
			void prefetchLocalMaps(const std::vector<int> &localMapIndices);
			int64_t getPassiveCpuPageCacheEstimatedBytes(void) const;
			void enforceCpuPageCacheBudget(const std::vector<int> &requiredMaps);
			void logPagingEvent(const char *action, int localMapIdx, double durationMs, int blockCount, int capacity, int64_t pageBytes);
		void logLoopEvent(const char *action, int localMapIdx, float distance);

		/// Pointer for storing the current input frame
		ITMView *view;
		std::vector<ORUtils::Matrix4<float> *> gtC2wPoses;
	public:
		ITMView* GetView() { return view; }

		ITMTrackingState* GetTrackingState(void);

		/// Process a frame with rgb and depth images and (optionally) a corresponding imu measurement
		ITMTrackingState::TrackingResult ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage, ITMIMUMeasurement *imuMeasurement = NULL);

		/// Get a result image as output
		Vector2i GetImageSize(void) const;

		void GetImage(ITMUChar4Image *out, GetImageType getImageType, ORUtils::SE3Pose *pose = NULL, ITMIntrinsics *intrinsics = NULL);

		void changeFreeviewLocalMapIdx(ORUtils::SE3Pose *pose, int newIdx);
		void setFreeviewLocalMapIdx(int newIdx)
		{
			freeviewLocalMapIdx = newIdx;
		}
		int getFreeviewLocalMapIdx(void) const
		{
			return freeviewLocalMapIdx;
		}
		int findPrimaryLocalMapIdx(void) const
		{
			return mActiveDataManager->findPrimaryLocalMapIdx();
		}
		size_t getLocalMapCount(void) const { return mapManager->numLocalMaps(); }
		std::vector<int> getActiveLocalMapIndices(void) const { return mActiveDataManager->getActiveLocalMapIndices(); }
		std::vector<ITMActiveMapManager::TransitionEvent> consumeRecentTransitionEvents(void) const { return mActiveDataManager->consumeRecentTransitionEvents(); }
		int getLocalMapBlockCount(int localMapIdx) const { return mapManager->getLocalMapSize(localMapIdx); }
		int getLocalMapCapacity(int localMapIdx) const { return SDF_LOCAL_BLOCK_NUM; }
		bool getLocalMapBlockBounds(int localMapIdx, Vector3f &localMin, Vector3f &localMax) const
		{
			return mapManager->getLocalMapBlockBounds(localMapIdx, localMin, localMax);
		}
		bool isLocalMapResident(int localMapIdx) const { return mapManager->isLocalMapResident(localMapIdx); }
		int getResidentLocalMapCount(void) const { return mapManager->countResidentLocalMaps(); }
		std::vector<int> getResidentLocalMapIndices(void) const { return mapManager->getResidentLocalMapIds(); }
		int getLastPrimaryRawTrackingResult(void) const { return lastPrimaryRawTrackingResult; }
		int getLastPrimaryPolicyTrackingResult(void) const { return lastPrimaryPolicyTrackingResult; }
		bool getLastPrimaryPoseRollback(void) const { return lastPrimaryPoseRollback; }
		bool getLastPrimaryFusionSuppressed(void) const { return lastPrimaryFusionSuppressed; }
		bool isPagingEnabled(void) const { return pagingEnabled; }
		const ORUtils::SE3Pose &getLocalMapGlobalToLocal(int localMapIdx) const { return mapManager->getEstimatedGlobalPose(localMapIdx); }
		void setLocalMapGlobalToLocal(int localMapIdx, const ORUtils::SE3Pose &pose) { mapManager->setEstimatedGlobalPose(localMapIdx, pose); }
		void setConsoleDiagnostics(bool enabled)
		{
			consoleDiagnostics = enabled;
			if (mActiveDataManager != NULL) mActiveDataManager->setConsoleDiagnostics(enabled);
		}
		ORUtils::Matrix4<float> getGlobalCameraPose(void) const;
		float getVoxelSize(void) const { return settings->sceneParams.voxelSize; }
		void configureHybridSubmaps(bool allowNewMaps, bool enableLoopClosure, bool enableGlobalAdjustment,
			float visibleRatioThreshold, int overlapFrames, int trialFrames, int emergencyOverlapFrames, int sampledOriginalBlocks,
			int maxFramesPerMap, float maxOccupancy, float newMapOccupancy, bool enablePaging, int residentMaps, const std::string &pageDirectory,
			const std::string &pageStorageMode = "disk",
			int minLoopSeparation = 3, int loopCooldownFrames = 30, bool asyncPageOut = false, int maxPendingSnapshots = 1,
					int64_t maxPendingSnapshotBytes = 0, bool asyncPageIn = false, int maxPendingPageIns = 1, int64_t maxPendingPageInBytes = 0,
					int64_t maxCpuPageCacheBytes = 0,
						const std::string &sequentialTransformPolicy = "validated_overlap", float sequentialTransformMaxTranslation = 0.03f,
						float sequentialTransformMaxRotationDeg = 3.0f, int invalidTransformRestartAttempts = 0, const std::string &handoffPosePolicy = "candidate",
						const std::string &newMapHandoffPolicy = "visibility",
						float newMapHandoffMinVisibleRatio = 0.0f, float newMapHandoffMinVisibleMargin = 0.0f,
						int newMapHandoffMinAcceptedWindows = 1, int newMapHandoffEmergencyMinAcceptedWindows = -1,
						const std::string &candidateRecoveryPolicy = "disabled", float candidateRecoveryMinVisibleRatio = 0.3f,
					int candidateRecoveryMinBlocks = 1000, int candidateRecoveryMaxAttempts = 2,
					const std::string &primaryFailureHandoffPolicy = "disabled", int primaryFailureHandoffMinRawFailedFrames = 30,
					const std::string &primaryFailureHandoffRawGate = "failed_only", int primaryFailureHandoffMinRawWeakFrames = 0,
					float primaryFailureHandoffMinCandidateVisibleRatio = 0.5f, int primaryFailureHandoffMinCandidateBlocks = 1000,
					int primaryFailureHandoffMinCandidateGoodFrames = 10, int primaryFailureHandoffMinOverlapFrames = 0,
					bool primaryFailureHandoffRequirePoseContinuity = false,
					const std::string &trackingBoundaryPolicy = "disabled", int rawWeakBoundaryFrames = 3, int rawFailedBoundaryFrames = 1,
					int trackingBoundaryMinPrimaryFrames = 20, int trackingBoundaryCooldownFrames = 20, bool trackingBoundaryForceHandoff = true,
					const std::string &trackingBoundaryHandoffPolicy = "weak_or_failed",
					int fusionReserveBlocks = 0, int occupancyGrowthLookaheadFrames = 0, int occupancyGrowthMinPrimaryFrames = 0,
					int occupancyGrowthSampleFrames = 1, bool skipPrimaryFusionOnCapacityHandoff = false,
					bool allowVisibilityHandoff = true, const std::string &occupancyThresholdPolicy = "fixed",
					float adaptiveNewMapOccupancy = 0.0f, float adaptiveMaxOccupancy = 0.0f,
					int adaptiveOccupancyMinPrimaryFrames = 0, int adaptiveOccupancyMinProcessedFrames = 0,
					int adaptiveOccupancyMaxRawWeakStreak = 0, int adaptiveOccupancyMaxRawFailedStreak = 0);
		void turnOffTracking(void);
		void turnOnTracking(void);
		void configureGroundTruthPoses(const std::vector<ORUtils::Matrix4<float> *> &poses);
		void runRaycast(ORUtils::SE3Pose *globalPose, ITMIntrinsics *intrinsics, int localMapIdx = -1);
		void prefetchRaycastMaps(const std::vector<int> &localMapIndices) { prefetchLocalMaps(localMapIndices); }
		bool tryAdoptPrefetchedRaycastMaps(const std::vector<int> &localMapIndices);
		void setExtraResidencyPins(const std::vector<int> &localMapIndices);
		void trimResidency(int pinnedLocalMapIdx = -1) { enforceResidency(pinnedLocalMapIdx); }
		void trimResidency(const std::vector<int> &requiredMaps) { enforceResidency(requiredMaps); }
		void synchronizePaging(void);
		ORUtils::Image<Vector4u> *GetFreeImage(void);
		ORUtils::Image<Vector4f> *GetFreeVertex(void);

		/// Extracts a mesh from the current scene and saves it to the model file specified by the file name
		void SaveSceneToMesh(const char *fileName);
		// Optional demonstration observer: reads resident maps only. Never pages,
		// changes a map transform, or touches tracking/free-view render buffers.
		bool CaptureResidentMesh(int localMapId, const std::string &filename);
		bool CaptureResidentView(int localMapId, const Matrix4f &cameraToLocal,
			ITMIntrinsics &intrinsics, ITMUChar4Image &colour, ITMFloat4Image &vertices);

		/// save and load the full scene and relocaliser (if any) to/from file
		void SaveToFile();
		void LoadFromFile();
		void SaveToFile(const std::string &directory);
		void LoadFromFile(const std::string &directory);

		//void writeFullTrajectory(void) const;
		//void SaveSceneToMesh(const char *objFileName);

		/** \brief Constructor
			Ommitting a separate image size for the depth images
			will assume same resolution as for the RGB images.
		*/
		ITMMultiEngine(const ITMLibSettings *settings, const ITMRGBDCalib &calib, Vector2i imgSize_rgb, Vector2i imgSize_d = Vector2i(-1, -1));
		~ITMMultiEngine(void);
	};
}

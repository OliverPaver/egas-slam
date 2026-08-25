// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMMultiEngine.h"

#include "../Engines/LowLevel/ITMLowLevelEngineFactory.h"
#include "../Engines/ViewBuilding/ITMViewBuilderFactory.h"
#include "../Engines/Visualisation/ITMVisualisationEngineFactory.h"
#include "../Engines/Visualisation/ITMMultiVisualisationEngineFactory.h"
#include "../Engines/Meshing/ITMMeshingEngineFactory.h"
#include "../Trackers/ITMTrackerFactory.h"
#include "../Utils/ITMTrackingFailurePolicy.h"

#include "../../MiniSlamGraphLib/QuaternionHelpers.h"
#include "../../ORUtils/FileUtils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <limits>
#include <memory>

using namespace ITMLib;

//#define DEBUG_MULTISCENE

// number of nearest neighbours to find in the loop closure detection
static const int k_loopcloseneighbours = 1;

// maximum distance reported by LCD library to attempt relocalisation
static const float F_maxdistattemptreloc = 0.05f;

// loop closure global adjustment runs on a separate thread
static const bool separateThreadGlobalAdjustment = true;

#define MULTI_ENGINE_CONSOLE(...) do { if (consoleDiagnostics) fprintf(stdout, __VA_ARGS__); } while (0)

static bool isFiniteMatrix(const ORUtils::Matrix4<float> &matrix)
{
	for (int i = 0; i < 16; ++i)
	{
		if (!std::isfinite(matrix.m[i])) return false;
	}
	return true;
}

static bool isFinitePose(const ORUtils::SE3Pose *pose)
{
	return pose != NULL && isFiniteMatrix(pose->GetM());
}

template <typename TVoxel, typename TIndex>
ITMMultiEngine<TVoxel, TIndex>::ITMMultiEngine(const ITMLibSettings *settings, const ITMRGBDCalib& calib, Vector2i imgSize_rgb, Vector2i imgSize_d)
{
	if ((imgSize_d.x == -1) || (imgSize_d.y == -1)) imgSize_d = imgSize_rgb;

	this->settings = settings;

	const ITMLibSettings::DeviceType deviceType = settings->deviceType;
	lowLevelEngine = ITMLowLevelEngineFactory::MakeLowLevelEngine(deviceType);
	viewBuilder = ITMViewBuilderFactory::MakeViewBuilder(calib, deviceType);
	visualisationEngine = ITMVisualisationEngineFactory::MakeVisualisationEngine<TVoxel, TIndex>(deviceType);

	meshingEngine = NULL;
	if (settings->createMeshingEngine)
		meshingEngine = ITMMultiMeshingEngineFactory::MakeMeshingEngine<TVoxel, TIndex>(deviceType);

	renderState_freeview = NULL; //will be created by the visualisation engine

	denseMapper = new ITMDenseMapper<TVoxel, TIndex>(settings);

	imuCalibrator = new ITMIMUCalibrator_iPad();
	tracker = ITMTrackerFactory::Instance().Make(imgSize_rgb, imgSize_d, settings, lowLevelEngine, imuCalibrator, &settings->sceneParams);
	trackingController = new ITMTrackingController(tracker, settings);
	trackedImageSize = trackingController->GetTrackedImageSize(imgSize_rgb, imgSize_d);

	freeviewLocalMapIdx = 0;
	mapManager = new ITMVoxelMapGraphManager<TVoxel, TIndex>(settings, visualisationEngine, denseMapper, trackedImageSize);
	mActiveDataManager = new ITMActiveMapManager(mapManager);
	mActiveDataManager->initiateNewLocalMap(true);

	//TODO	tracker->UpdateInitialPose(allData[0]->trackingState);

	view = NULL; // will be allocated by the view builder

	relocaliser = new FernRelocLib::Relocaliser<float>(imgSize_d, Vector2f(settings->sceneParams.viewFrustum_min, settings->sceneParams.viewFrustum_max), 0.1f, 1000, 4);

	mGlobalAdjustmentEngine = new ITMGlobalAdjustmentEngine();
	mScheduleGlobalAdjustment = false;
	loopClosureEnabled = false;
	globalAdjustmentEnabled = false;
	trackingActive = true;
	consoleDiagnostics = false;
	allowNewLocalMapsConfigured = true;
	framesProcessed = 0;
	maxFramesPerPrimary = 0;
	newMapBlockOccupancy = 0.0f;
	maxBlockOccupancy = 0.0f;
	occupancyThresholdPolicy = "fixed";
	adaptiveNewMapBlockOccupancy = 0.0f;
	adaptiveMaxBlockOccupancy = 0.0f;
	adaptiveOccupancyMinPrimaryFrames = 0;
	adaptiveOccupancyMinProcessedFrames = 0;
	adaptiveOccupancyMaxRawWeakStreak = 0;
	adaptiveOccupancyMaxRawFailedStreak = 0;
	fusionReserveBlocks = 0;
	lastPrimaryMapIdx = 0;
	lastOccupancyRequestMapIdx = -1;
	lastOccupancyRequestFrame = -1;
	lastPrimaryOccupancyFrame = -1;
	lastPrimaryOccupancyBlocks = -1;
	occupancyGrowthLookaheadFrames = 0;
	occupancyGrowthMinPrimaryFrames = 0;
	occupancyGrowthSampleFrames = 1;
	lastCapacityDeferMapIdx = -1;
	lastCapacityDeferFrame = -1;
	lastPrimaryHandoffFusionSkipMapIdx = -1;
	lastPrimaryHandoffFusionSkipFrame = -1;
	capacityLimitedPrimaryMapIdx = -1;
	skipPrimaryFusionOnCapacityHandoff = false;
	primaryMapFrames = 0;
	lastPrimaryRawTrackingResult = ITMTrackingState::TRACKING_GOOD;
	lastPrimaryPolicyTrackingResult = ITMTrackingState::TRACKING_GOOD;
	lastPrimaryPoseRollback = false;
	lastPrimaryFusionSuppressed = false;
	trackingBoundaryPolicy = "disabled";
	rawWeakFrameStreak = 0;
	rawFailedFrameStreak = 0;
	rawWeakBoundaryFrames = 3;
	rawFailedBoundaryFrames = 1;
	trackingBoundaryMinPrimaryFrames = 20;
	trackingBoundaryCooldownFrames = 20;
	lastTrackingBoundaryRequestFrame = -1000000;
	trackingBoundaryForceHandoff = true;
	trackingBoundaryHandoffPolicy = "weak_or_failed";
	pagingEnabled = false;
	asyncPageOutSnapshots = false;
	asyncPageInPrefetch = false;
	cpuPagingStorage = false;
	maxResidentMaps = 0;
	maxPendingPageSnapshots = 1;
	maxPendingPageSnapshotBytes = 0;
	maxPendingPageIns = 1;
	maxPendingPageInBytes = 0;
	minLoopSubmapSeparation = 3;
	loopCandidateCooldownFrames = 30;
	pagingStorageMode = "disk";

	multiVisualisationEngine = ITMMultiVisualisationEngineFactory::MakeVisualisationEngine<TVoxel,TIndex>(deviceType);
	renderState_multiscene = NULL;
}

template <typename TVoxel, typename TIndex>
ITMMultiEngine<TVoxel, TIndex>::~ITMMultiEngine(void)
{
	if (renderState_multiscene != NULL) delete renderState_multiscene;

	delete mGlobalAdjustmentEngine;
	delete mActiveDataManager;
	delete mapManager;

	if (renderState_freeview != NULL) delete renderState_freeview;

	delete denseMapper;
	delete trackingController;

	delete tracker;
	delete imuCalibrator;

	delete lowLevelEngine;
	delete viewBuilder;

	if (view != NULL) delete view;

	delete visualisationEngine;

	delete relocaliser;

	delete multiVisualisationEngine;
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::changeFreeviewLocalMapIdx(ORUtils::SE3Pose *pose, int newIdx)
{
	//if ((newIdx < 0) || ((unsigned)newIdx >= mapManager->numLocalMaps())) return;

	if (newIdx < -1) newIdx = (int)mapManager->numLocalMaps() - 1;
	if ((unsigned)newIdx >= mapManager->numLocalMaps()) newIdx = -1;

	ORUtils::SE3Pose trafo = mapManager->findTransformation(freeviewLocalMapIdx, newIdx);
	pose->SetM(pose->GetM() * trafo.GetInvM());
	pose->Coerce();
	freeviewLocalMapIdx = newIdx;
}

template <typename TVoxel, typename TIndex>
ITMTrackingState* ITMMultiEngine<TVoxel, TIndex>::GetTrackingState(void)
{
	int idx = mActiveDataManager->findPrimaryLocalMapIdx();
	if (idx < 0)
		throw std::runtime_error("MultiEngine has no primary local map tracking state");
	ITMLocalMap<TVoxel, TIndex> *localMap = mapManager->getLocalMap(idx);
	if (localMap == NULL || !localMap->isResident() || localMap->trackingState == NULL)
		throw std::runtime_error("MultiEngine primary local map tracking state is not resident: " + std::to_string(idx));
	return localMap->trackingState;
}

template <typename TVoxel, typename TIndex>
ORUtils::Matrix4<float> ITMMultiEngine<TVoxel, TIndex>::getGlobalCameraPose(void) const
{
	int idx = mActiveDataManager->findPrimaryLocalMapIdx();
	if (idx < 0)
		throw std::runtime_error("MultiEngine has no primary local map for global camera pose");
	const ITMLocalMap<TVoxel, TIndex> *localMap = mapManager->getLocalMap(idx);
	if (localMap == NULL || !localMap->isResident() || localMap->trackingState == NULL)
		throw std::runtime_error("MultiEngine primary local map is not resident for global camera pose: " + std::to_string(idx));
	return localMap->estimatedGlobalPose.GetInvM() * localMap->trackingState->pose_d->GetInvM();
}

template <typename TVoxel, typename TIndex>
typename ITMMultiEngine<TVoxel, TIndex>::EffectiveOccupancyThresholds
ITMMultiEngine<TVoxel, TIndex>::getEffectiveOccupancyThresholds(int occupancySampleFrames) const
{
	EffectiveOccupancyThresholds thresholds = {newMapBlockOccupancy, maxBlockOccupancy, false};
	if (occupancyThresholdPolicy != "adaptive_longspan") return thresholds;

	const bool baseHasNewMapLimit = newMapBlockOccupancy > 0.0f;
	const bool baseHasStopLimit = maxBlockOccupancy > 0.0f;
	if (!baseHasNewMapLimit && !baseHasStopLimit) return thresholds;

	const bool primaryOldEnough = primaryMapFrames >= adaptiveOccupancyMinPrimaryFrames;
	const bool runOldEnough = framesProcessed >= adaptiveOccupancyMinProcessedFrames;
	const bool hasOccupancyHistory = lastPrimaryOccupancyFrame >= 0 || occupancySampleFrames > 0;
	const bool trackingHealthy =
		rawWeakFrameStreak <= adaptiveOccupancyMaxRawWeakStreak &&
		rawFailedFrameStreak <= adaptiveOccupancyMaxRawFailedStreak;
	if (!primaryOldEnough || !runOldEnough || !hasOccupancyHistory || !trackingHealthy)
		return thresholds;

	if (baseHasStopLimit && adaptiveMaxBlockOccupancy > 0.0f)
		thresholds.maxOccupancy = MAX(maxBlockOccupancy, MIN(1.0f, adaptiveMaxBlockOccupancy));
	if (baseHasNewMapLimit && adaptiveNewMapBlockOccupancy > 0.0f)
	{
		const float newMapLimit = thresholds.maxOccupancy > 0.0f ? thresholds.maxOccupancy : 1.0f;
		thresholds.newMapOccupancy = MAX(newMapBlockOccupancy, MIN(newMapLimit, adaptiveNewMapBlockOccupancy));
	}
	thresholds.adaptiveActive =
		thresholds.newMapOccupancy != newMapBlockOccupancy ||
		thresholds.maxOccupancy != maxBlockOccupancy;
	return thresholds;
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::configureHybridSubmaps(bool allowNewMaps, bool enableLoopClosure, bool enableGlobalAdjustment,
	float visibleRatioThreshold, int overlapFrames, int trialFrames, int emergencyOverlapFrames, int sampledOriginalBlocks, int maxFramesPerMap, float maxOccupancy,
	float newMapOccupancy, bool enablePaging, int residentMaps, const std::string &pageDirectory, const std::string &pageStorageModeConfigured,
	int minLoopSeparation, int loopCooldownFrames,
			bool asyncPageOut, int maxPendingSnapshots, int64_t maxPendingSnapshotBytes, bool asyncPageIn, int maxPendingPageInsConfigured,
			int64_t maxPendingPageInBytesConfigured, int64_t maxCpuPageCacheBytesConfigured,
			const std::string &sequentialTransformPolicy, float sequentialTransformMaxTranslation,
			float sequentialTransformMaxRotationDeg, int invalidTransformRestartAttempts, const std::string &handoffPosePolicy, const std::string &newMapHandoffPolicy,
			float newMapHandoffMinVisibleRatio, float newMapHandoffMinVisibleMargin,
			int newMapHandoffMinAcceptedWindows, int newMapHandoffEmergencyMinAcceptedWindows,
			const std::string &candidateRecoveryPolicy, float candidateRecoveryMinVisibleRatio, int candidateRecoveryMinBlocks, int candidateRecoveryMaxAttempts,
			const std::string &primaryFailureHandoffPolicy, int primaryFailureHandoffMinRawFailedFrames,
			const std::string &primaryFailureHandoffRawGate, int primaryFailureHandoffMinRawWeakFrames,
			float primaryFailureHandoffMinCandidateVisibleRatio, int primaryFailureHandoffMinCandidateBlocks,
			int primaryFailureHandoffMinCandidateGoodFrames, int primaryFailureHandoffMinOverlapFrames,
			bool primaryFailureHandoffRequirePoseContinuity,
			const std::string &trackingBoundaryPolicyConfigured, int rawWeakBoundaryFramesConfigured, int rawFailedBoundaryFramesConfigured, int trackingBoundaryMinPrimaryFramesConfigured,
			int trackingBoundaryCooldownFramesConfigured, bool trackingBoundaryForceHandoffConfigured,
			const std::string &trackingBoundaryHandoffPolicyConfigured, int fusionReserveBlocksConfigured, int occupancyGrowthLookaheadFramesConfigured,
			int occupancyGrowthMinPrimaryFramesConfigured, int occupancyGrowthSampleFramesConfigured, bool skipPrimaryFusionOnCapacityHandoffConfigured,
			bool allowVisibilityHandoffConfigured, const std::string &occupancyThresholdPolicyConfigured,
			float adaptiveNewMapOccupancyConfigured, float adaptiveMaxOccupancyConfigured,
			int adaptiveOccupancyMinPrimaryFramesConfigured, int adaptiveOccupancyMinProcessedFramesConfigured,
			int adaptiveOccupancyMaxRawWeakStreakConfigured, int adaptiveOccupancyMaxRawFailedStreakConfigured)
{
	loopClosureEnabled = enableLoopClosure;
	globalAdjustmentEnabled = enableGlobalAdjustment;
	allowNewLocalMapsConfigured = allowNewMaps;
	// Historical links are proposed explicitly by the fern detector. The
	// legacy automatic restart of every previous sequential link creates
	// persistent overlap, unnecessary tracking work, and paging churn.
		mActiveDataManager->configureSequentialPolicy(allowNewMaps, false, visibleRatioThreshold, overlapFrames, trialFrames, emergencyOverlapFrames, sampledOriginalBlocks,
				sequentialTransformPolicy, sequentialTransformMaxTranslation, sequentialTransformMaxRotationDeg, invalidTransformRestartAttempts, handoffPosePolicy, newMapHandoffPolicy,
				newMapHandoffMinVisibleRatio, newMapHandoffMinVisibleMargin,
				newMapHandoffMinAcceptedWindows, newMapHandoffEmergencyMinAcceptedWindows,
				candidateRecoveryPolicy, candidateRecoveryMinVisibleRatio, candidateRecoveryMinBlocks, candidateRecoveryMaxAttempts,
				primaryFailureHandoffPolicy, primaryFailureHandoffMinRawFailedFrames,
				primaryFailureHandoffRawGate, primaryFailureHandoffMinRawWeakFrames,
				primaryFailureHandoffMinCandidateVisibleRatio, primaryFailureHandoffMinCandidateBlocks,
				primaryFailureHandoffMinCandidateGoodFrames, primaryFailureHandoffMinOverlapFrames,
				primaryFailureHandoffRequirePoseContinuity,
				allowVisibilityHandoffConfigured);
	mActiveDataManager->configureDiagnostics(std::filesystem::path(pageDirectory).parent_path().string());
	maxFramesPerPrimary = MAX(0, maxFramesPerMap);
	maxBlockOccupancy = MAX(0.0f, MIN(1.0f, maxOccupancy));
	const float newMapOccupancyLimit = maxBlockOccupancy > 0.0f ? maxBlockOccupancy : 1.0f;
	newMapBlockOccupancy = newMapOccupancy;
	if (newMapBlockOccupancy < 0.0f) newMapBlockOccupancy = 0.0f;
	if (newMapBlockOccupancy > newMapOccupancyLimit) newMapBlockOccupancy = newMapOccupancyLimit;
	occupancyThresholdPolicy = occupancyThresholdPolicyConfigured == "adaptive_longspan" ? occupancyThresholdPolicyConfigured : "fixed";
	adaptiveMaxBlockOccupancy = MAX(0.0f, MIN(1.0f, adaptiveMaxOccupancyConfigured));
	const float adaptiveNewMapOccupancyLimit = adaptiveMaxBlockOccupancy > 0.0f ? adaptiveMaxBlockOccupancy : 1.0f;
	adaptiveNewMapBlockOccupancy = MAX(0.0f, MIN(adaptiveNewMapOccupancyLimit, adaptiveNewMapOccupancyConfigured));
	adaptiveOccupancyMinPrimaryFrames = MAX(0, adaptiveOccupancyMinPrimaryFramesConfigured);
	adaptiveOccupancyMinProcessedFrames = MAX(0, adaptiveOccupancyMinProcessedFramesConfigured);
	adaptiveOccupancyMaxRawWeakStreak = MAX(0, adaptiveOccupancyMaxRawWeakStreakConfigured);
	adaptiveOccupancyMaxRawFailedStreak = MAX(0, adaptiveOccupancyMaxRawFailedStreakConfigured);
	fusionReserveBlocks = MAX(0, fusionReserveBlocksConfigured);
	lastOccupancyRequestMapIdx = -1;
	lastOccupancyRequestFrame = -1;
	lastPrimaryOccupancyFrame = -1;
	lastPrimaryOccupancyBlocks = -1;
	occupancyGrowthLookaheadFrames = MAX(0, occupancyGrowthLookaheadFramesConfigured);
	occupancyGrowthMinPrimaryFrames = MAX(0, occupancyGrowthMinPrimaryFramesConfigured);
	occupancyGrowthSampleFrames = MAX(1, occupancyGrowthSampleFramesConfigured);
	lastCapacityDeferMapIdx = -1;
	lastCapacityDeferFrame = -1;
	lastPrimaryHandoffFusionSkipMapIdx = -1;
	lastPrimaryHandoffFusionSkipFrame = -1;
	capacityLimitedPrimaryMapIdx = -1;
	skipPrimaryFusionOnCapacityHandoff = skipPrimaryFusionOnCapacityHandoffConfigured;
	trackingBoundaryPolicy = trackingBoundaryPolicyConfigured;
	rawWeakFrameStreak = 0;
	rawFailedFrameStreak = 0;
	rawWeakBoundaryFrames = MAX(1, rawWeakBoundaryFramesConfigured);
	rawFailedBoundaryFrames = MAX(1, rawFailedBoundaryFramesConfigured);
	trackingBoundaryMinPrimaryFrames = MAX(0, trackingBoundaryMinPrimaryFramesConfigured);
	trackingBoundaryCooldownFrames = MAX(0, trackingBoundaryCooldownFramesConfigured);
	lastTrackingBoundaryRequestFrame = -1000000;
	trackingBoundaryForceHandoff = trackingBoundaryForceHandoffConfigured;
	trackingBoundaryHandoffPolicy = trackingBoundaryHandoffPolicyConfigured;
	pagingEnabled = enablePaging;
	asyncPageOutSnapshots = asyncPageOut;
	asyncPageInPrefetch = asyncPageIn;
	pagingStorageMode = pageStorageModeConfigured;
	if (pagingStorageMode != "cpu") pagingStorageMode = "disk";
	cpuPagingStorage = pagingStorageMode == "cpu";
	maxResidentMaps = MAX(1, residentMaps);
	maxPendingPageSnapshots = MAX(0, maxPendingSnapshots);
	maxPendingPageSnapshotBytes = MAX(static_cast<int64_t>(0), maxPendingSnapshotBytes);
	maxPendingPageIns = MAX(0, maxPendingPageInsConfigured);
	maxPendingPageInBytes = MAX(static_cast<int64_t>(0), maxPendingPageInBytesConfigured);
	maxCpuPageCacheBytes = MAX(static_cast<int64_t>(0), maxCpuPageCacheBytesConfigured);
	minLoopSubmapSeparation = MAX(1, minLoopSeparation);
	loopCandidateCooldownFrames = MAX(0, loopCooldownFrames);
	pagingDirectory = pageDirectory;
	if (pagingEnabled) {
		std::filesystem::remove_all(pagingDirectory);
		std::filesystem::create_directories(pagingDirectory);
		const std::filesystem::path logPath = std::filesystem::path(pagingDirectory).parent_path() / "paging.csv";
		pagingLog.open(logPath.string());
		if (!pagingLog) throw std::runtime_error("Could not create paging diagnostics: " + logPath.string());
		pagingLog << "frame,action,submap_id,duration_ms,resident_count,pending_snapshots,pending_pageins,"
			<< "resident_estimated_bytes,pending_estimated_bytes,pending_pagein_estimated_bytes,"
			<< "passive_cpu_cache_estimated_bytes,cpu_page_cache_limit_bytes,blocks,capacity,page_bytes,storage_mode\n";
		pagingLog.flush();
	}
	if (loopClosureEnabled)
	{
		const std::filesystem::path logPath = std::filesystem::path(pageDirectory).parent_path() / "loop_events.csv";
		loopLog.open(logPath.string());
		if (!loopLog) throw std::runtime_error("Could not create loop diagnostics: " + logPath.string());
		loopLog << "frame,action,submap_id,distance,resident_count\n";
		loopLog.flush();
	}
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::logLoopEvent(const char *action, int localMapIdx, float distance)
{
	if (!loopLog) return;
	loopLog << framesProcessed << "," << action << "," << localMapIdx << "," << distance
		<< "," << mapManager->countResidentLocalMaps() << "\n";
	loopLog.flush();
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::logPagingEvent(const char *action, int localMapIdx, double durationMs, int blockCount, int capacity, int64_t pageBytes)
{
	if (!pagingLog) return;
	pagingLog << framesProcessed << "," << action << "," << localMapIdx << "," << durationMs
		<< "," << mapManager->countResidentLocalMaps()
		<< "," << mapManager->countPendingLocalMapSnapshots()
		<< "," << mapManager->countPendingLocalMapPageIns()
		<< "," << mapManager->getResidentLocalMapEstimatedBytes()
		<< "," << mapManager->getPendingLocalMapSnapshotEstimatedBytes()
		<< "," << mapManager->getPendingLocalMapPageInEstimatedBytes()
		<< "," << getPassiveCpuPageCacheEstimatedBytes()
		<< "," << maxCpuPageCacheBytes
		<< "," << blockCount
		<< "," << capacity
		<< "," << pageBytes
		<< "," << pagingStorageMode << "\n";
	pagingLog.flush();
}

static int64_t directorySizeBytes(const std::filesystem::path &directory)
{
	if (directory.empty() || !std::filesystem::exists(directory)) return -1;
	int64_t total = 0;
	for (const auto &entry : std::filesystem::recursive_directory_iterator(directory))
	{
		if (!entry.is_regular_file()) continue;
		std::error_code ec;
		const auto size = entry.file_size(ec);
		if (!ec) total += static_cast<int64_t>(size);
	}
	return total;
}

template <typename TVoxel, typename TIndex>
int64_t ITMMultiEngine<TVoxel, TIndex>::getPassiveCpuPageCacheEstimatedBytes(void) const
{
	int64_t total = 0;
	for (int i = 0; i < (int)mapManager->numLocalMaps(); ++i)
	{
		const ITMLocalMap<TVoxel, TIndex> *localMap = mapManager->getLocalMap(i);
		if (localMap == NULL || localMap->isResident()) continue;
		total += localMap->cpuPageCacheEstimatedBytes();
	}
	return total;
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::enforceCpuPageCacheBudget(const std::vector<int> &requiredMaps)
{
	if (!pagingEnabled || !cpuPagingStorage || maxCpuPageCacheBytes <= 0) return;
	int64_t cachedBytes = getPassiveCpuPageCacheEstimatedBytes();
	if (cachedBytes <= maxCpuPageCacheBytes) return;

	std::vector<int> protectedMaps;
	protectedMaps.reserve(requiredMaps.size() + extraResidencyPins.size());
	for (int localMapIdx : requiredMaps)
	{
		if (localMapIdx < 0 || localMapIdx >= (int)mapManager->numLocalMaps())
			continue;
		if (std::find(protectedMaps.begin(), protectedMaps.end(), localMapIdx) == protectedMaps.end())
			protectedMaps.push_back(localMapIdx);
	}
	for (int localMapIdx : extraResidencyPins)
	{
		if (localMapIdx < 0 || localMapIdx >= (int)mapManager->numLocalMaps())
			continue;
		if (std::find(protectedMaps.begin(), protectedMaps.end(), localMapIdx) == protectedMaps.end())
			protectedMaps.push_back(localMapIdx);
	}

	for (int localMapIdx = 0; localMapIdx < (int)mapManager->numLocalMaps() && cachedBytes > maxCpuPageCacheBytes; ++localMapIdx)
	{
		if (std::find(protectedMaps.begin(), protectedMaps.end(), localMapIdx) != protectedMaps.end())
			continue;
		ITMLocalMap<TVoxel, TIndex> *localMap = mapManager->getLocalMap(localMapIdx);
		if (localMap == NULL || localMap->isResident() || !localMap->hasCpuPageCache())
			continue;

		const int blocks = mapManager->getLocalMapSize(localMapIdx);
		const int capacity = SDF_LOCAL_BLOCK_NUM;
		const std::string directory = pagingDirectory + "/submap_" + std::to_string(localMapIdx);
		const auto start = std::chrono::steady_clock::now();
		const int64_t spilledBytes = localMap->spillCpuPageCacheToDirectory(directory);
		const double durationMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		cachedBytes = getPassiveCpuPageCacheEstimatedBytes();
		const int64_t pageBytes = directorySizeBytes(directory);
		logPagingEvent("cpu_cache_spill", localMapIdx, durationMs, blocks, capacity, pageBytes);
		MULTI_ENGINE_CONSOLE("[PAGING] cpu_cache_spill map=%d released_mb=%.1f passive_cache_mb=%.1f limit_mb=%.1f path=%s\n",
			localMapIdx,
			(double)spilledBytes / (1024.0 * 1024.0),
			(double)cachedBytes / (1024.0 * 1024.0),
			(double)maxCpuPageCacheBytes / (1024.0 * 1024.0),
			directory.c_str());
	}

	if (cachedBytes > maxCpuPageCacheBytes)
		logPagingEvent("cpu_cache_budget_exceeded", -1, 0.0, 0, SDF_LOCAL_BLOCK_NUM, -1);
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::ensureLocalMapResident(int localMapIdx)
{
	if (mapManager->isLocalMapResident(localMapIdx)) return;
	const auto start = std::chrono::steady_clock::now();
	mapManager->pageInLocalMap(localMapIdx);
	const double durationMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	const int blocks = mapManager->getLocalMapSize(localMapIdx);
	const int capacity = mapManager->getLocalMap(localMapIdx)->scene->index.getNumAllocatedVoxelBlocks();
	const int64_t pageBytes = directorySizeBytes(mapManager->getLocalMapPageDirectory(localMapIdx));
	logPagingEvent("page_in", localMapIdx, durationMs, blocks, capacity, pageBytes);
	MULTI_ENGINE_CONSOLE("[PAGING] page_in map=%d resident=%d\n", localMapIdx, mapManager->countResidentLocalMaps());
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::prefetchLocalMaps(const std::vector<int> &localMapIndices)
{
	if (!pagingEnabled || !asyncPageInPrefetch) return;
	auto synchronizePageIns = [&]() {
		for (int i = 0; i < (int)mapManager->numLocalMaps(); ++i)
			mapManager->waitForLocalMapPageIn(i);
	};
	for (int localMapIdx : localMapIndices)
	{
		if (localMapIdx < 0 || localMapIdx >= (int)mapManager->numLocalMaps())
			continue;
		if (mapManager->isLocalMapResident(localMapIdx))
			continue;
		const bool countLimitHit =
			maxPendingPageIns > 0 && mapManager->countPendingLocalMapPageIns() >= maxPendingPageIns;
		const bool memoryLimitHit =
			maxPendingPageInBytes > 0 &&
			mapManager->getPendingLocalMapPageInEstimatedBytes() >= maxPendingPageInBytes;
		if (countLimitHit || memoryLimitHit)
			synchronizePageIns();
		const int blocks = mapManager->getLocalMapSize(localMapIdx);
		const int capacity = SDF_LOCAL_BLOCK_NUM;
		const auto start = std::chrono::steady_clock::now();
		mapManager->prefetchLocalMap(localMapIdx);
		const double durationMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		const int64_t pageBytes = directorySizeBytes(mapManager->getLocalMapPageDirectory(localMapIdx));
		logPagingEvent("page_in_prefetch", localMapIdx, durationMs, blocks, capacity, pageBytes);
	}
}

template <typename TVoxel, typename TIndex>
bool ITMMultiEngine<TVoxel, TIndex>::tryAdoptPrefetchedRaycastMaps(const std::vector<int> &localMapIndices)
{
	if (!pagingEnabled) return true;
	bool allResident = true;
	for (int localMapIdx : localMapIndices)
	{
		if (localMapIdx < 0 || localMapIdx >= (int)mapManager->numLocalMaps())
			continue;
		if (mapManager->isLocalMapResident(localMapIdx))
			continue;
		if (mapManager->tryCompleteLocalMapPageIn(localMapIdx))
			continue;
		allResident = false;
	}
	return allResident;
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::setExtraResidencyPins(const std::vector<int> &localMapIndices)
{
	extraResidencyPins.clear();
	for (int localMapIdx : localMapIndices)
	{
		if (localMapIdx < 0 || localMapIdx >= (int)mapManager->numLocalMaps())
			continue;
		if (std::find(extraResidencyPins.begin(), extraResidencyPins.end(), localMapIdx) == extraResidencyPins.end())
			extraResidencyPins.push_back(localMapIdx);
	}
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::enforceResidency(int pinnedLocalMapIdx)
{
	if (!pagingEnabled) return;
	std::vector<int> requiredMaps;
	if (pinnedLocalMapIdx >= 0)
	{
		const int primary = findPrimaryLocalMapIdx();
		if (primary >= 0) requiredMaps.push_back(primary);
		if (std::find(requiredMaps.begin(), requiredMaps.end(), pinnedLocalMapIdx) == requiredMaps.end())
			requiredMaps.push_back(pinnedLocalMapIdx);
	}
	else requiredMaps = mActiveDataManager->getActiveLocalMapIndices();
	enforceResidency(requiredMaps);
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::enforceResidency(const std::vector<int> &requiredMaps)
{
	if (!pagingEnabled) return;
	std::vector<int> requiredWithPins;
	requiredWithPins.reserve(requiredMaps.size() + extraResidencyPins.size());
	for (int localMapIdx : requiredMaps)
	{
		if (localMapIdx < 0 || localMapIdx >= (int)mapManager->numLocalMaps())
			continue;
		if (std::find(requiredWithPins.begin(), requiredWithPins.end(), localMapIdx) == requiredWithPins.end())
			requiredWithPins.push_back(localMapIdx);
	}
	for (int localMapIdx : extraResidencyPins)
	{
		if (localMapIdx < 0 || localMapIdx >= (int)mapManager->numLocalMaps())
			continue;
		if (std::find(requiredWithPins.begin(), requiredWithPins.end(), localMapIdx) == requiredWithPins.end())
			requiredWithPins.push_back(localMapIdx);
	}
	const int required = MAX(maxResidentMaps, (int)requiredWithPins.size());
	auto evictOneNonRequired = [&]() {
		int victim = -1;
		const std::vector<int> resident = mapManager->getResidentLocalMapIds();
		for (size_t i = 0; i < resident.size(); ++i)
			if (std::find(requiredWithPins.begin(), requiredWithPins.end(), resident[i]) == requiredWithPins.end()) { victim = resident[i]; break; }
		if (victim < 0) return false;
		const bool countLimitHit =
			maxPendingPageSnapshots > 0 && mapManager->countPendingLocalMapSnapshots() >= maxPendingPageSnapshots;
		const bool memoryLimitHit =
			maxPendingPageSnapshotBytes > 0 &&
			mapManager->getPendingLocalMapSnapshotEstimatedBytes() >= maxPendingPageSnapshotBytes;
		if (!asyncPageOutSnapshots || countLimitHit || memoryLimitHit)
		{
			const int blocks = mapManager->getLocalMapSize(victim);
			const int capacity = mapManager->getLocalMap(victim)->scene->index.getNumAllocatedVoxelBlocks();
			const auto waitStart = std::chrono::steady_clock::now();
			// Bound hidden GPU memory held by detached scenes while their CPU
			// snapshots are still being copied. Do not wait pending page-ins
			// here: they are adopted and timed by ensureLocalMapResident().
			for (int i = 0; i < (int)mapManager->numLocalMaps(); ++i)
				mapManager->waitForLocalMapSnapshot(i);
			const double waitMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waitStart).count();
				if (waitMs > 0.01)
					logPagingEvent("page_out_backpressure_wait", victim, waitMs, blocks, capacity, -1);
			}
			enforceCpuPageCacheBudget(requiredWithPins);
			const std::string directory = pagingDirectory + "/submap_" + std::to_string(victim);
			const int blocks = mapManager->getLocalMapSize(victim);
			const int capacity = mapManager->getLocalMap(victim)->scene->index.getNumAllocatedVoxelBlocks();
			const auto start = std::chrono::steady_clock::now();
			mapManager->pageOutLocalMap(victim, directory, asyncPageOutSnapshots, cpuPagingStorage);
		const double durationMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		// The CPU snapshot is serialized asynchronously, so exact bytes are
		// reported by validation from the completed page directory.
		logPagingEvent("page_out", victim, durationMs, blocks, capacity, -1);
			MULTI_ENGINE_CONSOLE("[PAGING] page_out map=%d resident=%d pending=%d async=%d path=%s\n",
				victim, mapManager->countResidentLocalMaps(), mapManager->countPendingLocalMapSnapshots(),
				asyncPageOutSnapshots ? 1 : 0, directory.c_str());
			enforceCpuPageCacheBudget(requiredWithPins);
			return true;
		};
	for (size_t i = 0; i < requiredWithPins.size(); ++i)
	{
		if (requiredWithPins[i] < 0 || requiredWithPins[i] >= (int)mapManager->numLocalMaps())
			continue;
		if (!mapManager->isLocalMapResident(requiredWithPins[i]) &&
			mapManager->countResidentLocalMaps() >= required)
			evictOneNonRequired();
		ensureLocalMapResident(requiredWithPins[i]);
	}
		while (mapManager->countResidentLocalMaps() > required)
			if (!evictOneNonRequired()) break;
		enforceCpuPageCacheBudget(requiredWithPins);
	}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::synchronizePaging(void)
{
	if (!pagingEnabled) return;
	for (int i = 0; i < (int)mapManager->numLocalMaps(); ++i)
		mapManager->waitForLocalMapPageIn(i);
	for (int i = 0; i < (int)mapManager->numLocalMaps(); ++i)
		mapManager->waitForLocalMapSnapshot(i);
	enforceCpuPageCacheBudget(std::vector<int>());
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::turnOffTracking(void)
{
	trackingActive = false;
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::turnOnTracking(void)
{
	trackingActive = true;
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::configureGroundTruthPoses(const std::vector<ORUtils::Matrix4<float> *> &poses)
{
	gtC2wPoses = poses;
	trackingActive = false;
}

// -whenever a new local scene is added, add to list of "to be established 3D relations"
// - whenever a relocalisation is detected, add to the same list, preserving any existing information on that 3D relation
//
// - for all 3D relations to be established :
// -attempt tracking in both scenes
// - if success, add to list of new candidates
// - if less than n_overlap "new candidates" in more than n_reloctrialframes frames, discard
// - if at least n_overlap "new candidates" :
// 	- try to compute 3D relation, weighting old information accordingly
//	- if outlier ratio below p_relation_outliers and at least n_overlap inliers, success

struct TodoListEntry {
	TodoListEntry(int _activeDataID, bool _track, bool _fusion, bool _prepare)
		: dataId(_activeDataID), track(_track), fusion(_fusion), prepare(_prepare), preprepare(false) {}
	TodoListEntry(void) {}
	int dataId;
	bool track;
	bool fusion;
	bool prepare;
	bool preprepare;
};

template <typename TVoxel, typename TIndex>
ITMTrackingState::TrackingResult ITMMultiEngine<TVoxel, TIndex>::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage, ITMIMUMeasurement *imuMeasurement)
{
	std::vector<TodoListEntry> todoList;
	ITMTrackingState::TrackingResult primaryLocalMapTrackingResult = ITMTrackingState::TRACKING_FAILED;
	lastPrimaryRawTrackingResult = ITMTrackingState::TRACKING_FAILED;
	lastPrimaryPolicyTrackingResult = ITMTrackingState::TRACKING_FAILED;
	lastPrimaryPoseRollback = false;
	lastPrimaryFusionSuppressed = false;

	// prepare image and turn it into a depth image
	if (imuMeasurement == NULL) viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useBilateralFilter);
	else viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useBilateralFilter, imuMeasurement);

	// find primary data, if available
	int primaryDataIdx = mActiveDataManager->findPrimaryDataIdx();

	// if there is a "primary data index", process it
	if (primaryDataIdx >= 0) todoList.push_back(TodoListEntry(primaryDataIdx, true, true, true));

	// after primary local map, make sure to process all relocalisations, new scenes and loop closures
	for (int i = 0; i < mActiveDataManager->numActiveLocalMaps(); ++i)
	{
		switch (mActiveDataManager->getLocalMapType(i))
		{
		case ITMActiveMapManager::NEW_LOCAL_MAP: todoList.push_back(TodoListEntry(i, true, true, true)); break;
		case ITMActiveMapManager::LOOP_CLOSURE: todoList.push_back(TodoListEntry(i, true, false, true)); break;
		case ITMActiveMapManager::RELOCALISATION: todoList.push_back(TodoListEntry(i, true, false, true)); break;
		default: break;
		}
	}

	// finally, once all is done, call the loop closure detection engine
	todoList.push_back(TodoListEntry(-1, false, false, false));

	bool primaryTrackingSuccess = false;
	for (size_t i = 0; i < todoList.size(); ++i)
	{
		// - first pass of the todo list is for primary local map and ongoing relocalisation and loopclosure attempts
		// - an element with id -1 marks the end of the first pass, a request to call the loop closure detection engine, and
		//   the start of the second pass
		// - second tracking pass will be about newly detected loop closures, relocalisations, etc.

		if (todoList[i].dataId == -1)
		{
			if (!loopClosureEnabled) continue;
#ifdef DEBUG_MULTISCENE
			fprintf(stderr, " Reloc(%i)", primaryTrackingSuccess);
#endif
			int NN[k_loopcloseneighbours]; float distances[k_loopcloseneighbours];
			view->depth->UpdateHostFromDevice();

			//primary map index
			int primaryLocalMapIdx = -1;
			if (primaryDataIdx >= 0) primaryLocalMapIdx = mActiveDataManager->getLocalMapIndex(primaryDataIdx);

			//check if relocaliser has fired
			ORUtils::SE3Pose *pose = primaryLocalMapIdx >= 0 ? mapManager->getLocalMap(primaryLocalMapIdx)->trackingState->pose_d : NULL;
			bool hasAddedKeyframe = relocaliser->ProcessFrame(view->depth, pose, primaryLocalMapIdx, k_loopcloseneighbours, NN, distances, primaryTrackingSuccess);

			//frame not added and tracking failed -> we need to relocalise
			if (!hasAddedKeyframe)
			{
				for (int j = 0; j < k_loopcloseneighbours; ++j)
				{
					if (distances[j] > F_maxdistattemptreloc) continue;
					const FernRelocLib::PoseDatabase::PoseInScene & keyframe = relocaliser->RetrievePose(NN[j]);
					logLoopEvent("proposal", keyframe.sceneIdx, distances[j]);
					if (primaryLocalMapIdx >= 0 && std::abs(keyframe.sceneIdx - primaryLocalMapIdx) < minLoopSubmapSeparation)
					{
						logLoopEvent("rejected_sequential", keyframe.sceneIdx, distances[j]);
						continue;
					}
					auto previousAttempt = loopCandidateLastAttempt.find(keyframe.sceneIdx);
					if (previousAttempt != loopCandidateLastAttempt.end() &&
						framesProcessed - previousAttempt->second < loopCandidateCooldownFrames)
					{
						logLoopEvent("rejected_cooldown", keyframe.sceneIdx, distances[j]);
						continue;
					}
					if (!mActiveDataManager->canInitiateNewLink(keyframe.sceneIdx))
					{
						logLoopEvent("rejected_active", keyframe.sceneIdx, distances[j]);
						continue;
					}
					ensureLocalMapResident(keyframe.sceneIdx);
					int newDataIdx = mActiveDataManager->initiateNewLink(keyframe.sceneIdx, keyframe.pose, (primaryLocalMapIdx < 0));
					if (newDataIdx >= 0)
					{
						loopCandidateLastAttempt[keyframe.sceneIdx] = framesProcessed;
						logLoopEvent("accepted_tracking", keyframe.sceneIdx, distances[j]);
						TodoListEntry todoItem(newDataIdx, true, false, true);
						todoItem.preprepare = true;
						todoList.push_back(todoItem);
					}
					else logLoopEvent("rejected_reset", keyframe.sceneIdx, distances[j]);
				}
			}

			continue;
		}

		ITMLocalMap<TVoxel, TIndex> *currentLocalMap = NULL;
		int currentLocalMapIdx = mActiveDataManager->getLocalMapIndex(todoList[i].dataId);
		ensureLocalMapResident(currentLocalMapIdx);
		currentLocalMap = mapManager->getLocalMap(currentLocalMapIdx);

		// if a new relocalisation/loopclosure is started, this will do the initial raycasting before tracking can start
		if (todoList[i].preprepare) 
		{
			denseMapper->UpdateVisibleList(view, currentLocalMap->trackingState, currentLocalMap->scene, currentLocalMap->renderState);
			trackingController->Prepare(currentLocalMap->trackingState, currentLocalMap->scene, view, visualisationEngine, currentLocalMap->renderState);
		}

		if (todoList[i].track)
		{
			int dataId = todoList[i].dataId;

#ifdef DEBUG_MULTISCENE
			int blocksInUse = currentLocalMap->scene->index.getNumAllocatedVoxelBlocks() - currentLocalMap->scene->localVBA.lastFreeBlockId - 1;
			fprintf(stderr, " %i%s (%i)", currentLocalMapIdx, (todoList[i].dataId == primaryDataIdx) ? "*" : "", blocksInUse);
#endif

			// actual tracking
			ORUtils::SE3Pose oldPose(*(currentLocalMap->trackingState->pose_d));
			const bool oldPoseFinite = isFinitePose(&oldPose);
			bool nonFinitePoseRolledBack = false;
			if (trackingActive)
			{
				trackingController->Track(currentLocalMap->trackingState, view);
			}
			else if (framesProcessed < (int)gtC2wPoses.size())
			{
				Matrix4f localC2w = currentLocalMap->estimatedGlobalPose.GetM() * (*gtC2wPoses[framesProcessed]);
				currentLocalMap->trackingState->pose_d->SetInvM(localC2w);
				currentLocalMap->trackingState->pose_d->Coerce();
				currentLocalMap->trackingState->trackerResult = ITMTrackingState::TRACKING_GOOD;
			}
			if (!isFinitePose(currentLocalMap->trackingState->pose_d))
			{
				currentLocalMap->trackingState->trackerResult = ITMTrackingState::TRACKING_FAILED;
				if (oldPoseFinite)
				{
					*(currentLocalMap->trackingState->pose_d) = oldPose;
					nonFinitePoseRolledBack = true;
				}
				MULTI_ENGINE_CONSOLE(
					"[MULTI TRACKING] non_finite_pose_reject frame=%d map=%d data=%d old_pose_finite=%d\n",
					framesProcessed, currentLocalMapIdx, dataId, oldPoseFinite ? 1 : 0);
			}

			// Match the BasicEngine failure policy for the primary tracker, but
			// keep secondary map tracking strict so bad overlap observations do
			// not become submap constraints.
			const ITMActiveMapManager::LocalMapActivity mapType = mActiveDataManager->getLocalMapType(dataId);
			const ITMTrackingState::TrackingResult rawTrackingResult = currentLocalMap->trackingState->trackerResult;
			ITMTrackingState::TrackingResult trackingResult = rawTrackingResult;
			if (mapType == ITMActiveMapManager::PRIMARY_LOCAL_MAP)
			{
				lastPrimaryRawTrackingResult = rawTrackingResult;
				const ITMTrackingFailureDecision failureDecision =
					DecideTrackingFailure(rawTrackingResult, settings->behaviourOnFailure);
				trackingResult = failureDecision.policyResult;
				if (failureDecision.rollbackPose)
				{
					if (oldPoseFinite)
					{
						*(currentLocalMap->trackingState->pose_d) = oldPose;
						lastPrimaryPoseRollback = true;
					}
					else trackingResult = ITMTrackingState::TRACKING_FAILED;
				}
				lastPrimaryPoseRollback = lastPrimaryPoseRollback || nonFinitePoseRolledBack;
				lastPrimaryFusionSuppressed = failureDecision.suppressFusion;
				lastPrimaryPolicyTrackingResult = trackingResult;

				if (trackingBoundaryPolicy == "raw_weak_streak")
				{
					if (rawTrackingResult != ITMTrackingState::TRACKING_GOOD) rawWeakFrameStreak++;
					else rawWeakFrameStreak = 0;
					if (rawTrackingResult == ITMTrackingState::TRACKING_FAILED) rawFailedFrameStreak++;
					else rawFailedFrameStreak = 0;

					const bool weakEnough = rawWeakFrameStreak >= rawWeakBoundaryFrames;
					const bool failedEnough = rawFailedFrameStreak >= rawFailedBoundaryFrames;
					const bool primaryOldEnough = primaryMapFrames >= trackingBoundaryMinPrimaryFrames;
					const bool cooldownReady =
						lastTrackingBoundaryRequestFrame < 0 ||
						framesProcessed - lastTrackingBoundaryRequestFrame >= trackingBoundaryCooldownFrames;
					const bool forceHandoffByPolicy =
						trackingBoundaryForceHandoff &&
						((trackingBoundaryHandoffPolicy == "weak_or_failed" && (weakEnough || failedEnough)) ||
						 (trackingBoundaryHandoffPolicy == "failed_only" && failedEnough));
					// A hard failure must escalate an already-warming candidate even
					// during the new-map cooldown. Otherwise a weak-frame request can
					// leave a viable recovery map unpromotable for the entire cooldown.
					if (allowNewLocalMapsConfigured && primaryOldEnough && forceHandoffByPolicy &&
						!mActiveDataManager->isPrimaryHandoffRequested())
					{
						mActiveDataManager->requestPrimaryHandoff();
						MULTI_ENGINE_CONSOLE(
							"[MULTI TRACKING] boundary_escalate frame=%d map=%d raw=%d failed_streak=%d active_candidate=%d cooldown_ready=%d\n",
							framesProcessed, currentLocalMapIdx, rawTrackingResult, rawFailedFrameStreak,
							mActiveDataManager->hasActiveNewLocalMap() ? 1 : 0, cooldownReady ? 1 : 0);
					}
					const bool failedCandidateRestart =
						failedEnough && !mActiveDataManager->hasActiveNewLocalMap();
					const bool boundaryRequestReady = cooldownReady || failedCandidateRestart;
					if (allowNewLocalMapsConfigured && (weakEnough || failedEnough) && primaryOldEnough && boundaryRequestReady)
					{
						mActiveDataManager->requestNewArea();
						lastTrackingBoundaryRequestFrame = framesProcessed;
						MULTI_ENGINE_CONSOLE(
							"[MULTI TRACKING] boundary_request frame=%d map=%d raw=%d policy=%d weak_streak=%d failed_streak=%d min_primary_frames=%d cooldown=%d force_handoff=%d handoff_policy=%s\n",
							framesProcessed, currentLocalMapIdx, rawTrackingResult, trackingResult, rawWeakFrameStreak, rawFailedFrameStreak,
							trackingBoundaryMinPrimaryFrames, trackingBoundaryCooldownFrames, forceHandoffByPolicy ? 1 : 0,
							trackingBoundaryHandoffPolicy.c_str());
					}
				}
				else
				{
					rawWeakFrameStreak = 0;
					rawFailedFrameStreak = 0;
				}
			}
			else if (trackingResult == ITMTrackingState::TRACKING_POOR)
				trackingResult = ITMTrackingState::TRACKING_FAILED;

			// actions on tracking result for all scenes TODO: incorporate behaviour on tracking failure from settings
			if (trackingResult != ITMTrackingState::TRACKING_GOOD) todoList[i].fusion = false;

			if (trackingResult == ITMTrackingState::TRACKING_FAILED)
			{
				todoList[i].prepare = false;
				if (oldPoseFinite) *(currentLocalMap->trackingState->pose_d) = oldPose;
			}

			// actions on tracking result for primary local map
			if (mapType == ITMActiveMapManager::PRIMARY_LOCAL_MAP)
			{
				primaryLocalMapTrackingResult = trackingResult;

				if (trackingResult == ITMTrackingState::TRACKING_GOOD) primaryTrackingSuccess = true;

				// we need to relocalise in the primary local map
				else if (trackingResult == ITMTrackingState::TRACKING_FAILED)
				{
					primaryDataIdx = -1;
					todoList.resize(i + 1);
					todoList.push_back(TodoListEntry(-1, false, false, false));
				}
			}

			mActiveDataManager->recordTrackingResult(dataId, trackingResult, primaryTrackingSuccess);
			mActiveDataManager->recordRawTrackingResult(dataId, rawTrackingResult);
		}

		bool retiredCurrentMapForCapacity = false;
		auto stopFusionForCapacity = [&](const char *phase) -> bool {
			retiredCurrentMapForCapacity = false;
			const int blocks = mapManager->getLocalMapSize(currentLocalMapIdx);
			const int capacity = currentLocalMap->scene->index.getNumAllocatedVoxelBlocks();
			const float occupancy = capacity > 0 ? (float)blocks / (float)capacity : 1.0f;
			const ITMActiveMapManager::LocalMapActivity mapType =
				mActiveDataManager->getLocalMapType(todoList[i].dataId);
			const int occupancySampleFrames =
				lastPrimaryOccupancyFrame >= 0 ? MAX(1, framesProcessed - lastPrimaryOccupancyFrame) : 0;
			EffectiveOccupancyThresholds effectiveThresholds = {newMapBlockOccupancy, maxBlockOccupancy, false};
			if (mapType == ITMActiveMapManager::PRIMARY_LOCAL_MAP)
				effectiveThresholds = getEffectiveOccupancyThresholds(occupancySampleFrames);
			const int reserveBlocks = capacity > 0
				? MAX(64, fusionReserveBlocks > 0 ? fusionReserveBlocks : capacity / 16)
				: 0;
			const bool reachedConfiguredLimit =
				effectiveThresholds.maxOccupancy > 0.0f && capacity > 0 && occupancy >= effectiveThresholds.maxOccupancy;
			const bool reachedReserveLimit =
				capacity > 0 && reserveBlocks > 0 && blocks >= capacity - reserveBlocks;
			const bool reachedHardLimit = capacity > 0 && (blocks >= capacity - 64 || occupancy >= 0.98f);
			if (!reachedConfiguredLimit && !reachedReserveLimit && !reachedHardLimit) return false;

			const bool isPrimaryLocalMap = mapType == ITMActiveMapManager::PRIMARY_LOCAL_MAP;
			const bool isNewLocalMap = mapType == ITMActiveMapManager::NEW_LOCAL_MAP;
			const char *reason =
				reachedHardLimit ? "hard_limit" : (reachedReserveLimit ? "reserve_limit" : "configured_limit");
			const bool shouldLog =
				currentLocalMapIdx != lastCapacityDeferMapIdx ||
				lastCapacityDeferFrame < 0 ||
				framesProcessed - lastCapacityDeferFrame >= 30;

			if (isPrimaryLocalMap && allowNewLocalMapsConfigured)
			{
				capacityLimitedPrimaryMapIdx = currentLocalMapIdx;
				mActiveDataManager->requestPrimaryHandoff();
				mActiveDataManager->requestNewArea();
				if (shouldLog)
				{
					MULTI_ENGINE_CONSOLE(
						"[MULTI CAPACITY] defer_fusion phase=%s reason=%s frame=%d map=%d blocks=%d capacity=%d occupancy=%.6f reserve=%d stop_threshold=%.6f adaptive=%d policy=%s\n",
						phase, reason, framesProcessed, currentLocalMapIdx, blocks, capacity, occupancy, reserveBlocks,
						effectiveThresholds.maxOccupancy, effectiveThresholds.adaptiveActive ? 1 : 0, occupancyThresholdPolicy.c_str());
					lastCapacityDeferMapIdx = currentLocalMapIdx;
					lastCapacityDeferFrame = framesProcessed;
				}
				return true;
			}

			if (isNewLocalMap && allowNewLocalMapsConfigured && (reachedReserveLimit || reachedHardLimit))
			{
				mActiveDataManager->retireNewLocalMap(currentLocalMapIdx, std::string(phase) + "_" + reason);
				retiredCurrentMapForCapacity = true;
				if (shouldLog)
				{
					MULTI_ENGINE_CONSOLE(
						"[MULTI CAPACITY] retire_candidate phase=%s reason=%s frame=%d map=%d blocks=%d capacity=%d occupancy=%.6f reserve=%d stop_threshold=%.6f adaptive=%d policy=%s\n",
						phase, reason, framesProcessed, currentLocalMapIdx, blocks, capacity, occupancy, reserveBlocks,
						effectiveThresholds.maxOccupancy, effectiveThresholds.adaptiveActive ? 1 : 0, occupancyThresholdPolicy.c_str());
					lastCapacityDeferMapIdx = currentLocalMapIdx;
					lastCapacityDeferFrame = framesProcessed;
				}
				return true;
			}

			if (shouldLog)
			{
				MULTI_ENGINE_CONSOLE(
					"[MULTI CAPACITY] defer_nonprimary_fusion phase=%s reason=%s frame=%d map=%d blocks=%d capacity=%d occupancy=%.6f reserve=%d stop_threshold=%.6f adaptive=%d policy=%s\n",
					phase, reason, framesProcessed, currentLocalMapIdx, blocks, capacity, occupancy, reserveBlocks,
					effectiveThresholds.maxOccupancy, effectiveThresholds.adaptiveActive ? 1 : 0, occupancyThresholdPolicy.c_str());
				lastCapacityDeferMapIdx = currentLocalMapIdx;
				lastCapacityDeferFrame = framesProcessed;
			}
			return true;
		};

		const ITMActiveMapManager::LocalMapActivity fusionMapType =
			mActiveDataManager->getLocalMapType(todoList[i].dataId);
		if (todoList[i].fusion &&
			skipPrimaryFusionOnCapacityHandoff &&
			allowNewLocalMapsConfigured &&
			fusionMapType == ITMActiveMapManager::PRIMARY_LOCAL_MAP &&
			currentLocalMapIdx == capacityLimitedPrimaryMapIdx &&
			mActiveDataManager->isPrimaryHandoffRequested() &&
			mActiveDataManager->hasActiveNewLocalMap())
		{
			todoList[i].fusion = false;
			const bool shouldLog =
				currentLocalMapIdx != lastPrimaryHandoffFusionSkipMapIdx ||
				lastPrimaryHandoffFusionSkipFrame < 0 ||
				framesProcessed - lastPrimaryHandoffFusionSkipFrame >= 30;
			if (shouldLog)
			{
				MULTI_ENGINE_CONSOLE(
					"[MULTI HANDOFF] skip_primary_fusion frame=%d map=%d reason=pending_handoff_candidate\n",
					framesProcessed, currentLocalMapIdx);
				lastPrimaryHandoffFusionSkipMapIdx = currentLocalMapIdx;
				lastPrimaryHandoffFusionSkipFrame = framesProcessed;
			}
		}

		// fusion in any subscene as long as tracking is good for the respective subscene
		if (todoList[i].fusion && stopFusionForCapacity("pre_fusion")) {
			todoList[i].fusion = false;
			if (retiredCurrentMapForCapacity)
				todoList[i].prepare = false;
		}
		if (todoList[i].fusion) {
			denseMapper->ProcessFrame(view, currentLocalMap->trackingState, currentLocalMap->scene, currentLocalMap->renderState);
			currentLocalMap->markDirty();
			if (stopFusionForCapacity("post_fusion") && retiredCurrentMapForCapacity)
				todoList[i].prepare = false;
		}
		else if (todoList[i].prepare) denseMapper->UpdateVisibleList(view, currentLocalMap->trackingState, currentLocalMap->scene, currentLocalMap->renderState);

		// raycast to renderState_live for tracking and free visualisation
		if (todoList[i].prepare) trackingController->Prepare(currentLocalMap->trackingState, currentLocalMap->scene, view, visualisationEngine, currentLocalMap->renderState);
	}

	const int primaryBeforeMaintenance = mActiveDataManager->findPrimaryLocalMapIdx();
	if (primaryBeforeMaintenance != lastPrimaryMapIdx)
	{
		lastPrimaryMapIdx = primaryBeforeMaintenance;
		primaryMapFrames = 0;
		lastPrimaryOccupancyFrame = -1;
		lastPrimaryOccupancyBlocks = -1;
		capacityLimitedPrimaryMapIdx = -1;
		rawWeakFrameStreak = 0;
		rawFailedFrameStreak = 0;
	}
	primaryMapFrames++;
	if (primaryBeforeMaintenance >= 0)
	{
		const int blocks = mapManager->getLocalMapSize(primaryBeforeMaintenance);
		const int capacity = mapManager->getLocalMap(primaryBeforeMaintenance)->scene->index.getNumAllocatedVoxelBlocks();
		const float occupancy = capacity > 0 ? float(blocks) / float(capacity) : 1.0f;
		const int occupancySampleFrames =
			lastPrimaryOccupancyFrame >= 0 ? MAX(1, framesProcessed - lastPrimaryOccupancyFrame) : 0;
		const int occupancySampleBlockGrowth =
			lastPrimaryOccupancyBlocks >= 0 ? MAX(0, blocks - lastPrimaryOccupancyBlocks) : 0;
		const float occupancyGrowthBlocksPerFrame =
			occupancySampleFrames > 0 ? float(occupancySampleBlockGrowth) / float(occupancySampleFrames) : 0.0f;
		const EffectiveOccupancyThresholds effectiveThresholds = getEffectiveOccupancyThresholds(occupancySampleFrames);
		const float stopOccupancyLimit = effectiveThresholds.maxOccupancy > 0.0f ? effectiveThresholds.maxOccupancy : 1.0f;
		const float projectedOccupancy =
			capacity > 0 ? (float(blocks) + occupancyGrowthBlocksPerFrame * float(occupancyGrowthLookaheadFrames)) / float(capacity) : 1.0f;
		const bool reachedFrameLimit = maxFramesPerPrimary > 0 && primaryMapFrames >= maxFramesPerPrimary;
		const bool reachedNewMapOccupancy =
			effectiveThresholds.newMapOccupancy > 0.0f && capacity > 0 && occupancy >= effectiveThresholds.newMapOccupancy;
		const bool projectedStopMapOccupancy =
			occupancyGrowthLookaheadFrames > 0 &&
			primaryMapFrames >= occupancyGrowthMinPrimaryFrames &&
			occupancySampleFrames >= occupancyGrowthSampleFrames &&
			occupancyGrowthBlocksPerFrame > 0.0f &&
			capacity > 0 &&
			occupancy < effectiveThresholds.newMapOccupancy &&
			projectedOccupancy >= stopOccupancyLimit;
		const bool reachedStopMapOccupancy =
			effectiveThresholds.maxOccupancy > 0.0f && capacity > 0 && occupancy >= effectiveThresholds.maxOccupancy;
		if (reachedStopMapOccupancy)
		{
			capacityLimitedPrimaryMapIdx = primaryBeforeMaintenance;
			mActiveDataManager->requestPrimaryHandoff();
		}
		if (reachedFrameLimit || reachedNewMapOccupancy || projectedStopMapOccupancy)
		{
			if (reachedNewMapOccupancy || projectedStopMapOccupancy)
				capacityLimitedPrimaryMapIdx = primaryBeforeMaintenance;
			mActiveDataManager->requestNewArea();
			if (reachedNewMapOccupancy || projectedStopMapOccupancy)
			{
				const char *reason = reachedNewMapOccupancy ? "occupancy" : "projected_growth";
				const bool shouldLog =
					primaryBeforeMaintenance != lastOccupancyRequestMapIdx ||
					lastOccupancyRequestFrame < 0 ||
					framesProcessed - lastOccupancyRequestFrame >= 30;
				if (shouldLog)
				{
					MULTI_ENGINE_CONSOLE(
						"[MULTI CAPACITY] request_new_area frame=%d map=%d reason=%s blocks=%d capacity=%d occupancy=%.6f start_threshold=%.6f stop_threshold=%.6f base_start=%.6f base_stop=%.6f adaptive=%d policy=%s projected_occupancy=%.6f growth_blocks_per_frame=%.2f lookahead_frames=%d primary_age=%d sample_frames=%d min_primary_age=%d sample_window=%d raw_weak_streak=%d raw_failed_streak=%d\n",
						framesProcessed, primaryBeforeMaintenance, reason, blocks, capacity, occupancy, effectiveThresholds.newMapOccupancy, effectiveThresholds.maxOccupancy,
						newMapBlockOccupancy, maxBlockOccupancy, effectiveThresholds.adaptiveActive ? 1 : 0, occupancyThresholdPolicy.c_str(),
						projectedOccupancy, occupancyGrowthBlocksPerFrame, occupancyGrowthLookaheadFrames,
						primaryMapFrames, occupancySampleFrames, occupancyGrowthMinPrimaryFrames, occupancyGrowthSampleFrames,
						rawWeakFrameStreak, rawFailedFrameStreak);
					lastOccupancyRequestMapIdx = primaryBeforeMaintenance;
					lastOccupancyRequestFrame = framesProcessed;
				}
			}
		}
		if (lastPrimaryOccupancyFrame < 0 || occupancySampleFrames >= occupancyGrowthSampleFrames)
		{
			lastPrimaryOccupancyFrame = framesProcessed;
			lastPrimaryOccupancyBlocks = blocks;
		}
	}
		mScheduleGlobalAdjustment |= mActiveDataManager->maintainActiveData(framesProcessed);
	enforceResidency();

	if (globalAdjustmentEnabled && mScheduleGlobalAdjustment)
	{
		if (mGlobalAdjustmentEngine->updateMeasurements(*mapManager)) 
		{
			mGlobalAdjustmentEngine->runGlobalAdjustment();

			mScheduleGlobalAdjustment = false;
		}
	}
	if (globalAdjustmentEnabled) mGlobalAdjustmentEngine->retrieveNewEstimates(*mapManager);
	framesProcessed++;

	return primaryLocalMapTrackingResult;
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::SaveSceneToMesh(const char *modelFileName)
{
	if (meshingEngine == NULL) return;
	if (pagingEnabled || mapManager->numLocalMaps() > MAX_NUM_LOCALMAPS)
	{
		const std::filesystem::path target(modelFileName);
		const std::filesystem::path staging = target.string() + ".tmp";
		const std::filesystem::path backup = target.string() + ".bak";
		const std::filesystem::path vertexBody = target.string() + ".vertices.tmp";
		const std::filesystem::path faceBody = target.string() + ".faces.tmp";
		std::ofstream vertices(vertexBody, std::ios::trunc);
		std::ofstream faces(faceBody, std::ios::trunc);
		if (!vertices || !faces) throw std::runtime_error("Could not create staged paged mesh output");

		ITMMeshingEngine<TVoxel, TIndex> *singleMesher =
			ITMMeshingEngineFactory::MakeMeshingEngine<TVoxel, TIndex>(settings->deviceType);
		if (singleMesher == NULL) throw std::runtime_error("Could not create single-map meshing engine");

		uint64_t totalTriangles = 0;
		try
		{
			for (int localMapId = 0; localMapId < (int)mapManager->numLocalMaps(); ++localMapId)
			{
				ensureLocalMapResident(localMapId);
				ITMLocalMap<TVoxel, TIndex> *localMap = mapManager->getLocalMap(localMapId);
				const int blocks = MAX(1, mapManager->getLocalMapSize(localMapId));
				const uint64_t maximum = std::min<uint64_t>(uint64_t(blocks) * 512, std::numeric_limits<uint>::max());
				uint64_t requested = std::min<uint64_t>(maximum, std::max<uint64_t>(1024, uint64_t(blocks) * 32));
				std::unique_ptr<ITMMesh> localMesh;
				while (true)
				{
					localMesh.reset(new ITMMesh(settings->GetMemoryType(), (uint)requested));
					singleMesher->MeshScene(localMesh.get(), localMap->scene);
					if (localMesh->noTotalTriangles < localMesh->noMaxTriangles - 1) break;
					if (requested >= maximum)
						throw std::runtime_error("Paged mesh triangle capacity exhausted for submap " + std::to_string(localMapId));
					requested = std::min<uint64_t>(maximum, requested * 2);
					MULTI_ENGINE_CONSOLE("[MESH] retry submap=%d triangle_capacity=%llu\n",
						localMapId, (unsigned long long)requested);
				}

				ORUtils::MemoryBlock<ITMMesh::Triangle> cpuTriangles(localMesh->noMaxTriangles, MEMORYDEVICE_CPU);
				if (settings->GetMemoryType() == MEMORYDEVICE_CUDA)
					cpuTriangles.SetFrom(localMesh->triangles, ORUtils::MemoryBlock<ITMMesh::Triangle>::CUDA_TO_CPU);
				else
					cpuTriangles.SetFrom(localMesh->triangles, ORUtils::MemoryBlock<ITMMesh::Triangle>::CPU_TO_CPU);
				const ITMMesh::Triangle *triangleArray = cpuTriangles.GetData(MEMORYDEVICE_CPU);
				const Matrix4f localToGlobal = mapManager->getEstimatedGlobalPose(localMapId).GetInvM();
				for (uint triangleId = 0; triangleId < localMesh->noTotalTriangles; ++triangleId)
				{
					const ITMMesh::Triangle &triangle = triangleArray[triangleId];
					const Vector3f p0 = localToGlobal * triangle.p0;
					const Vector3f p1 = localToGlobal * triangle.p1;
					const Vector3f p2 = localToGlobal * triangle.p2;
					vertices << p0.x << " " << p0.y << " " << p0.z << " 0 0 0\n"
							 << p1.x << " " << p1.y << " " << p1.z << " 0 0 0\n"
							 << p2.x << " " << p2.y << " " << p2.z << " 0 0 0\n";
					const uint64_t base = (totalTriangles + triangleId) * 3;
					faces << "3 " << base << " " << base + 1 << " " << base + 2 << "\n";
				}
				totalTriangles += localMesh->noTotalTriangles;
				MULTI_ENGINE_CONSOLE("[MESH] submap=%d blocks=%d triangles=%u total=%llu\n",
					localMapId, blocks, localMesh->noTotalTriangles, (unsigned long long)totalTriangles);
				enforceResidency(localMapId);
			}
			delete singleMesher;
			singleMesher = NULL;
			vertices.close();
			faces.close();
			if (!vertices || !faces) throw std::runtime_error("Failed while writing paged mesh bodies");

			std::ofstream output(staging, std::ios::trunc);
			if (!output) throw std::runtime_error("Could not create staged global mesh");
			output << "ply\nformat ascii 1.0\n"
				   << "element vertex " << totalTriangles * 3 << "\n"
				   << "property float x\nproperty float y\nproperty float z\n"
				   << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
				   << "element face " << totalTriangles << "\n"
				   << "property list uchar int vertex_indices\nend_header\n";
			std::ifstream vertexInput(vertexBody), faceInput(faceBody);
			output << vertexInput.rdbuf() << faceInput.rdbuf();
			output.close();
			if (!output) throw std::runtime_error("Failed while assembling paged global mesh");
			std::filesystem::remove(backup);
			if (std::filesystem::exists(target)) std::filesystem::rename(target, backup);
			try { std::filesystem::rename(staging, target); }
			catch (...)
			{
				if (!std::filesystem::exists(target) && std::filesystem::exists(backup))
					std::filesystem::rename(backup, target);
				throw;
			}
			std::filesystem::remove(backup);
			std::filesystem::remove(vertexBody);
			std::filesystem::remove(faceBody);
			enforceResidency();
			return;
		}
		catch (...)
		{
			delete singleMesher;
			std::filesystem::remove(staging);
			if (!std::filesystem::exists(target) && std::filesystem::exists(backup))
				std::filesystem::rename(backup, target);
			std::filesystem::remove(vertexBody);
			std::filesystem::remove(faceBody);
			throw;
		}
	}

	ITMMesh *mesh = new ITMMesh(settings->GetMemoryType());

	meshingEngine->MeshScene(mesh, *mapManager);
	mesh->WritePLY(modelFileName);
	
	delete mesh;
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::SaveToFile()
{
	SaveToFile("State/");
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::LoadFromFile()
{
	LoadFromFile("State/");
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::SaveToFile(const std::string &directory)
{
	MakeDir(directory.c_str());
	std::ofstream manifest(directory + "/multi_engine.bin", std::ios::binary | std::ios::trunc);
	if (!manifest) throw std::runtime_error("Could not write MultiEngine manifest: " + directory);
	const int mapCount = (int)mapManager->numLocalMaps();
	const int primaryMap = findPrimaryLocalMapIdx();
	manifest.write(reinterpret_cast<const char *>(&mapCount), sizeof(mapCount));
	manifest.write(reinterpret_cast<const char *>(&primaryMap), sizeof(primaryMap));
	for (int i = 0; i < mapCount; ++i)
	{
		const Matrix4f transform = mapManager->getEstimatedGlobalPose(i).GetM();
		manifest.write(reinterpret_cast<const char *>(transform.m), sizeof(float) * 16);
		const std::string mapDirectory = directory + "/submap_" + std::to_string(i) + "/";
		MakeDir(mapDirectory.c_str());
		if (mapManager->isLocalMapResident(i))
		{
			if (std::getenv("GPS_SLAM_DISABLE_SPARSE_TSDF_PAGES") != NULL)
				mapManager->getLocalMap(i)->scene->SaveToDirectory(mapDirectory);
			else if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			{
				ITMScene<TVoxel, TIndex> cpuScene(&settings->sceneParams, false, MEMORYDEVICE_CPU);
				cpuScene.CopyFromCUDA(*mapManager->getLocalMap(i)->scene);
				cpuScene.SaveSparseToDirectory(mapDirectory);
			}
			else
				mapManager->getLocalMap(i)->scene->SaveSparseToDirectory(mapDirectory);
		}
		else
		{
			mapManager->waitForLocalMapSnapshot(i);
			if (mapManager->getLocalMap(i)->hasCpuPageCache())
			{
				mapManager->getLocalMap(i)->saveCpuPageCacheToDirectory(mapDirectory);
			}
			else
			{
				const std::string &pageDirectory = mapManager->getLocalMapPageDirectory(i);
				if (pageDirectory.empty() || !std::filesystem::exists(pageDirectory))
					throw std::runtime_error("Paged MultiEngine map has no snapshot: " + std::to_string(i));
				std::filesystem::copy(pageDirectory, mapDirectory,
					std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
			}
		}
	}
	manifest.close();
	if (!manifest) throw std::runtime_error("Failed while writing MultiEngine manifest: " + directory);
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::LoadFromFile(const std::string &directory)
{
	std::ifstream manifest(directory + "/multi_engine.bin", std::ios::binary);
	if (!manifest) throw std::runtime_error("Could not read MultiEngine manifest: " + directory);
	int mapCount = 0, primaryMap = 0;
	manifest.read(reinterpret_cast<char *>(&mapCount), sizeof(mapCount));
	manifest.read(reinterpret_cast<char *>(&primaryMap), sizeof(primaryMap));
	if (!manifest || mapCount <= 0 || mapCount > 100000 || primaryMap < 0 || primaryMap >= mapCount)
		throw std::runtime_error("Invalid MultiEngine manifest: " + directory);
	std::vector<Matrix4f> transforms(mapCount);
	std::vector<std::string> mapDirectories(mapCount);
	for (int i = 0; i < mapCount; ++i)
	{
		manifest.read(reinterpret_cast<char *>(transforms[i].m), sizeof(float) * 16);
		if (!manifest) throw std::runtime_error("Truncated MultiEngine manifest: " + directory);
		for (int j = 0; j < 16; ++j)
			if (!std::isfinite(transforms[i].m[j]))
				throw std::runtime_error("Non-finite transform in MultiEngine manifest: " + directory);
		std::string mapDirectory = directory + "/submap_" + std::to_string(i) + "/";
		std::ifstream sceneProbe(mapDirectory + "hash.dat", std::ios::binary);
		if (!sceneProbe.good())
		{
			// Compatibility with checkpoints produced before submap directories were fixed.
			mapDirectory = directory + "/submap_" + std::to_string(i);
			std::ifstream legacySceneProbe(mapDirectory + "hash.dat", std::ios::binary);
			if (!legacySceneProbe.good())
				throw std::runtime_error("Missing scene data for MultiEngine submap " + std::to_string(i));
		}
		mapDirectories[i] = mapDirectory;
	}
	std::vector<int> keepResidentAfterLoad;
	if (pagingEnabled)
	{
		keepResidentAfterLoad.push_back(primaryMap);
		if (mapCount <= maxResidentMaps)
		{
			for (int i = 0; i < mapCount; ++i)
				if (std::find(keepResidentAfterLoad.begin(), keepResidentAfterLoad.end(), i) == keepResidentAfterLoad.end())
					keepResidentAfterLoad.push_back(i);
		}
		else
		{
			for (int i = mapCount - 1; i >= 0 && (int)keepResidentAfterLoad.size() < maxResidentMaps; --i)
				if (std::find(keepResidentAfterLoad.begin(), keepResidentAfterLoad.end(), i) == keepResidentAfterLoad.end())
					keepResidentAfterLoad.push_back(i);
		}
	}
	for (int i = 0; i < mapCount; ++i)
	{
		while ((int)mapManager->numLocalMaps() <= i) mapManager->createNewLocalMap();
		if (!mapManager->isLocalMapResident(i)) mapManager->pageInLocalMap(i);
		mapManager->setEstimatedGlobalPose(i, ORUtils::SE3Pose(transforms[i]));
		mapManager->getLocalMap(i)->scene->LoadFromDirectory(mapDirectories[i]);
		if (pagingEnabled &&
			std::find(keepResidentAfterLoad.begin(), keepResidentAfterLoad.end(), i) == keepResidentAfterLoad.end())
			mapManager->pageOutLocalMap(i, pagingDirectory + "/submap_" + std::to_string(i), false, cpuPagingStorage);
	}
	if ((int)mapManager->numLocalMaps() != mapCount) throw std::runtime_error("MultiEngine load requires a fresh engine");
	mActiveDataManager->resetToPrimaryMap(primaryMap);
	enforceResidency();
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::runRaycast(ORUtils::SE3Pose *globalPose, ITMIntrinsics *intrinsics, int localMapIdx)
{
	freeviewLocalMapIdx = localMapIdx;
	if (localMapIdx >= 0)
	{
		ensureLocalMapResident(localMapIdx);
		ITMLocalMap<TVoxel, TIndex> *localMap = mapManager->getLocalMap(localMapIdx);
		if (renderState_freeview == NULL) renderState_freeview = visualisationEngine->CreateRenderState(localMap->scene, trackedImageSize);
		ORUtils::SE3Pose localPose;
		localPose.SetInvM(localMap->estimatedGlobalPose.GetM() * globalPose->GetInvM());
		visualisationEngine->FindVisibleBlocks(localMap->scene, &localPose, intrinsics, renderState_freeview);
		visualisationEngine->CreateExpectedDepths(localMap->scene, &localPose, intrinsics, renderState_freeview);
		visualisationEngine->RenderImage(localMap->scene, &localPose, intrinsics, renderState_freeview, renderState_freeview->raycastImage, IITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME);
		return;
	}
	if (pagingEnabled)
		throw std::runtime_error("Combined MultiEngine raycast with paged maps requires explicit batched rendering");
	if (mapManager->numLocalMaps() > MAX_NUM_LOCALMAPS)
		throw std::runtime_error("Combined MultiEngine raycast exceeds MAX_NUM_LOCALMAPS; paging/batched rendering is required");
	if (renderState_multiscene == NULL) renderState_multiscene = multiVisualisationEngine->CreateRenderState(mapManager->getLocalMap(0)->scene, trackedImageSize);
	multiVisualisationEngine->PrepareRenderState(*mapManager, renderState_multiscene);
	multiVisualisationEngine->CreateExpectedDepths(globalPose, intrinsics, renderState_multiscene);
	multiVisualisationEngine->RenderImage(globalPose, intrinsics, renderState_multiscene, renderState_multiscene->raycastImage, IITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME);
}

template <typename TVoxel, typename TIndex>
ORUtils::Image<Vector4u> *ITMMultiEngine<TVoxel, TIndex>::GetFreeImage(void)
{
	return freeviewLocalMapIdx >= 0 ? renderState_freeview->raycastImage : renderState_multiscene->raycastImage;
}

template <typename TVoxel, typename TIndex>
ORUtils::Image<Vector4f> *ITMMultiEngine<TVoxel, TIndex>::GetFreeVertex(void)
{
	return freeviewLocalMapIdx >= 0 ? renderState_freeview->raycastResult : renderState_multiscene->raycastResult;
}

template <typename TVoxel, typename TIndex>
Vector2i ITMMultiEngine<TVoxel, TIndex>::GetImageSize(void) const
{
	return trackedImageSize;
}

template <typename TVoxel, typename TIndex>
void ITMMultiEngine<TVoxel, TIndex>::GetImage(ITMUChar4Image *out, GetImageType getImageType, ORUtils::SE3Pose *pose, ITMIntrinsics *intrinsics)
{
	if (view == NULL) return;

	out->Clear();

	switch (getImageType)
	{
	case ITMMultiEngine::InfiniTAM_IMAGE_ORIGINAL_RGB:
		out->ChangeDims(view->rgb->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			out->SetFrom(view->rgb, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		else out->SetFrom(view->rgb, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		break;
	case ITMMultiEngine::InfiniTAM_IMAGE_ORIGINAL_DEPTH:
		out->ChangeDims(view->depth->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA) view->depth->UpdateHostFromDevice();
		ITMVisualisationEngine<TVoxel, TIndex>::DepthToUchar4(out, view->depth);
		break;
    case ITMMultiEngine::InfiniTAM_IMAGE_COLOUR_FROM_VOLUME: //TODO: add colour rendering
	case ITMMultiEngine::InfiniTAM_IMAGE_SCENERAYCAST:
	case ITMMultiEngine::InfiniTAM_IMAGE_COLOUR_FROM_NORMAL:
	case ITMMultiEngine::InfiniTAM_IMAGE_COLOUR_FROM_CONFIDENCE:
	{
		int visualisationLocalMapIdx = mActiveDataManager->findBestVisualisationLocalMapIdx();
		if (visualisationLocalMapIdx < 0) break; // TODO: clear image? what else to do when tracking is lost?

		ITMLocalMap<TVoxel, TIndex> *activeLocalMap = mapManager->getLocalMap(visualisationLocalMapIdx);

		IITMVisualisationEngine::RenderRaycastSelection raycastType;
		if (activeLocalMap->trackingState->age_pointCloud <= 0) raycastType = IITMVisualisationEngine::RENDER_FROM_OLD_RAYCAST;
		else raycastType = IITMVisualisationEngine::RENDER_FROM_OLD_FORWARDPROJ;

		IITMVisualisationEngine::RenderImageType imageType;
		switch (getImageType) 
		{
		case ITMMultiEngine::InfiniTAM_IMAGE_COLOUR_FROM_CONFIDENCE:
			imageType = IITMVisualisationEngine::RENDER_COLOUR_FROM_CONFIDENCE;
			break;
		case ITMMultiEngine::InfiniTAM_IMAGE_COLOUR_FROM_NORMAL:
			imageType = IITMVisualisationEngine::RENDER_COLOUR_FROM_NORMAL;
			break;
		default:
			imageType = IITMVisualisationEngine::RENDER_SHADED_GREYSCALE_IMAGENORMALS;
		}

		visualisationEngine->RenderImage(activeLocalMap->scene, activeLocalMap->trackingState->pose_d, &view->calib.intrinsics_d, activeLocalMap->renderState, activeLocalMap->renderState->raycastImage, imageType, raycastType);

		ORUtils::Image<Vector4u> *srcImage = activeLocalMap->renderState->raycastImage;
		out->ChangeDims(srcImage->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			out->SetFrom(srcImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		else out->SetFrom(srcImage, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		break;
	}
	case ITMMultiEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED:
	case ITMMultiEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME:
	case ITMMultiEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL:
	case ITMMultiEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_CONFIDENCE:
	{
		IITMVisualisationEngine::RenderImageType type = IITMVisualisationEngine::RENDER_SHADED_GREYSCALE;
		if (getImageType == ITMMultiEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME) type = IITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME;
		else if (getImageType == ITMMultiEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL) type = IITMVisualisationEngine::RENDER_COLOUR_FROM_NORMAL;
		else if (getImageType == ITMMultiEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_CONFIDENCE) type = IITMVisualisationEngine::RENDER_COLOUR_FROM_CONFIDENCE;

		if (freeviewLocalMapIdx >= 0) 
		{
			ITMLocalMap<TVoxel, TIndex> *activeData = mapManager->getLocalMap(freeviewLocalMapIdx);
			if (renderState_freeview == NULL) renderState_freeview = visualisationEngine->CreateRenderState(activeData->scene, out->noDims);

			visualisationEngine->FindVisibleBlocks(activeData->scene, pose, intrinsics, renderState_freeview);
			visualisationEngine->CreateExpectedDepths(activeData->scene, pose, intrinsics, renderState_freeview);
			visualisationEngine->RenderImage(activeData->scene, pose, intrinsics, renderState_freeview, renderState_freeview->raycastImage, type);

			if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
				out->SetFrom(renderState_freeview->raycastImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
			else out->SetFrom(renderState_freeview->raycastImage, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		}
		else 
		{
			if (renderState_multiscene == NULL) renderState_multiscene = multiVisualisationEngine->CreateRenderState(mapManager->getLocalMap(0)->scene, out->noDims);
			multiVisualisationEngine->PrepareRenderState(*mapManager, renderState_multiscene);
			multiVisualisationEngine->CreateExpectedDepths(pose, intrinsics, renderState_multiscene);
			multiVisualisationEngine->RenderImage(pose, intrinsics, renderState_multiscene, renderState_multiscene->raycastImage, type);
			if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
				out->SetFrom(renderState_multiscene->raycastImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
			else out->SetFrom(renderState_multiscene->raycastImage, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		}

		break;
	}
	case ITMMultiEngine::InfiniTAM_IMAGE_UNKNOWN:
		break;
	};
}

// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMActiveMapManager.h"
#include <algorithm>
#include <cmath>

using namespace ITMLib;

// try loop closures for this number of frames
static const int N_linktrials = 20;
// at least these many frames have to be tracked successfully
static const int N_linkoverlap = 10;
// try relocalisations for this number of frames
static const int N_reloctrials = 20;
// at least these many tracking attempts have to succeed for relocalisation
static const int N_relocsuccess = 10;
// When checking "overlap with original local map", find how many of the first N
// blocks are still visible
static const int N_originalblocks = 1000;
static const float F_originalBlocksThreshold = 0.2f; //0.4f

#define ACTIVE_MAP_CONSOLE(...) do { if (consoleDiagnostics) fprintf(stdout, __VA_ARGS__); } while (0)

ITMActiveMapManager::ITMActiveMapManager(ITMMapGraphManager *_localMapManager)
{
	localMapManager = _localMapManager;
	linkTrials = N_linktrials;
	linkOverlap = N_linkoverlap;
	emergencyLinkOverlap = 0;
	originalBlocks = N_originalblocks;
	originalBlocksThreshold = F_originalBlocksThreshold;
	allowNewLocalMaps = true;
	allowHistoricalLinks = true;
	allowVisibilityHandoff = true;
	consoleDiagnostics = false;
	forceNewArea = false;
	forcePrimaryHandoff = false;
	diagnosticFrame = -1;
	sequentialTransformPolicy = "validated_overlap";
	sequentialTransformMaxTranslation = 0.03f;
	sequentialTransformMaxRotationDeg = 3.0f;
	invalidTransformRestartAttempts = 0;
	handoffPosePolicy = "candidate";
	newMapHandoffPolicy = "visibility";
	newMapHandoffMinVisibleRatio = 0.0f;
	newMapHandoffMinVisibleMargin = 0.0f;
	newMapHandoffMinAcceptedWindows = 1;
	newMapHandoffEmergencyMinAcceptedWindows = -1;
	candidateRecoveryPolicy = "disabled";
	candidateRecoveryMinVisibleRatio = 0.3f;
	candidateRecoveryMinBlocks = 1000;
	candidateRecoveryMaxAttempts = 2;
	primaryFailureHandoffPolicy = "disabled";
	primaryFailureHandoffRawGate = "failed_only";
	primaryFailureHandoffMinRawFailedFrames = 30;
	emergencyLinkMinRawWeakFrames = 0;
	primaryFailureHandoffMinCandidateVisibleRatio = 0.5f;
	primaryFailureHandoffMinCandidateBlocks = 1000;
	primaryFailureHandoffMinCandidateGoodFrames = 10;
	primaryFailureHandoffMinOverlapFrames = 0;
	primaryFailureHandoffRequirePoseContinuity = false;
}

void ITMActiveMapManager::configureSequentialPolicy(bool allowNewMaps, bool allowOldLinks, float visibleRatioThreshold, int overlapFrames, int trialFrames, int emergencyOverlapFrames, int sampledOriginalBlocks,
	const std::string &transformPolicy, float maxTransformTranslation, float maxTransformRotationDeg, int invalidTransformRestartAttemptsConfigured, const std::string &handoffPolicy, const std::string &newMapHandoffPolicyConfigured,
	float newMapHandoffMinVisibleRatioConfigured, float newMapHandoffMinVisibleMarginConfigured,
	int newMapHandoffMinAcceptedWindowsConfigured, int newMapHandoffEmergencyMinAcceptedWindowsConfigured,
	const std::string &candidateRecoveryPolicyConfigured, float candidateRecoveryMinVisibleRatioConfigured, int candidateRecoveryMinBlocksConfigured, int candidateRecoveryMaxAttemptsConfigured,
	const std::string &primaryFailureHandoffPolicyConfigured, int primaryFailureHandoffMinRawFailedFramesConfigured,
	const std::string &primaryFailureHandoffRawGateConfigured, int primaryFailureHandoffMinRawWeakFramesConfigured,
	float primaryFailureHandoffMinCandidateVisibleRatioConfigured, int primaryFailureHandoffMinCandidateBlocksConfigured,
	int primaryFailureHandoffMinCandidateGoodFramesConfigured, int primaryFailureHandoffMinOverlapFramesConfigured,
	bool primaryFailureHandoffRequirePoseContinuityConfigured,
	bool allowVisibilityHandoffConfigured)
{
	allowNewLocalMaps = allowNewMaps;
	allowHistoricalLinks = allowOldLinks;
	allowVisibilityHandoff = allowVisibilityHandoffConfigured;
	originalBlocksThreshold = MAX(0.0f, MIN(1.0f, visibleRatioThreshold));
	linkOverlap = MAX(1, overlapFrames);
	linkTrials = MAX(linkOverlap, trialFrames);
	emergencyLinkOverlap = MAX(0, MIN(linkOverlap, emergencyOverlapFrames));
	originalBlocks = MAX(1, sampledOriginalBlocks);
	sequentialTransformPolicy = transformPolicy;
	sequentialTransformMaxTranslation = MAX(0.0f, maxTransformTranslation);
	sequentialTransformMaxRotationDeg = MAX(0.0f, maxTransformRotationDeg);
	invalidTransformRestartAttempts = MAX(0, invalidTransformRestartAttemptsConfigured);
	handoffPosePolicy = handoffPolicy;
	newMapHandoffPolicy = newMapHandoffPolicyConfigured;
	newMapHandoffMinVisibleRatio = MAX(0.0f, MIN(1.0f, newMapHandoffMinVisibleRatioConfigured));
	newMapHandoffMinVisibleMargin = MAX(0.0f, MIN(1.0f, newMapHandoffMinVisibleMarginConfigured));
	newMapHandoffMinAcceptedWindows = MAX(0, newMapHandoffMinAcceptedWindowsConfigured);
	newMapHandoffEmergencyMinAcceptedWindows =
		newMapHandoffEmergencyMinAcceptedWindowsConfigured < 0 ? -1 : MAX(0, newMapHandoffEmergencyMinAcceptedWindowsConfigured);
	candidateRecoveryPolicy = candidateRecoveryPolicyConfigured;
	candidateRecoveryMinVisibleRatio = MAX(0.0f, MIN(1.0f, candidateRecoveryMinVisibleRatioConfigured));
	candidateRecoveryMinBlocks = MAX(0, candidateRecoveryMinBlocksConfigured);
	candidateRecoveryMaxAttempts = MAX(0, candidateRecoveryMaxAttemptsConfigured);
	primaryFailureHandoffPolicy = primaryFailureHandoffPolicyConfigured;
	primaryFailureHandoffRawGate =
		primaryFailureHandoffRawGateConfigured == "failed_or_weak" ? primaryFailureHandoffRawGateConfigured : "failed_only";
	primaryFailureHandoffMinRawFailedFrames = MAX(1, primaryFailureHandoffMinRawFailedFramesConfigured);
	emergencyLinkMinRawWeakFrames = MAX(0, primaryFailureHandoffMinRawWeakFramesConfigured);
	primaryFailureHandoffMinCandidateVisibleRatio = MAX(0.0f, MIN(1.0f, primaryFailureHandoffMinCandidateVisibleRatioConfigured));
	primaryFailureHandoffMinCandidateBlocks = MAX(0, primaryFailureHandoffMinCandidateBlocksConfigured);
	primaryFailureHandoffMinCandidateGoodFrames = MAX(1, primaryFailureHandoffMinCandidateGoodFramesConfigured);
	primaryFailureHandoffMinOverlapFrames = MAX(0, primaryFailureHandoffMinOverlapFramesConfigured);
	primaryFailureHandoffRequirePoseContinuity = primaryFailureHandoffRequirePoseContinuityConfigured;
}

void ITMActiveMapManager::configureDiagnostics(const std::string &directory)
{
	if (directory.empty()) return;
	const std::string transitionPath = directory + "/submap_transitions.csv";
	bool writeHeader = true;
	{
		std::ifstream existingLog(transitionPath);
		writeHeader = !existingLog.good() || existingLog.peek() == std::ifstream::traits_type::eof();
	}
	if (transitionLog.is_open()) transitionLog.close();
	transitionLog.open(transitionPath, std::ios::out | std::ios::app);
	if (transitionLog)
		{
			if (writeHeader)
				transitionLog << "frame,event,primary_local_map,new_local_map,policy,action,inliers,delta_m,delta_deg,"
					"max_m,max_deg,primary_visible,new_visible,new_blocks,force_handoff,move_to_primary,"
					"candidate_good,candidate_poor,candidate_failed,candidate_recovery_attempts,"
					"primary_raw_weak,primary_raw_failed,detail\n";
			transitionLog.flush();
		}
	}

void ITMActiveMapManager::logTransitionEvent(const std::string &event, int primaryLocalMap, int newLocalMap,
	const std::string &policy, const std::string &action, int inliers,
	float deltaM, float deltaDeg, float primaryVisible, float newVisible,
	int newBlocks, bool forceHandoff, bool moveToPrimary,
	int candidateGoodFrames, int candidatePoorFrames, int candidateFailedFrames, int candidateRecoveryAttempts,
	int primaryRawWeakFrames, int primaryRawFailedFrames, const std::string &detail) const
{
	TransitionEvent transitionEvent;
	transitionEvent.frame = diagnosticFrame;
	transitionEvent.event = event;
	transitionEvent.primaryLocalMap = primaryLocalMap;
	transitionEvent.newLocalMap = newLocalMap;
	transitionEvent.policy = policy;
	transitionEvent.action = action;
	transitionEvent.inliers = inliers;
	transitionEvent.deltaM = deltaM;
	transitionEvent.deltaDeg = deltaDeg;
	transitionEvent.primaryVisible = primaryVisible;
	transitionEvent.newVisible = newVisible;
	transitionEvent.newBlocks = newBlocks;
	transitionEvent.forceHandoff = forceHandoff;
	transitionEvent.moveToPrimary = moveToPrimary;
	transitionEvent.candidateGoodFrames = candidateGoodFrames;
	transitionEvent.candidatePoorFrames = candidatePoorFrames;
	transitionEvent.candidateFailedFrames = candidateFailedFrames;
	transitionEvent.candidateRecoveryAttempts = candidateRecoveryAttempts;
	transitionEvent.primaryRawWeakFrames = primaryRawWeakFrames;
	transitionEvent.primaryRawFailedFrames = primaryRawFailedFrames;
	transitionEvent.detail = detail;
	recentTransitionEvents.push_back(transitionEvent);
	if (recentTransitionEvents.size() > 256)
		recentTransitionEvents.erase(recentTransitionEvents.begin(),
			recentTransitionEvents.begin() + (recentTransitionEvents.size() - 256));

	if (!transitionLog) return;
	transitionLog
		<< diagnosticFrame << ","
		<< event << ","
		<< primaryLocalMap << ","
		<< newLocalMap << ","
		<< policy << ","
		<< action << ","
		<< inliers << ","
		<< deltaM << ","
		<< deltaDeg << ","
		<< sequentialTransformMaxTranslation << ","
		<< sequentialTransformMaxRotationDeg << ","
		<< primaryVisible << ","
		<< newVisible << ","
		<< newBlocks << ","
		<< (forceHandoff ? 1 : 0) << ","
		<< (moveToPrimary ? 1 : 0) << ","
		<< candidateGoodFrames << ","
		<< candidatePoorFrames << ","
		<< candidateFailedFrames << ","
		<< candidateRecoveryAttempts << ","
		<< primaryRawWeakFrames << ","
		<< primaryRawFailedFrames << ","
		<< detail << "\n";
	transitionLog.flush();
}

std::vector<ITMActiveMapManager::TransitionEvent> ITMActiveMapManager::consumeRecentTransitionEvents(void) const
{
	std::vector<TransitionEvent> events = recentTransitionEvents;
	recentTransitionEvents.clear();
	return events;
}

void ITMActiveMapManager::initializeActiveDescriptor(ActiveDataDescriptor &descriptor, int localMapIndex, LocalMapActivity type) const
{
	descriptor.localMapIndex = localMapIndex;
	descriptor.type = type;
	descriptor.constraints.clear();
	descriptor.estimatedPose = ORUtils::SE3Pose();
	descriptor.trackingAttempts = 0;
	descriptor.lastSequentialTransformAccepted = false;
	descriptor.invalidSequentialTransformAttempts = 0;
	descriptor.candidateRecoveryAttempts = 0;
	descriptor.acceptedSequentialWindows = 0;
	descriptor.lastSequentialTransformEmergency = false;
	descriptor.goodTrackingFrames = 0;
	descriptor.poorTrackingFrames = 0;
	descriptor.failedTrackingFrames = 0;
	descriptor.lastTrackingResult = ITMTrackingState::TRACKING_FAILED;
	descriptor.rawWeakTrackingFrames = 0;
	descriptor.rawFailedTrackingFrames = 0;
	descriptor.lastRawTrackingResult = ITMTrackingState::TRACKING_FAILED;
}

void ITMActiveMapManager::resetToPrimaryMap(int localMapIndex)
{
	activeData.clear();
	ActiveDataDescriptor primary;
	initializeActiveDescriptor(primary, localMapIndex, PRIMARY_LOCAL_MAP);
	activeData.push_back(primary);
	forcePrimaryHandoff = false;
}

bool ITMActiveMapManager::retireNewLocalMap(int localMapIndex, const std::string &reason)
{
	for (size_t i = 0; i < activeData.size(); ++i)
	{
		ActiveDataDescriptor &data = activeData[i];
		if (data.localMapIndex != localMapIndex || data.type != NEW_LOCAL_MAP) continue;

		data.type = LOST_NEW;
		data.constraints.clear();
		data.trackingAttempts = 0;
			data.lastSequentialTransformAccepted = false;
			data.invalidSequentialTransformAttempts = 0;
			data.candidateRecoveryAttempts = 0;
			data.acceptedSequentialWindows = 0;
			data.lastSequentialTransformEmergency = false;
		data.goodTrackingFrames = 0;
		data.poorTrackingFrames = 0;
		data.failedTrackingFrames = 0;
		data.lastTrackingResult = ITMTrackingState::TRACKING_FAILED;
		data.rawWeakTrackingFrames = 0;
		data.rawFailedTrackingFrames = 0;
		data.lastRawTrackingResult = ITMTrackingState::TRACKING_FAILED;
		forcePrimaryHandoff = false;
		forceNewArea = true;
		ACTIVE_MAP_CONSOLE("[MULTI ACTIVE] retire_new_map map=%d reason=%s\n",
			localMapIndex, reason.c_str());
		logTransitionEvent("retire_new_map", findPrimaryLocalMapIdx(), localMapIndex,
			"capacity", reason, -1, 0.0f, 0.0f, -1.0f, -1.0f,
			localMapManager->getLocalMapSize(localMapIndex), false, false);
		return true;
	}
	return false;
}

std::vector<int> ITMActiveMapManager::getActiveLocalMapIndices(void) const
{
	std::vector<int> result;
	for (size_t i = 0; i < activeData.size(); ++i) result.push_back(activeData[i].localMapIndex);
	return result;
}

bool ITMActiveMapManager::hasActiveNewLocalMap(void) const
{
	for (size_t i = 0; i < activeData.size(); ++i)
		if (activeData[i].type == NEW_LOCAL_MAP) return true;
	return false;
}

int ITMActiveMapManager::initiateNewLocalMap(bool isPrimaryLocalMap)
{
	int newIdx = localMapManager->createNewLocalMap();

	ActiveDataDescriptor newLink;
	initializeActiveDescriptor(newLink, newIdx, isPrimaryLocalMap ? PRIMARY_LOCAL_MAP : NEW_LOCAL_MAP);
	activeData.push_back(newLink);

	return newIdx;
}

bool ITMActiveMapManager::canInitiateNewLink(int localMapId) const
{
	for (size_t i = 0; i < activeData.size(); ++i)
	{
		if (activeData[i].type == LOOP_CLOSURE || activeData[i].type == RELOCALISATION)
			return false;
		if (activeData[i].localMapIndex == localMapId)
			return false;
	}
	return true;
}

int ITMActiveMapManager::initiateNewLink(int localMapId, const ORUtils::SE3Pose & pose, bool isRelocalisation)
{
	if (!canInitiateNewLink(localMapId)) return -1;

	// make sure only one relocalisation per local map is attempted at a time
	if (!localMapManager->resetTracking(localMapId, pose)) return -1;

	ActiveDataDescriptor newLink;
	initializeActiveDescriptor(newLink, localMapId, isRelocalisation ? RELOCALISATION : LOOP_CLOSURE);
	activeData.push_back(newLink);
	
	return (int)activeData.size() - 1;
}

float ITMActiveMapManager::visibleOriginalBlocks(int dataID) const
{
	int localMapId = activeData[dataID].localMapIndex;

	int allocated = localMapManager->getLocalMapSize(localMapId);
	int counted = localMapManager->countVisibleBlocks(localMapId, 0, originalBlocks, true);
	
	int tmp = originalBlocks;
	if (allocated < tmp) tmp = allocated;
	if (tmp <= 0) return 1.0f;
	return (float)counted / (float)tmp;
}

int ITMActiveMapManager::handoffMaturityMinAcceptedWindows(int dataID, bool allowEmergencyOverride) const
{
	if (dataID < 0 || dataID >= (int)activeData.size()) return newMapHandoffMinAcceptedWindows;
	if (allowEmergencyOverride)
	{
		if (activeData[dataID].lastSequentialTransformEmergency &&
			newMapHandoffEmergencyMinAcceptedWindows >= 0)
			return newMapHandoffEmergencyMinAcceptedWindows;
	}
	return newMapHandoffMinAcceptedWindows;
}

bool ITMActiveMapManager::handoffMaturitySatisfied(int dataID, bool allowEmergencyOverride) const
{
	if (dataID < 0 || dataID >= (int)activeData.size()) return false;
	const int minAcceptedWindows = handoffMaturityMinAcceptedWindows(dataID, allowEmergencyOverride);
	return activeData[dataID].acceptedSequentialWindows >= minAcceptedWindows;
}

bool ITMActiveMapManager::shouldStartNewArea(void) const
{
	if (!allowNewLocalMaps) return false;
	int primaryLocalMapIdx = -1;
	int primaryDataIdx = -1;

	// don't start two new local maps at a time
	for (int i = 0; i < (int)activeData.size(); ++i)
	{
		if (activeData[i].type == NEW_LOCAL_MAP) return false;
		if (activeData[i].type == PRIMARY_LOCAL_MAP) 
		{
			primaryDataIdx = i;
			primaryLocalMapIdx = activeData[i].localMapIndex;
		}
	}

	// TODO: check: if relocalisation fails for some time, start new local map
	if (primaryLocalMapIdx < 0) return false;
	else return forceNewArea || visibleOriginalBlocks(primaryDataIdx) < originalBlocksThreshold;

	return false;
}

bool ITMActiveMapManager::handoffVisibilitySatisfied(float primaryVisible, float newVisible) const
{
	if (newVisible < newMapHandoffMinVisibleRatio) return false;
	if (newMapHandoffMinVisibleMargin > 0.0f &&
		(newVisible - primaryVisible) < newMapHandoffMinVisibleMargin)
		return false;
	return true;
}

std::string ITMActiveMapManager::primaryFailureTriggerReason(const ActiveDataDescriptor &primary) const
{
	const bool primaryFailedEnough =
		primary.rawFailedTrackingFrames >= primaryFailureHandoffMinRawFailedFrames;
	const bool primaryWeakEnough =
		primaryFailureHandoffRawGate == "failed_or_weak" &&
		emergencyLinkMinRawWeakFrames > 0 &&
		primary.rawWeakTrackingFrames >= emergencyLinkMinRawWeakFrames;
	if (primaryFailedEnough && primaryWeakEnough) return "failed_and_weak";
	if (primaryFailedEnough) return "failed";
	if (primaryWeakEnough) return "weak";
	return "none";
}

bool ITMActiveMapManager::canPromoteCandidateAfterPrimaryFailure(int candidateDataIdx, int primaryDataIdx,
	int inliers, float candidateVisible, int candidateBlocks) const
{
	if (primaryFailureHandoffPolicy != "promote_visible_candidate") return false;
	if (!forcePrimaryHandoff) return false;
	if (candidateDataIdx < 0 || primaryDataIdx < 0) return false;
	if (candidateDataIdx >= (int)activeData.size() || primaryDataIdx >= (int)activeData.size()) return false;

	const ActiveDataDescriptor &candidate = activeData[candidateDataIdx];
	const ActiveDataDescriptor &primary = activeData[primaryDataIdx];
	if (candidate.type != NEW_LOCAL_MAP || primary.type != PRIMARY_LOCAL_MAP) return false;
	if (primaryFailureTriggerReason(primary) == "none") return false;
	if (candidate.lastTrackingResult != ITMTrackingState::TRACKING_GOOD) return false;
	if (candidate.goodTrackingFrames < primaryFailureHandoffMinCandidateGoodFrames) return false;
	if (candidateVisible < primaryFailureHandoffMinCandidateVisibleRatio) return false;
	if (candidateBlocks < primaryFailureHandoffMinCandidateBlocks) return false;
	if (inliers < primaryFailureHandoffMinOverlapFrames) return false;
	return true;
}

static bool isFiniteMatrix4f(const Matrix4f &matrix)
{
	for (int i = 0; i < 16; ++i)
		if (!std::isfinite(matrix.m[i])) return false;
	return true;
}

static float relativeTranslationNorm(const ORUtils::SE3Pose &lhs, const ORUtils::SE3Pose &rhs);
static float relativeRotationDeg(const ORUtils::SE3Pose &lhs, const ORUtils::SE3Pose &rhs);

bool ITMActiveMapManager::preserveCandidateGlobalPoseForHandoff(int candidateDataIdx, int primaryDataIdx,
	const std::string &reason)
{
	// validated_candidate keeps the independently tracked candidate pose after
	// a normal, validated overlap handoff, but emergency recovery must still
	// inherit the last finite primary global pose.
	if (handoffPosePolicy != "preserve_global" &&
		handoffPosePolicy != "validated_candidate") return false;
	if (candidateDataIdx < 0 || primaryDataIdx < 0) return false;
	if (candidateDataIdx >= (int)activeData.size() || primaryDataIdx >= (int)activeData.size()) return false;
	if (activeData[candidateDataIdx].type != NEW_LOCAL_MAP ||
		activeData[primaryDataIdx].type != PRIMARY_LOCAL_MAP)
		return false;

	const int primaryLocalMapIdx = activeData[primaryDataIdx].localMapIndex;
	const int candidateLocalMapIdx = activeData[candidateDataIdx].localMapIndex;
	const ORUtils::SE3Pose *primaryTrackingPose = localMapManager->getTrackingPose(primaryLocalMapIdx);
	const ORUtils::SE3Pose *candidateTrackingPose = localMapManager->getTrackingPose(candidateLocalMapIdx);
	if (primaryTrackingPose == NULL || candidateTrackingPose == NULL) return false;

	const ORUtils::SE3Pose &primaryGlobalToLocal =
		localMapManager->getEstimatedGlobalPose(primaryLocalMapIdx);
	const ORUtils::SE3Pose beforeCandidateGlobalToLocal =
		localMapManager->getEstimatedGlobalPose(candidateLocalMapIdx);
	const Matrix4f primaryGlobalC2w =
		primaryGlobalToLocal.GetInvM() * primaryTrackingPose->GetInvM();
	Matrix4f primaryGlobalW2c;
	if (!primaryGlobalC2w.inv(primaryGlobalW2c)) return false;

	const Matrix4f candidateLocalC2w = candidateTrackingPose->GetInvM();
	const Matrix4f candidateGlobalToLocalM = candidateLocalC2w * primaryGlobalW2c;
	if (!isFiniteMatrix4f(candidateGlobalToLocalM)) return false;

	ORUtils::SE3Pose candidateGlobalToLocal(candidateGlobalToLocalM);
	candidateGlobalToLocal.Coerce();
	if (!isFiniteMatrix4f(candidateGlobalToLocal.GetM())) return false;

	const float deltaM = relativeTranslationNorm(candidateGlobalToLocal, beforeCandidateGlobalToLocal);
	const float deltaDeg = relativeRotationDeg(candidateGlobalToLocal, beforeCandidateGlobalToLocal);
	localMapManager->setEstimatedGlobalPose(candidateLocalMapIdx, candidateGlobalToLocal);

	const float primaryVisible = visibleOriginalBlocks(primaryDataIdx);
	const float candidateVisible = visibleOriginalBlocks(candidateDataIdx);
	const int candidateBlocks = localMapManager->getLocalMapSize(candidateLocalMapIdx);
	ACTIVE_MAP_CONSOLE(
		"[MULTI ACTIVE] preserve_global_handoff primary=%d candidate=%d reason=%s delta_m=%.6f delta_deg=%.6f visible=%.6f blocks=%d\n",
		primaryLocalMapIdx, candidateLocalMapIdx, reason.c_str(), deltaM, deltaDeg,
		candidateVisible, candidateBlocks);
	logTransitionEvent("preserve_global_handoff", primaryLocalMapIdx, candidateLocalMapIdx,
		handoffPosePolicy, reason, -1, deltaM, deltaDeg, primaryVisible,
		candidateVisible, candidateBlocks, forcePrimaryHandoff, false,
		activeData[candidateDataIdx].goodTrackingFrames,
		activeData[candidateDataIdx].poorTrackingFrames,
		activeData[candidateDataIdx].failedTrackingFrames,
		activeData[candidateDataIdx].candidateRecoveryAttempts,
		activeData[primaryDataIdx].rawWeakTrackingFrames,
		activeData[primaryDataIdx].rawFailedTrackingFrames);
	return true;
}

bool ITMActiveMapManager::shouldMovePrimaryLocalMap(int newDataId, int bestDataId, int primaryDataId) const
{
	// A verified historical link is a pose-graph constraint, not a reason to
	// hand tracking ownership back to an old map. Relocalisation handles the
	// separate case where no healthy primary remains.
	if (newDataId >= 0 && activeData[newDataId].type == LOOP_CLOSURE && primaryDataId >= 0)
		return false;

	int localMapIdx_primary = -1;
	int localMapIdx_best = -1;
	int localMapIdx_new = -1;

	int blocksInUse_primary = -1;
	float visibleRatio_primary = 1.0f;
	int blocksInUse_best = -1;
	float visibleRatio_best = 1.0f;
	bool isNewLocalMap_best = false;
	int blocksInUse_new = -1;
	float visibleRatio_new = 1.0f;
	bool isNewLocalMap_new = false;

	if (primaryDataId >= 0) localMapIdx_primary = activeData[primaryDataId].localMapIndex;
	if (bestDataId >= 0) localMapIdx_best = activeData[bestDataId].localMapIndex;
	if (newDataId >= 0) localMapIdx_new = activeData[newDataId].localMapIndex;

	// count blocks in all relevant localMaps
	if (localMapIdx_primary >= 0) 
	{
		blocksInUse_primary = localMapManager->getLocalMapSize(localMapIdx_primary);
		visibleRatio_primary = visibleOriginalBlocks(primaryDataId);
	}

	if (localMapIdx_new >= 0) 
	{
		isNewLocalMap_new = (activeData[newDataId].type == NEW_LOCAL_MAP);
		blocksInUse_new = localMapManager->getLocalMapSize(localMapIdx_new);
		if (blocksInUse_new < 0) return false;
		visibleRatio_new = visibleOriginalBlocks(newDataId);
	}

	if (localMapIdx_best >= 0) 
	{
		isNewLocalMap_best = (activeData[bestDataId].type == NEW_LOCAL_MAP);
		blocksInUse_best = localMapManager->getLocalMapSize(localMapIdx_best);
		visibleRatio_best = visibleOriginalBlocks(bestDataId);
	}

	if (blocksInUse_primary < 0) {
		// TODO: if relocalisation fails, a new local map gets started,
		//       and is eventually accepted, this case will get relevant
		return true;
	}

	// step 1: is "new" better than "primary" ?

	// don't continue a local map that is already full
/*	if (blocksInUse_new >= N_maxblocknum) return false;

	if (blocksInUse_new >= blocksInUse_primary) return false;*/
			if (forcePrimaryHandoff && isNewLocalMap_new && blocksInUse_new > 0)
			{
				const int minAcceptedWindows = handoffMaturityMinAcceptedWindows(newDataId, true);
				if (!handoffMaturitySatisfied(newDataId, true))
				{
					ACTIVE_MAP_CONSOLE(
						"[MULTI ACTIVE] capacity_handoff_reject primary=%d new=%d accepted_windows=%d min_accepts=%d emergency=%d forced=%d reason=maturity_gate\n",
						localMapIdx_primary, localMapIdx_new, activeData[newDataId].acceptedSequentialWindows,
						minAcceptedWindows, activeData[newDataId].lastSequentialTransformEmergency ? 1 : 0,
						forcePrimaryHandoff ? 1 : 0);
					logTransitionEvent("capacity_handoff_reject", localMapIdx_primary, localMapIdx_new,
						newMapHandoffPolicy, "wait_maturity", -1, 0.0f, 0.0f,
						visibleRatio_primary, visibleRatio_new, blocksInUse_new, true, false,
						activeData[newDataId].goodTrackingFrames, activeData[newDataId].poorTrackingFrames,
						activeData[newDataId].failedTrackingFrames, activeData[newDataId].candidateRecoveryAttempts,
						primaryDataId >= 0 ? activeData[primaryDataId].rawWeakTrackingFrames : -1,
						primaryDataId >= 0 ? activeData[primaryDataId].rawFailedTrackingFrames : -1,
						"accepted_windows=" + std::to_string(activeData[newDataId].acceptedSequentialWindows) +
						" min_accepts=" + std::to_string(minAcceptedWindows) +
						" emergency=" + std::to_string(activeData[newDataId].lastSequentialTransformEmergency ? 1 : 0) +
						" forced=" + std::to_string(forcePrimaryHandoff ? 1 : 0));
					return false;
				}
				if (!handoffVisibilitySatisfied(visibleRatio_primary, visibleRatio_new))
				{
					ACTIVE_MAP_CONSOLE(
					"[MULTI ACTIVE] capacity_handoff_reject primary=%d new=%d primary_visible=%.6f new_visible=%.6f new_blocks=%d reason=visibility_gate min_visible=%.6f min_margin=%.6f\n",
					localMapIdx_primary, localMapIdx_new, visibleRatio_primary, visibleRatio_new, blocksInUse_new,
					newMapHandoffMinVisibleRatio, newMapHandoffMinVisibleMargin);
				logTransitionEvent("capacity_handoff_reject", localMapIdx_primary, localMapIdx_new,
					newMapHandoffPolicy, "wait_visibility_gate", -1, 0.0f, 0.0f,
					visibleRatio_primary, visibleRatio_new, blocksInUse_new, true, false);
				return false;
			}
			if (newMapHandoffPolicy == "validated_visibility" &&
				!activeData[newDataId].lastSequentialTransformAccepted)
			{
				ACTIVE_MAP_CONSOLE(
					"[MULTI ACTIVE] capacity_handoff_reject primary=%d new=%d primary_visible=%.6f new_visible=%.6f new_blocks=%d reason=invalid_transform\n",
					localMapIdx_primary, localMapIdx_new, visibleRatio_primary, visibleRatio_new, blocksInUse_new);
				logTransitionEvent("capacity_handoff_reject", localMapIdx_primary, localMapIdx_new,
					newMapHandoffPolicy, "wait_transform", -1, 0.0f, 0.0f,
					visibleRatio_primary, visibleRatio_new, blocksInUse_new, true, false);
				return false;
			}
			ACTIVE_MAP_CONSOLE(
				"[MULTI ACTIVE] capacity_handoff primary=%d new=%d primary_visible=%.6f new_visible=%.6f new_blocks=%d\n",
				localMapIdx_primary, localMapIdx_new, visibleRatio_primary, visibleRatio_new, blocksInUse_new);
			logTransitionEvent("capacity_handoff", localMapIdx_primary, localMapIdx_new,
				newMapHandoffPolicy, "move", -1, 0.0f, 0.0f,
				visibleRatio_primary, visibleRatio_new, blocksInUse_new, true, true);
			return true;
			}
			if (newMapHandoffPolicy == "accepted" && isNewLocalMap_new && blocksInUse_new > 0 &&
				activeData[newDataId].lastSequentialTransformAccepted)
			{
				const int minAcceptedWindows = handoffMaturityMinAcceptedWindows(newDataId, true);
				if (!handoffMaturitySatisfied(newDataId, true))
				{
					logTransitionEvent("accepted_handoff_reject", localMapIdx_primary, localMapIdx_new,
						newMapHandoffPolicy, "wait_maturity", -1, 0.0f, 0.0f,
						visibleRatio_primary, visibleRatio_new, blocksInUse_new, forcePrimaryHandoff, false,
						activeData[newDataId].goodTrackingFrames, activeData[newDataId].poorTrackingFrames,
						activeData[newDataId].failedTrackingFrames, activeData[newDataId].candidateRecoveryAttempts,
						primaryDataId >= 0 ? activeData[primaryDataId].rawWeakTrackingFrames : -1,
						primaryDataId >= 0 ? activeData[primaryDataId].rawFailedTrackingFrames : -1,
						"accepted_windows=" + std::to_string(activeData[newDataId].acceptedSequentialWindows) +
						" min_accepts=" + std::to_string(minAcceptedWindows) +
						" emergency=" + std::to_string(activeData[newDataId].lastSequentialTransformEmergency ? 1 : 0) +
						" forced=" + std::to_string(forcePrimaryHandoff ? 1 : 0));
					return false;
				}
				if (!handoffVisibilitySatisfied(visibleRatio_primary, visibleRatio_new))
				{
					logTransitionEvent("accepted_handoff_reject", localMapIdx_primary, localMapIdx_new,
					newMapHandoffPolicy, "wait_visibility_gate", -1, 0.0f, 0.0f,
					visibleRatio_primary, visibleRatio_new, blocksInUse_new, forcePrimaryHandoff, false);
				return false;
			}
			ACTIVE_MAP_CONSOLE(
				"[MULTI ACTIVE] accepted_handoff primary=%d new=%d primary_visible=%.6f new_visible=%.6f new_blocks=%d\n",
				localMapIdx_primary, localMapIdx_new, visibleRatio_primary, visibleRatio_new, blocksInUse_new);
			logTransitionEvent("accepted_handoff", localMapIdx_primary, localMapIdx_new,
				newMapHandoffPolicy, "move", -1, 0.0f, 0.0f,
				visibleRatio_primary, visibleRatio_new, blocksInUse_new, forcePrimaryHandoff, true);
			return true;
		}
			if (newMapHandoffPolicy == "validated_visibility" && isNewLocalMap_new &&
				!activeData[newDataId].lastSequentialTransformAccepted)
			{
				logTransitionEvent("visibility_reject", localMapIdx_primary, localMapIdx_new,
				newMapHandoffPolicy, "wait_transform", -1, 0.0f, 0.0f,
				visibleRatio_primary, visibleRatio_new, blocksInUse_new, forcePrimaryHandoff, false);
				return false;
			}
			if (!allowVisibilityHandoff && !forcePrimaryHandoff)
			{
				if (isNewLocalMap_new)
					logTransitionEvent("visibility_reject", localMapIdx_primary, localMapIdx_new,
						newMapHandoffPolicy, "disabled", -1, 0.0f, 0.0f,
						visibleRatio_primary, visibleRatio_new, blocksInUse_new, forcePrimaryHandoff, false);
				return false;
			}
			const int minAcceptedWindows = handoffMaturityMinAcceptedWindows(newDataId, true);
			if (isNewLocalMap_new && !handoffMaturitySatisfied(newDataId, true))
			{
				logTransitionEvent("visibility_reject", localMapIdx_primary, localMapIdx_new,
					newMapHandoffPolicy, "wait_maturity", -1, 0.0f, 0.0f,
					visibleRatio_primary, visibleRatio_new, blocksInUse_new, forcePrimaryHandoff, false,
					activeData[newDataId].goodTrackingFrames, activeData[newDataId].poorTrackingFrames,
					activeData[newDataId].failedTrackingFrames, activeData[newDataId].candidateRecoveryAttempts,
					primaryDataId >= 0 ? activeData[primaryDataId].rawWeakTrackingFrames : -1,
					primaryDataId >= 0 ? activeData[primaryDataId].rawFailedTrackingFrames : -1,
					"accepted_windows=" + std::to_string(activeData[newDataId].acceptedSequentialWindows) +
					" min_accepts=" + std::to_string(minAcceptedWindows) +
					" emergency=" + std::to_string(activeData[newDataId].lastSequentialTransformEmergency ? 1 : 0) +
					" forced=" + std::to_string(forcePrimaryHandoff ? 1 : 0));
				return false;
			}
			if (visibleRatio_new <= visibleRatio_primary)
			{
			if (isNewLocalMap_new)
				logTransitionEvent("visibility_reject", localMapIdx_primary, localMapIdx_new,
					newMapHandoffPolicy, "keep_primary", -1, 0.0f, 0.0f,
					visibleRatio_primary, visibleRatio_new, blocksInUse_new, forcePrimaryHandoff, false);
			return false;
		}
		if (isNewLocalMap_new && !handoffVisibilitySatisfied(visibleRatio_primary, visibleRatio_new))
		{
			logTransitionEvent("visibility_reject", localMapIdx_primary, localMapIdx_new,
				newMapHandoffPolicy, "wait_visibility_gate", -1, 0.0f, 0.0f,
				visibleRatio_primary, visibleRatio_new, blocksInUse_new, forcePrimaryHandoff, false);
			return false;
		}

	// step 2: is there any contender for a new local map to move to?
	if (blocksInUse_best < 0)
	{
		if (isNewLocalMap_new)
			logTransitionEvent("visibility_handoff", localMapIdx_primary, localMapIdx_new,
				newMapHandoffPolicy, "move", -1, 0.0f, 0.0f,
				visibleRatio_primary, visibleRatio_new, blocksInUse_new, forcePrimaryHandoff, true);
		return true;
	}

	// if this is a new local map, but we previously found that we can loop
	// close, don't accept the new local map!
	if (isNewLocalMap_new && !isNewLocalMap_best) return false;
	// if this is a loop closure and we have not found any alternative
	// loop closure before, accept the new one!
	if (!isNewLocalMap_new && isNewLocalMap_best) return true;

	// if the two are equal, take the smaller one
	//return (blocksInUse_new < blocksInUse_best);
	{
		const bool move = visibleRatio_new > visibleRatio_best;
		if (isNewLocalMap_new)
			logTransitionEvent(move ? "visibility_handoff" : "visibility_reject", localMapIdx_primary, localMapIdx_new,
				newMapHandoffPolicy, move ? "move" : "keep_primary", -1, 0.0f, 0.0f,
				visibleRatio_primary, visibleRatio_new, blocksInUse_new, forcePrimaryHandoff, move);
		return move;
	}
}

int ITMActiveMapManager::findPrimaryDataIdx(void) const
{
	for (int i = 0; i < (int)activeData.size(); ++i) 
		if (activeData[i].type == PRIMARY_LOCAL_MAP) return i;

	return -1;
}

int ITMActiveMapManager::findPrimaryLocalMapIdx(void) const
{
	int id = findPrimaryDataIdx();
	if (id < 0) return -1;
	return activeData[id].localMapIndex;
}

int ITMActiveMapManager::findBestVisualisationDataIdx(void) const
{
	int bestIdx = -1;
	for (int i = 0; i < static_cast<int>(activeData.size()); ++i) 
	{
		if (activeData[i].type == PRIMARY_LOCAL_MAP) return i;
		else if (activeData[i].type == NEW_LOCAL_MAP) bestIdx = i;
		else if (activeData[i].type == RELOCALISATION) 
		{
			if (bestIdx < 0) { bestIdx = i; continue; }
			if (activeData[bestIdx].type == NEW_LOCAL_MAP) continue;
			if (activeData[bestIdx].constraints.size() < activeData[i].constraints.size()) bestIdx = i;
		}
	}
	return bestIdx;
}

int ITMActiveMapManager::findBestVisualisationLocalMapIdx(void) const
{
	int id = findBestVisualisationDataIdx();
	if (id < 0) return -1;
	return activeData[id].localMapIndex;
}

void ITMActiveMapManager::recordTrackingResult(int dataID, ITMTrackingState::TrackingResult trackingResult, bool primaryTrackingSuccess)
{
	ActiveDataDescriptor & data = activeData[dataID];

	int primaryLocalMapID = findPrimaryLocalMapIdx();
	int localMapId = data.localMapIndex;
	data.trackingAttempts++;
	data.lastTrackingResult = trackingResult;
	if (trackingResult == ITMTrackingState::TRACKING_GOOD) data.goodTrackingFrames++;
	else if (trackingResult == ITMTrackingState::TRACKING_POOR) data.poorTrackingFrames++;
	else if (trackingResult == ITMTrackingState::TRACKING_FAILED) data.failedTrackingFrames++;

	if (trackingResult == ITMTrackingState::TRACKING_GOOD)
	{
		if (data.type == RELOCALISATION) data.constraints.push_back(localMapManager->getTrackingPose(localMapId)->GetM());
		else if (((data.type == NEW_LOCAL_MAP) || (data.type == LOOP_CLOSURE)) && primaryTrackingSuccess)
		{
			Matrix4f Tnew_inv = localMapManager->getTrackingPose(localMapId)->GetInvM();
			Matrix4f Told = localMapManager->getTrackingPose(primaryLocalMapID)->GetM();
			Matrix4f Told_to_new = Tnew_inv * Told;

			data.constraints.push_back(Told_to_new);
		}
	}
	else if (trackingResult == ITMTrackingState::TRACKING_FAILED)
	{
		if (data.type == PRIMARY_LOCAL_MAP)
		{
			for (size_t j = 0; j < activeData.size(); ++j)
			{
				if (activeData[j].type == NEW_LOCAL_MAP) activeData[j].type = LOST_NEW;
				else activeData[j].type = LOST;
			}
		}
	}
}

void ITMActiveMapManager::recordRawTrackingResult(int dataID, ITMTrackingState::TrackingResult rawTrackingResult)
{
	ActiveDataDescriptor &data = activeData[dataID];
	data.lastRawTrackingResult = rawTrackingResult;
	if (rawTrackingResult != ITMTrackingState::TRACKING_GOOD) data.rawWeakTrackingFrames++;
	else data.rawWeakTrackingFrames = 0;
	if (rawTrackingResult == ITMTrackingState::TRACKING_FAILED) data.rawFailedTrackingFrames++;
	else data.rawFailedTrackingFrames = 0;
}

static float huber_weight(float residual, float b)
{
	double r_abs = fabs(residual);
	if (r_abs < b) return 1.0f;
	double tmp = 2.0 * b * r_abs - b*b;
	return (float)(sqrt(tmp) / r_abs);
}

static float relativeTranslationNorm(const ORUtils::SE3Pose &lhs, const ORUtils::SE3Pose &rhs)
{
	const Matrix4f delta = lhs.GetM() * rhs.GetInvM();
	return std::sqrt(delta.m30 * delta.m30 + delta.m31 * delta.m31 + delta.m32 * delta.m32);
}

static float relativeRotationDeg(const ORUtils::SE3Pose &lhs, const ORUtils::SE3Pose &rhs)
{
	const Matrix4f delta = lhs.GetM() * rhs.GetInvM();
	float cosAngle = (delta.m00 + delta.m11 + delta.m22 - 1.0f) * 0.5f;
	cosAngle = std::max(-1.0f, std::min(1.0f, cosAngle));
	return std::acos(cosAngle) * 57.29577951308232f;
}

/** estimate a relative pose, taking into account a previous estimate (weight 0
	indicates that no previous estimate is available).
	out_numInliers and out_inlierPose is the number of inliers from amongst the
	new observations and the pose computed from them.
*/
static ORUtils::SE3Pose estimateRelativePose(const std::vector<Matrix4f> & observations, const ORUtils::SE3Pose & previousEstimate, float previousEstimate_weight, int *out_numInliers, ORUtils::SE3Pose *out_inlierPose)
{
	static const float huber_b = 0.1f;
	static const float weightsConverged = 0.01f;
	static const int maxIter = 10;
	static const float inlierThresholdForFinalResult = 0.8f;
	if (observations.empty() && previousEstimate_weight <= 0.0f)
	{
		if (out_numInliers) *out_numInliers = 0;
		if (out_inlierPose) *out_inlierPose = ORUtils::SE3Pose();
		return ORUtils::SE3Pose();
	}
	std::vector<float> weights(observations.size() + 1, 1.0f);
	std::vector<ORUtils::SE3Pose> poses;

	for (size_t i = 0; i < observations.size(); ++i) poses.push_back(ORUtils::SE3Pose(observations[i]));

	float params[6];
	for (int iter = 0; iter < maxIter; ++iter) 
	{
		// estimate with fixed weights
		float sumweight = previousEstimate_weight;
		for (int j = 0; j < 6; ++j) params[j] = weights.back() * previousEstimate_weight * previousEstimate.GetParams()[j];
		for (size_t i = 0; i < poses.size(); ++i) 
		{
			for (int j = 0; j < 6; ++j) params[j] += weights[i] * poses[i].GetParams()[j];
			sumweight += weights[i];
		}
		for (int j = 0; j < 6; ++j) params[j] /= sumweight;

		// compute new weights
		float weightchanges = 0.0f;
		for (size_t i = 0; i < weights.size(); ++i) 
		{
			const ORUtils::SE3Pose *p;
			float w = 1.0f;
			if (i < poses.size()) p = &(poses[i]);
			else
			{
				p = &(previousEstimate);
				w = previousEstimate_weight;
			}

			float residual = 0.0f;
			for (int j = 0; j < 6; ++j) 
			{
				float r = p->GetParams()[j] - params[j];
				residual += r*r;
			}
			residual = sqrt(residual);
			float newweight = huber_weight(residual, huber_b);
			weightchanges += w * fabs(newweight - weights[i]);
			weights[i] = newweight;
		}

		float avgweightchange = weightchanges / (weights.size() - 1 + previousEstimate_weight);
		if (avgweightchange < weightsConverged) break;
	}

	int inliers = 0;
	Matrix4f inlierTrafo;
	inlierTrafo.setZeros();

	for (size_t i = 0; i < poses.size(); ++i) if (weights[i] > inlierThresholdForFinalResult) 
	{
		inlierTrafo += observations[i];
		++inliers;
	}
	if (out_inlierPose) out_inlierPose->SetM(inlierTrafo / (float)MAX(inliers, 1));
	if (out_numInliers) *out_numInliers = inliers;

	return ORUtils::SE3Pose(params);
}

int ITMActiveMapManager::CheckSuccess_relocalisation(int dataID) const
{
	// sucessfully relocalised
	if (activeData[dataID].constraints.size() >= N_relocsuccess) return 1;

	// relocalisation failed: declare as LOST
	if ((N_reloctrials - activeData[dataID].trackingAttempts) < (N_relocsuccess - (int)activeData[dataID].constraints.size())) return -1;

	// keep trying
	return 0;
}

int ITMActiveMapManager::CheckSuccess_newlink(int dataID, int primaryDataID, int *inliers, ORUtils::SE3Pose *inlierPose) const
{
	const ActiveDataDescriptor & link = activeData[dataID];

	// take previous data from local map relations into account!
	//ORUtils::SE3Pose previousEstimate;
	//int previousEstimate_weight = 0;
	int primaryLocalMapIndex = -1;
	if (primaryDataID >= 0) primaryLocalMapIndex = activeData[primaryDataID].localMapIndex;
	const ITMPoseConstraint & previousInformation = localMapManager->getRelation_const(primaryLocalMapIndex, link.localMapIndex);
	/* hmm... do we want the "Estimate" (i.e. the pose corrected by pose
	   graph optimization) or the "Observations" (i.e. the accumulated
	   poses seen in previous frames?
	   This should only really make a difference, if there is a large
	   disagreement between the two, in which case one might argue that
	   most likely something went wrong with a loop-closure, and we are
	   not really sure the "Estimate" is true or just based on an erroneous
	   loop closure. We therefore want to be consistent with previous
	   observations not estimations...
	*/

	ORUtils::SE3Pose previousEstimate = previousInformation.GetAccumulatedObservations();
	int previousEstimate_weight = previousInformation.GetNumAccumulatedObservations();

	int inliers_local;
	ORUtils::SE3Pose inlierPose_local;
	if (inliers == NULL) inliers = &inliers_local;
	if (inlierPose == NULL) inlierPose = &inlierPose_local;

	estimateRelativePose(link.constraints, previousEstimate, (float)previousEstimate_weight, inliers, inlierPose);

		int emergencyRequiredOverlap = 0;
		if (forcePrimaryHandoff && link.type == NEW_LOCAL_MAP &&
			emergencyLinkOverlap > 0 && primaryDataID >= 0)
		{
			const float candidateVisible = visibleOriginalBlocks(dataID);
			const int candidateBlocks = localMapManager->getLocalMapSize(link.localMapIndex);
			if (canPromoteCandidateAfterPrimaryFailure(dataID, primaryDataID, *inliers, candidateVisible, candidateBlocks))
			{
				emergencyRequiredOverlap = emergencyLinkOverlap;
				if (primaryFailureHandoffMinOverlapFrames > 0)
					emergencyRequiredOverlap = MAX(emergencyRequiredOverlap, primaryFailureHandoffMinOverlapFrames);
				emergencyRequiredOverlap = MIN(linkOverlap, emergencyRequiredOverlap);
			}
		}

		// accept link
		if (*inliers >= linkOverlap) return 1;
		if (emergencyRequiredOverlap > 0 && *inliers >= emergencyRequiredOverlap)
			return 2;

		// reject link
		const int requiredOverlap = emergencyRequiredOverlap > 0 ? emergencyRequiredOverlap : linkOverlap;
		if ((linkTrials - link.trackingAttempts) < (requiredOverlap - *inliers)) return -1;

	// keep trying
	return 0;
}

void ITMActiveMapManager::AcceptNewLink(int fromData, int toData, const ORUtils::SE3Pose & pose, int weight)
{
	int fromLocalMapIdx = activeData[fromData].localMapIndex;
	int toLocalMapIdx = activeData[toData].localMapIndex;

	{
		ITMPoseConstraint &c = localMapManager->getRelation(fromLocalMapIdx, toLocalMapIdx);
		c.AddObservation(pose, weight);
	}
	{
		ORUtils::SE3Pose invPose(pose.GetInvM());
		ITMPoseConstraint &c = localMapManager->getRelation(toLocalMapIdx, fromLocalMapIdx);
		c.AddObservation(invPose, weight);
	}
}

bool ITMActiveMapManager::maintainActiveData(int frameId)
{
	diagnosticFrame = frameId;
	bool localMapGraphChanged = false;

	int primaryDataIdx = findPrimaryDataIdx();
	int moveToDataIdx = -1;
	auto resetLinkRuntimeState = [](ActiveDataDescriptor &link, bool resetRecoveryAttempts)
	{
		link.constraints.clear();
		link.trackingAttempts = 0;
		link.goodTrackingFrames = 0;
		link.poorTrackingFrames = 0;
		link.failedTrackingFrames = 0;
		link.lastTrackingResult = ITMTrackingState::TRACKING_FAILED;
		link.rawWeakTrackingFrames = 0;
		link.rawFailedTrackingFrames = 0;
		link.lastRawTrackingResult = ITMTrackingState::TRACKING_FAILED;
		if (resetRecoveryAttempts) link.candidateRecoveryAttempts = 0;
	};

	for (int i = 0; i < (int)activeData.size(); ++i)
	{
		ActiveDataDescriptor & link = activeData[i];

		if (link.type == RELOCALISATION)
		{
			int success = CheckSuccess_relocalisation(i);
			if (success == 1)
			{
				if (moveToDataIdx < 0) moveToDataIdx = i;
				else link.type = LOST;
			}
			else if (success == -1) link.type = LOST;
		}

		if ((link.type == LOOP_CLOSURE) || (link.type == NEW_LOCAL_MAP))
		{
			ORUtils::SE3Pose inlierPose; int inliers;

			int success = CheckSuccess_newlink(i, primaryDataIdx, &inliers, &inlierPose);
			if (success == 1 || success == 2)
			{
				const bool emergencyLink = success == 2;
				bool storeGraphObservation = true;
				bool restartInvalidSequentialCandidate = false;
				bool emergencyPrimaryFailurePromotion = false;
				link.lastSequentialTransformAccepted = false;
				link.lastSequentialTransformEmergency = emergencyLink;

					// Sequential submaps must inherit the accepted overlap transform
					// even when global pose-graph adjustment is disabled. The initial
					// transform assigned when the candidate is created is only a
					// rough camera-pose estimate; retaining it causes a discontinuity
					// every time the candidate becomes primary.
					if ((link.type == NEW_LOCAL_MAP) && (primaryDataIdx >= 0))
					{
						link.acceptedSequentialWindows++;
						const int primaryLocalMapIdx = activeData[primaryDataIdx].localMapIndex;
						const int newLocalMapIdx = link.localMapIndex;
					ORUtils::SE3Pose observedGlobalPose(
						inlierPose.GetM() * localMapManager->getEstimatedGlobalPose(primaryLocalMapIdx).GetM());
					const ORUtils::SE3Pose initialGlobalPose = localMapManager->getEstimatedGlobalPose(newLocalMapIdx);
					const float transformDeltaM = relativeTranslationNorm(observedGlobalPose, initialGlobalPose);
					const float transformDeltaDeg = relativeRotationDeg(observedGlobalPose, initialGlobalPose);
					bool useObservedPose = false;
					if (sequentialTransformPolicy == "overlap") useObservedPose = true;
					else if (sequentialTransformPolicy == "initial") useObservedPose = false;
					else if (sequentialTransformPolicy == "validated_overlap")
					{
						useObservedPose =
							transformDeltaM <= sequentialTransformMaxTranslation &&
							transformDeltaDeg <= sequentialTransformMaxRotationDeg;
					}
					else
					{
						fprintf(stderr, "[MULTI ACTIVE] unsupported sequential_transform_policy=%s, keeping initial transform\n",
							sequentialTransformPolicy.c_str());
						useObservedPose = false;
					}

					if (sequentialTransformPolicy == "validated_overlap" && !useObservedPose)
						storeGraphObservation = false;
					if (newMapHandoffPolicy == "validated_visibility" && !useObservedPose)
						storeGraphObservation = false;
					if (emergencyLink && !useObservedPose)
						storeGraphObservation = false;

					if (useObservedPose)
					{
						localMapManager->setEstimatedGlobalPose(newLocalMapIdx, observedGlobalPose);
						link.invalidSequentialTransformAttempts = 0;
					}
					else
					{
						link.invalidSequentialTransformAttempts++;
						restartInvalidSequentialCandidate =
							invalidTransformRestartAttempts > 0 &&
							link.invalidSequentialTransformAttempts >= invalidTransformRestartAttempts;
					}
					link.lastSequentialTransformAccepted = useObservedPose;

					const float primaryVisible = visibleOriginalBlocks(primaryDataIdx);
					const float newVisible = visibleOriginalBlocks(i);
					const int newBlocks = localMapManager->getLocalMapSize(newLocalMapIdx);
					emergencyPrimaryFailurePromotion =
						!useObservedPose &&
						canPromoteCandidateAfterPrimaryFailure(i, primaryDataIdx, inliers, newVisible, newBlocks);
					bool preservedGlobal = false;
					if (emergencyPrimaryFailurePromotion)
					{
						storeGraphObservation = false;
						restartInvalidSequentialCandidate = false;
						preservedGlobal =
							preserveCandidateGlobalPoseForHandoff(i, primaryDataIdx, "rejected_transform");
						if (primaryFailureHandoffRequirePoseContinuity && !preservedGlobal)
						{
							emergencyPrimaryFailurePromotion = false;
							restartInvalidSequentialCandidate = true;
							ACTIVE_MAP_CONSOLE(
								"[MULTI ACTIVE] primary_failure_handoff_reject primary=%d candidate=%d reason=rejected_transform detail=pose_continuity_unavailable\n",
								primaryLocalMapIdx, newLocalMapIdx);
							logTransitionEvent("primary_failure_handoff_reject", primaryLocalMapIdx, newLocalMapIdx,
								primaryFailureHandoffPolicy, "pose_continuity_unavailable", inliers,
								transformDeltaM, transformDeltaDeg, primaryVisible, newVisible, newBlocks,
								forcePrimaryHandoff, false, link.goodTrackingFrames, link.poorTrackingFrames,
								link.failedTrackingFrames, link.candidateRecoveryAttempts,
								activeData[primaryDataIdx].rawWeakTrackingFrames,
								activeData[primaryDataIdx].rawFailedTrackingFrames,
								"rejected_transform pose_continuity_unavailable");
						}
					}
					if (emergencyPrimaryFailurePromotion)
					{
							ACTIVE_MAP_CONSOLE(
								"[MULTI ACTIVE] primary_failure_handoff primary=%d new=%d raw_weak=%d raw_failed=%d candidate_good=%d visible=%.6f blocks=%d inliers=%d reason=rejected_transform preserve_global=%d\n",
								primaryLocalMapIdx, newLocalMapIdx, activeData[primaryDataIdx].rawWeakTrackingFrames,
								activeData[primaryDataIdx].rawFailedTrackingFrames,
								link.goodTrackingFrames, newVisible, newBlocks, inliers, preservedGlobal ? 1 : 0);
							logTransitionEvent("primary_failure_handoff", primaryLocalMapIdx, newLocalMapIdx,
									primaryFailureHandoffPolicy, "promote_visible_candidate", inliers,
									transformDeltaM, transformDeltaDeg, primaryVisible, newVisible, newBlocks,
									forcePrimaryHandoff, true, link.goodTrackingFrames, link.poorTrackingFrames,
									link.failedTrackingFrames, link.candidateRecoveryAttempts,
									activeData[primaryDataIdx].rawWeakTrackingFrames,
									activeData[primaryDataIdx].rawFailedTrackingFrames,
									std::string("rejected_transform trigger=") +
										primaryFailureTriggerReason(activeData[primaryDataIdx]) +
										" preserve_global=" + (preservedGlobal ? "1" : "0"));
					}
					if (!emergencyPrimaryFailurePromotion && handoffPosePolicy == "preserve_global")
					{
						const ORUtils::SE3Pose &primaryGlobalToLocal = localMapManager->getEstimatedGlobalPose(primaryLocalMapIdx);
						const ORUtils::SE3Pose &newGlobalToLocal = localMapManager->getEstimatedGlobalPose(newLocalMapIdx);
						const Matrix4f primaryGlobalC2w =
							primaryGlobalToLocal.GetInvM() * localMapManager->getTrackingPose(primaryLocalMapIdx)->GetInvM();
						ORUtils::SE3Pose newLocalTrackingPose;
						newLocalTrackingPose.SetInvM(newGlobalToLocal.GetM() * primaryGlobalC2w);
						newLocalTrackingPose.Coerce();
						localMapManager->resetTracking(newLocalMapIdx, newLocalTrackingPose);
					}
						else if (handoffPosePolicy != "candidate" &&
							handoffPosePolicy != "preserve_global" &&
							handoffPosePolicy != "validated_candidate")
					{
						fprintf(stderr, "[MULTI ACTIVE] unsupported handoff_pose_policy=%s, keeping candidate tracking pose\n",
							handoffPosePolicy.c_str());
					}
						ACTIVE_MAP_CONSOLE(
							"[MULTI ACTIVE] sequential_transform primary=%d new=%d policy=%s action=%s handoff_pose=%s inliers=%d delta_m=%.6f delta_deg=%.6f max_m=%.6f max_deg=%.6f emergency=%d graph_observation=%d invalid_attempts=%d restart=%d candidate_good=%d primary_raw_weak=%d primary_raw_failed=%d\n",
							primaryLocalMapIdx, newLocalMapIdx, sequentialTransformPolicy.c_str(),
							useObservedPose ? "observed" : "initial", handoffPosePolicy.c_str(), inliers, transformDeltaM, transformDeltaDeg,
							sequentialTransformMaxTranslation, sequentialTransformMaxRotationDeg, emergencyLink ? 1 : 0, storeGraphObservation ? 1 : 0,
							link.invalidSequentialTransformAttempts, restartInvalidSequentialCandidate ? 1 : 0,
							link.goodTrackingFrames, activeData[primaryDataIdx].rawWeakTrackingFrames,
							activeData[primaryDataIdx].rawFailedTrackingFrames);
						logTransitionEvent("sequential_transform", primaryLocalMapIdx, newLocalMapIdx,
							sequentialTransformPolicy, emergencyLink ? (useObservedPose ? "emergency_observed" : "emergency_initial") : (useObservedPose ? "observed" : "initial"), inliers,
							transformDeltaM, transformDeltaDeg, -1.0f, -1.0f, -1,
							forcePrimaryHandoff, false, link.goodTrackingFrames, link.poorTrackingFrames,
							link.failedTrackingFrames, link.candidateRecoveryAttempts,
							activeData[primaryDataIdx].rawWeakTrackingFrames,
							activeData[primaryDataIdx].rawFailedTrackingFrames);
					if (!storeGraphObservation)
						logTransitionEvent("sequential_transform_reject", primaryLocalMapIdx, newLocalMapIdx,
							sequentialTransformPolicy, "skip_graph_observation", inliers,
							transformDeltaM, transformDeltaDeg, -1.0f, -1.0f, -1,
							forcePrimaryHandoff, false, -1, -1, -1, -1, -1, -1,
							useObservedPose ? "graph_observation_disabled" : "weak_sequential_edge_rejected_transform");
					if (restartInvalidSequentialCandidate)
					{
						ACTIVE_MAP_CONSOLE(
							"[MULTI ACTIVE] sequential_transform_restart primary=%d new=%d invalid_attempts=%d threshold=%d\n",
							primaryLocalMapIdx, newLocalMapIdx, link.invalidSequentialTransformAttempts, invalidTransformRestartAttempts);
						logTransitionEvent("sequential_transform_restart", primaryLocalMapIdx, newLocalMapIdx,
							sequentialTransformPolicy, "restart_candidate", inliers,
							transformDeltaM, transformDeltaDeg, -1.0f, -1.0f, -1,
							forcePrimaryHandoff, false);
					}
				}
					if (restartInvalidSequentialCandidate)
					{
						link.type = LOST_NEW;
						resetLinkRuntimeState(link, true);
						link.acceptedSequentialWindows = 0;
						link.lastSequentialTransformEmergency = false;
						localMapGraphChanged = true;
						continue;
					}
				if (storeGraphObservation && primaryDataIdx >= 0) AcceptNewLink(primaryDataIdx, i, inlierPose, inliers);
				resetLinkRuntimeState(link, true);
				if (emergencyPrimaryFailurePromotion) moveToDataIdx = i;
				else if (shouldMovePrimaryLocalMap(i, moveToDataIdx, primaryDataIdx)) moveToDataIdx = i;
				else if (link.type == LOOP_CLOSURE) link.type = LOST;
				localMapGraphChanged = true;
			}
			else if (success == -1)
			{
				const int primaryLocalMapIdx = primaryDataIdx >= 0 ? activeData[primaryDataIdx].localMapIndex : -1;
				const int rejectedLocalMapIdx = link.localMapIndex;
				const float primaryVisible = primaryDataIdx >= 0 ? visibleOriginalBlocks(primaryDataIdx) : -1.0f;
				const float rejectedVisible = visibleOriginalBlocks(i);
				const int rejectedBlocks = localMapManager->getLocalMapSize(rejectedLocalMapIdx);
				ACTIVE_MAP_CONSOLE(
					"[MULTI ACTIVE] link_reject primary=%d rejected=%d type=%d inliers=%d attempts=%d required=%d trial=%d primary_visible=%.6f rejected_visible=%.6f rejected_blocks=%d\n",
					primaryLocalMapIdx, rejectedLocalMapIdx, static_cast<int>(link.type), inliers, link.trackingAttempts,
					linkOverlap, linkTrials, primaryVisible, rejectedVisible, rejectedBlocks);
				logTransitionEvent("link_reject", primaryLocalMapIdx, rejectedLocalMapIdx,
					sequentialTransformPolicy, "insufficient_overlap", inliers, 0.0f, 0.0f,
					primaryVisible, rejectedVisible, rejectedBlocks, forcePrimaryHandoff, false,
					link.goodTrackingFrames, link.poorTrackingFrames, link.failedTrackingFrames, link.candidateRecoveryAttempts,
					primaryDataIdx >= 0 ? activeData[primaryDataIdx].rawWeakTrackingFrames : -1,
					primaryDataIdx >= 0 ? activeData[primaryDataIdx].rawFailedTrackingFrames : -1);
				if (link.type == NEW_LOCAL_MAP)
				{
						bool promoteForPrimaryFailure =
							canPromoteCandidateAfterPrimaryFailure(i, primaryDataIdx, inliers, rejectedVisible, rejectedBlocks);
						bool preservedGlobal = false;
						if (promoteForPrimaryFailure)
						{
							preservedGlobal =
								preserveCandidateGlobalPoseForHandoff(i, primaryDataIdx, "insufficient_overlap");
							if (primaryFailureHandoffRequirePoseContinuity && !preservedGlobal)
							{
								promoteForPrimaryFailure = false;
								ACTIVE_MAP_CONSOLE(
									"[MULTI ACTIVE] primary_failure_handoff_reject primary=%d candidate=%d reason=insufficient_overlap detail=pose_continuity_unavailable\n",
									primaryLocalMapIdx, rejectedLocalMapIdx);
								logTransitionEvent("primary_failure_handoff_reject", primaryLocalMapIdx, rejectedLocalMapIdx,
									primaryFailureHandoffPolicy, "pose_continuity_unavailable", inliers,
									0.0f, 0.0f, primaryVisible, rejectedVisible, rejectedBlocks,
									forcePrimaryHandoff, false, link.goodTrackingFrames, link.poorTrackingFrames,
									link.failedTrackingFrames, link.candidateRecoveryAttempts,
									activeData[primaryDataIdx].rawWeakTrackingFrames,
									activeData[primaryDataIdx].rawFailedTrackingFrames,
									"insufficient_overlap pose_continuity_unavailable");
							}
						}
						if (promoteForPrimaryFailure)
						{
							ACTIVE_MAP_CONSOLE(
								"[MULTI ACTIVE] primary_failure_handoff primary=%d candidate=%d raw_weak=%d raw_failed=%d candidate_good=%d visible=%.6f blocks=%d inliers=%d reason=insufficient_overlap preserve_global=%d\n",
								primaryLocalMapIdx, rejectedLocalMapIdx, activeData[primaryDataIdx].rawWeakTrackingFrames,
								activeData[primaryDataIdx].rawFailedTrackingFrames,
								link.goodTrackingFrames, rejectedVisible, rejectedBlocks, inliers, preservedGlobal ? 1 : 0);
								logTransitionEvent("primary_failure_handoff", primaryLocalMapIdx, rejectedLocalMapIdx,
									primaryFailureHandoffPolicy, "promote_visible_candidate", inliers,
									0.0f, 0.0f, primaryVisible, rejectedVisible, rejectedBlocks,
									forcePrimaryHandoff, true, link.goodTrackingFrames, link.poorTrackingFrames,
									link.failedTrackingFrames, link.candidateRecoveryAttempts,
									activeData[primaryDataIdx].rawWeakTrackingFrames,
									activeData[primaryDataIdx].rawFailedTrackingFrames,
									std::string("insufficient_overlap trigger=") +
										primaryFailureTriggerReason(activeData[primaryDataIdx]) +
										" preserve_global=" + (preservedGlobal ? "1" : "0"));
						resetLinkRuntimeState(link, true);
						link.lastSequentialTransformEmergency = false;
						moveToDataIdx = i;
						localMapGraphChanged = true;
						continue;
					}
					const bool canRetryCandidate =
						candidateRecoveryPolicy == "keep_if_visible" &&
						candidateRecoveryMaxAttempts > 0 &&
						forcePrimaryHandoff &&
						link.candidateRecoveryAttempts < candidateRecoveryMaxAttempts &&
						link.goodTrackingFrames > 0 &&
						rejectedVisible >= candidateRecoveryMinVisibleRatio &&
						rejectedBlocks >= candidateRecoveryMinBlocks;
					if (canRetryCandidate)
					{
						ACTIVE_MAP_CONSOLE(
							"[MULTI ACTIVE] link_recovery_keep primary=%d candidate=%d attempts=%d/%d good=%d poor=%d failed=%d visible=%.6f blocks=%d min_visible=%.6f min_blocks=%d\n",
							primaryLocalMapIdx, rejectedLocalMapIdx, link.candidateRecoveryAttempts + 1, candidateRecoveryMaxAttempts,
							link.goodTrackingFrames, link.poorTrackingFrames, link.failedTrackingFrames, rejectedVisible, rejectedBlocks,
							candidateRecoveryMinVisibleRatio, candidateRecoveryMinBlocks);
							logTransitionEvent("link_recovery_keep", primaryLocalMapIdx, rejectedLocalMapIdx,
								candidateRecoveryPolicy, "retry_candidate", inliers, 0.0f, 0.0f,
								primaryVisible, rejectedVisible, rejectedBlocks, forcePrimaryHandoff, false,
								link.goodTrackingFrames, link.poorTrackingFrames, link.failedTrackingFrames, link.candidateRecoveryAttempts + 1,
								primaryDataIdx >= 0 ? activeData[primaryDataIdx].rawWeakTrackingFrames : -1,
								primaryDataIdx >= 0 ? activeData[primaryDataIdx].rawFailedTrackingFrames : -1,
								"retry_candidate");
							resetLinkRuntimeState(link, false);
							link.lastSequentialTransformAccepted = false;
							link.lastSequentialTransformEmergency = false;
							link.invalidSequentialTransformAttempts = 0;
							link.acceptedSequentialWindows = 0;
							link.candidateRecoveryAttempts++;
						localMapGraphChanged = true;
						continue;
					}
					link.type = LOST_NEW;
					link.lastSequentialTransformEmergency = false;
					forcePrimaryHandoff = false;
				}
				else link.type = LOST;
				localMapGraphChanged = true;
			}
			else if (success == 0 && link.type == NEW_LOCAL_MAP)
			{
				const int primaryLocalMapIdx = primaryDataIdx >= 0 ? activeData[primaryDataIdx].localMapIndex : -1;
				const int candidateLocalMapIdx = link.localMapIndex;
				const float primaryVisible = primaryDataIdx >= 0 ? visibleOriginalBlocks(primaryDataIdx) : -1.0f;
				const float candidateVisible = visibleOriginalBlocks(i);
				const int candidateBlocks = localMapManager->getLocalMapSize(candidateLocalMapIdx);
				bool promoteForPrimaryFailure =
					canPromoteCandidateAfterPrimaryFailure(i, primaryDataIdx, inliers, candidateVisible, candidateBlocks);
				bool preservedGlobal = false;
				if (promoteForPrimaryFailure)
				{
					preservedGlobal = preserveCandidateGlobalPoseForHandoff(i, primaryDataIdx, "trial_window");
					if (primaryFailureHandoffRequirePoseContinuity && !preservedGlobal)
					{
						promoteForPrimaryFailure = false;
						ACTIVE_MAP_CONSOLE(
							"[MULTI ACTIVE] primary_failure_handoff_reject primary=%d candidate=%d reason=trial_window detail=pose_continuity_unavailable\n",
							primaryLocalMapIdx, candidateLocalMapIdx);
						logTransitionEvent("primary_failure_handoff_reject", primaryLocalMapIdx, candidateLocalMapIdx,
							primaryFailureHandoffPolicy, "pose_continuity_unavailable", inliers,
							0.0f, 0.0f, primaryVisible, candidateVisible, candidateBlocks,
							forcePrimaryHandoff, false, link.goodTrackingFrames, link.poorTrackingFrames,
							link.failedTrackingFrames, link.candidateRecoveryAttempts,
							activeData[primaryDataIdx].rawWeakTrackingFrames,
							activeData[primaryDataIdx].rawFailedTrackingFrames,
							"trial_window pose_continuity_unavailable");
					}
				}
				if (promoteForPrimaryFailure)
				{
						ACTIVE_MAP_CONSOLE(
							"[MULTI ACTIVE] primary_failure_handoff primary=%d candidate=%d raw_weak=%d raw_failed=%d candidate_good=%d visible=%.6f blocks=%d inliers=%d reason=trial_window preserve_global=%d\n",
							primaryLocalMapIdx, candidateLocalMapIdx, activeData[primaryDataIdx].rawWeakTrackingFrames,
							activeData[primaryDataIdx].rawFailedTrackingFrames,
							link.goodTrackingFrames, candidateVisible, candidateBlocks, inliers, preservedGlobal ? 1 : 0);
						logTransitionEvent("primary_failure_handoff", primaryLocalMapIdx, candidateLocalMapIdx,
							primaryFailureHandoffPolicy, "promote_visible_candidate", inliers,
							0.0f, 0.0f, primaryVisible, candidateVisible, candidateBlocks,
							forcePrimaryHandoff, true, link.goodTrackingFrames, link.poorTrackingFrames,
							link.failedTrackingFrames, link.candidateRecoveryAttempts,
							activeData[primaryDataIdx].rawWeakTrackingFrames,
							activeData[primaryDataIdx].rawFailedTrackingFrames,
							std::string("trial_window trigger=") +
								primaryFailureTriggerReason(activeData[primaryDataIdx]) +
								" preserve_global=" + (preservedGlobal ? "1" : "0"));
					resetLinkRuntimeState(link, true);
					link.lastSequentialTransformEmergency = false;
					moveToDataIdx = i;
					localMapGraphChanged = true;
				}
			}
		}
	}

	std::vector<int> restartLinksToLocalMaps;
	primaryDataIdx = -1;
	for (int i = 0; i < (int)activeData.size(); ++i)
	{
		ActiveDataDescriptor & link = activeData[i];

		if ((signed)i == moveToDataIdx) link.type = PRIMARY_LOCAL_MAP;

		if ((link.type == PRIMARY_LOCAL_MAP) && (moveToDataIdx >= 0) && ((signed)i != moveToDataIdx)) 
		{
			link.type = LOST;
			restartLinksToLocalMaps.push_back(link.localMapIndex);
		}
		
		if ((link.type == NEW_LOCAL_MAP) && (moveToDataIdx >= 0))
		{
			link.type = LOST_NEW;
			link.lastSequentialTransformEmergency = false;
		}
		
		if ((link.type == LOOP_CLOSURE) && (moveToDataIdx >= 0)) 
		{
			link.type = LOST;
			restartLinksToLocalMaps.push_back(link.localMapIndex);
		}

		if ((link.type == RELOCALISATION) && (moveToDataIdx >= 0)) link.type = LOST;

		if (link.type == PRIMARY_LOCAL_MAP)
		{
			if (primaryDataIdx >= 0) fprintf(stderr, "OOOPS, two or more primary localMaps...\n");
			primaryDataIdx = i;
		}
	}

	for (size_t i = 0; i < activeData.size(); )
	{
		ActiveDataDescriptor & link = activeData[i];
		if (link.type == LOST_NEW)
		{
			// NOTE: there will only be at most one new local map at
			// any given time and it's guaranteed to be the last
			// in the list. Removing this new local map will therefore
			// not require rearranging indices!
			localMapManager->removeLocalMap(link.localMapIndex);
			link.type = LOST;
		}
		if (link.type == LOST) activeData.erase(activeData.begin() + i);
		else i++;
	}

	primaryDataIdx = findPrimaryDataIdx();
	if (moveToDataIdx >= 0) forcePrimaryHandoff = false;
	if (primaryDataIdx < 0)
	{
		if (!activeData.empty())
		{
			const int fallbackIdx = static_cast<int>(activeData.size()) - 1;
			activeData[fallbackIdx].type = PRIMARY_LOCAL_MAP;
				activeData[fallbackIdx].trackingAttempts = 0;
				forceNewArea = false;
				forcePrimaryHandoff = false;
				localMapGraphChanged = true;
			fprintf(stderr, "[MULTI ACTIVE] restored primary active data localMap=%d\n", activeData[fallbackIdx].localMapIndex);
			primaryDataIdx = fallbackIdx;
		}
		else if (localMapManager->numLocalMaps() > 0)
		{
			const int fallbackLocalMap = static_cast<int>(localMapManager->numLocalMaps()) - 1;
			resetToPrimaryMap(fallbackLocalMap);
			forceNewArea = false;
			localMapGraphChanged = true;
			fprintf(stderr, "[MULTI ACTIVE] restored primary from existing local map=%d\n", fallbackLocalMap);
			primaryDataIdx = findPrimaryDataIdx();
		}
	}
	if (moveToDataIdx >= 0) forceNewArea = false;

	if (allowHistoricalLinks)
		for (std::vector<int>::const_iterator it = restartLinksToLocalMaps.begin(); it != restartLinksToLocalMaps.end(); ++it)
			initiateNewLink(*it, *(localMapManager->getTrackingPose(*it)), false);

	// NOTE: this has to be done AFTER removing any previous new local map
	if (shouldStartNewArea())
	{
		int newIdx = initiateNewLocalMap();
		forceNewArea = false;

		if (primaryDataIdx >= 0)
		{
			int primaryLocalMapIdx = activeData[primaryDataIdx].localMapIndex;
			localMapManager->setEstimatedGlobalPose(newIdx, ORUtils::SE3Pose(localMapManager->getTrackingPose(primaryLocalMapIdx)->GetM() * localMapManager->getEstimatedGlobalPose(primaryLocalMapIdx).GetM()));
		}
	}

	return localMapGraphChanged;
}

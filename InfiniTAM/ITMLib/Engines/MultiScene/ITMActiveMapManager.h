// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMMapGraphManager.h"
#include <fstream>
#include <string>
#include <vector>

namespace ITMLib
{
	/** \brief
	*/
	class ITMActiveMapManager
	{
	public:
		typedef enum { PRIMARY_LOCAL_MAP, NEW_LOCAL_MAP, LOOP_CLOSURE, RELOCALISATION, LOST, LOST_NEW } LocalMapActivity;

		struct TransitionEvent
		{
			int frame = -1;
			std::string event;
			int primaryLocalMap = -1;
			int newLocalMap = -1;
			std::string policy;
			std::string action;
			int inliers = -1;
			float deltaM = 0.0f;
			float deltaDeg = 0.0f;
			float primaryVisible = -1.0f;
			float newVisible = -1.0f;
			int newBlocks = -1;
			bool forceHandoff = false;
			bool moveToPrimary = false;
			int candidateGoodFrames = -1;
			int candidatePoorFrames = -1;
			int candidateFailedFrames = -1;
			int candidateRecoveryAttempts = -1;
			int primaryRawWeakFrames = -1;
			int primaryRawFailedFrames = -1;
			std::string detail;
		};

	private:
		struct ActiveDataDescriptor 
		{
			int localMapIndex;
			LocalMapActivity type;
			std::vector<Matrix4f> constraints;
			ORUtils::SE3Pose estimatedPose;
			int trackingAttempts;
			bool lastSequentialTransformAccepted;
				int invalidSequentialTransformAttempts;
				int candidateRecoveryAttempts;
				int acceptedSequentialWindows;
				bool lastSequentialTransformEmergency;
				int goodTrackingFrames;
				int poorTrackingFrames;
				int failedTrackingFrames;
			ITMTrackingState::TrackingResult lastTrackingResult;
			int rawWeakTrackingFrames;
			int rawFailedTrackingFrames;
			ITMTrackingState::TrackingResult lastRawTrackingResult;
		};

		ITMMapGraphManager *localMapManager;
		std::vector<ActiveDataDescriptor> activeData;
		int linkTrials;
		int linkOverlap;
		int emergencyLinkOverlap;
		int originalBlocks;
		float originalBlocksThreshold;
		bool allowNewLocalMaps;
		bool allowHistoricalLinks;
		bool allowVisibilityHandoff;
		bool consoleDiagnostics;
		mutable bool forceNewArea;
		bool forcePrimaryHandoff;
		int diagnosticFrame;
		mutable std::ofstream transitionLog;
		mutable std::vector<TransitionEvent> recentTransitionEvents;
		std::string sequentialTransformPolicy;
		float sequentialTransformMaxTranslation;
		float sequentialTransformMaxRotationDeg;
		int invalidTransformRestartAttempts;
		std::string handoffPosePolicy;
			std::string newMapHandoffPolicy;
			float newMapHandoffMinVisibleRatio;
			float newMapHandoffMinVisibleMargin;
			int newMapHandoffMinAcceptedWindows;
			int newMapHandoffEmergencyMinAcceptedWindows;
			std::string candidateRecoveryPolicy;
			float candidateRecoveryMinVisibleRatio;
			int candidateRecoveryMinBlocks;
		int candidateRecoveryMaxAttempts;
			std::string primaryFailureHandoffPolicy;
			std::string primaryFailureHandoffRawGate;
			int primaryFailureHandoffMinRawFailedFrames;
			int emergencyLinkMinRawWeakFrames;
			float primaryFailureHandoffMinCandidateVisibleRatio;
			int primaryFailureHandoffMinCandidateBlocks;
			int primaryFailureHandoffMinCandidateGoodFrames;
			int primaryFailureHandoffMinOverlapFrames;
			bool primaryFailureHandoffRequirePoseContinuity;

		void initializeActiveDescriptor(ActiveDataDescriptor &descriptor, int localMapIndex, LocalMapActivity type) const;
		int CheckSuccess_relocalisation(int dataID) const;
		int CheckSuccess_newlink(int dataID, int primaryDataID, int *inliers, ORUtils::SE3Pose *inlierPose) const;
		void AcceptNewLink(int dataId, int primaryDataId, const ORUtils::SE3Pose & pose, int weight);

			float visibleOriginalBlocks(int dataID) const;
				int handoffMaturityMinAcceptedWindows(int dataID, bool allowEmergencyOverride) const;
				bool handoffMaturitySatisfied(int dataID, bool allowEmergencyOverride = false) const;
				bool handoffVisibilitySatisfied(float primaryVisible, float newVisible) const;
				std::string primaryFailureTriggerReason(const ActiveDataDescriptor &primary) const;
				bool canPromoteCandidateAfterPrimaryFailure(int candidateDataIdx, int primaryDataIdx,
				int inliers, float candidateVisible, int candidateBlocks) const;
				bool preserveCandidateGlobalPoseForHandoff(int candidateDataIdx, int primaryDataIdx,
				const std::string &reason);
			bool shouldStartNewArea(void) const;
		bool shouldMovePrimaryLocalMap(int newDataIdx, int bestDataIdx, int primaryDataIdx) const;
		void logTransitionEvent(const std::string &event, int primaryLocalMap, int newLocalMap,
			const std::string &policy, const std::string &action, int inliers,
			float deltaM, float deltaDeg, float primaryVisible, float newVisible,
			int newBlocks, bool forceHandoff, bool moveToPrimary,
			int candidateGoodFrames = -1, int candidatePoorFrames = -1,
			int candidateFailedFrames = -1, int candidateRecoveryAttempts = -1,
			int primaryRawWeakFrames = -1, int primaryRawFailedFrames = -1,
			const std::string &detail = "") const;

	public:
		int initiateNewLocalMap(bool isPrimaryLocalMap = false);
		bool canInitiateNewLink(int sceneID) const;
		int initiateNewLink(int sceneID, const ORUtils::SE3Pose & pose, bool isRelocalisation);

		void recordTrackingResult(int dataID, ITMTrackingState::TrackingResult trackingResult, bool primaryTrackingSuccess);
		void recordRawTrackingResult(int dataID, ITMTrackingState::TrackingResult rawTrackingResult);

		// return whether or not the local map graph has changed
		bool maintainActiveData(int frameId = -1);

		int findPrimaryDataIdx(void) const;
		int findPrimaryLocalMapIdx(void) const;

		int findBestVisualisationDataIdx(void) const;
		int findBestVisualisationLocalMapIdx(void) const;

		int numActiveLocalMaps(void) const { return static_cast<int>(activeData.size()); }
		int getLocalMapIndex(int dataIdx) const { return activeData[dataIdx].localMapIndex; }
		LocalMapActivity getLocalMapType(int dataIdx) const { return activeData[dataIdx].type; }
		bool hasActiveNewLocalMap(void) const;
		bool isPrimaryHandoffRequested(void) const { return forcePrimaryHandoff; }
		std::vector<int> getActiveLocalMapIndices(void) const;
		std::vector<TransitionEvent> consumeRecentTransitionEvents(void) const;
		void configureSequentialPolicy(bool allowNewMaps, bool allowOldLinks, float visibleRatioThreshold, int overlapFrames, int trialFrames, int emergencyOverlapFrames, int sampledOriginalBlocks,
				const std::string &transformPolicy = "validated_overlap", float maxTransformTranslation = 0.03f, float maxTransformRotationDeg = 3.0f,
				int invalidTransformRestartAttempts = 0, const std::string &handoffPolicy = "candidate", const std::string &newMapHandoffPolicy = "visibility",
				float newMapHandoffMinVisibleRatio = 0.0f, float newMapHandoffMinVisibleMargin = 0.0f,
				int newMapHandoffMinAcceptedWindows = 1, int newMapHandoffEmergencyMinAcceptedWindows = -1,
				const std::string &candidateRecoveryPolicy = "disabled", float candidateRecoveryMinVisibleRatio = 0.3f,
				int candidateRecoveryMinBlocks = 1000, int candidateRecoveryMaxAttempts = 2,
				const std::string &primaryFailureHandoffPolicy = "disabled", int primaryFailureHandoffMinRawFailedFrames = 30,
				const std::string &primaryFailureHandoffRawGate = "failed_only", int primaryFailureHandoffMinRawWeakFrames = 0,
				float primaryFailureHandoffMinCandidateVisibleRatio = 0.5f, int primaryFailureHandoffMinCandidateBlocks = 1000,
				int primaryFailureHandoffMinCandidateGoodFrames = 10, int primaryFailureHandoffMinOverlapFrames = 0,
				bool primaryFailureHandoffRequirePoseContinuity = false,
				bool allowVisibilityHandoff = true);
		void requestNewArea(void) { if (allowNewLocalMaps) forceNewArea = true; }
		void requestPrimaryHandoff(void) { forcePrimaryHandoff = true; }
		bool retireNewLocalMap(int localMapIndex, const std::string &reason);
		void resetToPrimaryMap(int localMapIndex);
		void configureDiagnostics(const std::string &directory);
		void setConsoleDiagnostics(bool enabled) { consoleDiagnostics = enabled; }

		ITMActiveMapManager(ITMMapGraphManager *localMapManager);
		~ITMActiveMapManager(void) {}
	};
}

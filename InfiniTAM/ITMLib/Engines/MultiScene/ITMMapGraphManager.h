// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <cstdint>
#include <vector>

#include "../../Objects/Scene/ITMLocalMap.h"
#include "../../Core/ITMDenseMapper.h"
#include "../../Engines/Visualisation/Interface/ITMVisualisationEngine.h"

namespace ITMLib
{
	/* This helpful abstract interface allows you to ignore the fact that
	scenes are templates.
	*/
	class ITMMapGraphManager
	{
	public:
		virtual ~ITMMapGraphManager(void) {}

		virtual int createNewLocalMap(void) = 0;
		virtual void removeLocalMap(int index) = 0;
		virtual size_t numLocalMaps(void) const = 0;

		virtual const ITMPoseConstraint & getRelation_const(int fromLocalMap, int toLocalMap) const = 0;
		virtual ITMPoseConstraint & getRelation(int fromLocalMap, int toLocalMap) = 0;
		virtual void eraseRelation(int fromLocalMap, int toLocalMap) = 0;
		virtual const ConstraintList & getConstraints(int localMapId) const = 0;

		virtual void setEstimatedGlobalPose(int localMapId, const ORUtils::SE3Pose & pose) = 0;
		virtual const ORUtils::SE3Pose & getEstimatedGlobalPose(int localMapId) const = 0;

		virtual bool resetTracking(int localMapId, const ORUtils::SE3Pose & pose) = 0;

		virtual const ORUtils::SE3Pose* getTrackingPose(int localMapId) const = 0;
		virtual int getLocalMapSize(int localMapId) const = 0;
		virtual bool getLocalMapBlockBounds(int localMapId, Vector3f &localMin, Vector3f &localMax) const = 0;
		virtual int countVisibleBlocks(int localMapId, int minBlockId, int maxBlockId, bool invertIDs) const = 0;
	};

	template<class TVoxel, class TIndex>
	class ITMVoxelMapGraphManager : public ITMMapGraphManager
	{
	private:
		const ITMLibSettings *settings;
		const ITMVisualisationEngine<TVoxel, TIndex> *visualisationEngine;
		const ITMDenseMapper<TVoxel, TIndex> *denseMapper;
		Vector2i trackedImageSize;

		std::vector<ITMLocalMap<TVoxel, TIndex>*> allData;

	public:
		ITMVoxelMapGraphManager(const ITMLibSettings *settings, const ITMVisualisationEngine<TVoxel, TIndex> *visualisationEngine, const ITMDenseMapper<TVoxel, TIndex> *denseMapper, const Vector2i & trackedImageSize);
		~ITMVoxelMapGraphManager(void);

		int createNewLocalMap(void);
		void removeLocalMap(int index);
		size_t numLocalMaps(void) const { return allData.size(); }

		const ITMLocalMap<TVoxel, TIndex>* getLocalMap(int localMapId) const { return allData[localMapId]; }

		ITMLocalMap<TVoxel, TIndex>* getLocalMap(int localMapId) { return allData[localMapId]; }

		const ITMPoseConstraint & getRelation_const(int fromLocalMap, int toLocalMap) const;
		ITMPoseConstraint & getRelation(int fromLocalMap, int toLocalMap);
		void eraseRelation(int fromLocalMap, int toLocalMap);
		const ConstraintList & getConstraints(int localMapId) const { return allData[localMapId]->relations; }

		void setEstimatedGlobalPose(int localMapId, const ORUtils::SE3Pose & pose) { allData[localMapId]->estimatedGlobalPose = pose; }
		const ORUtils::SE3Pose & getEstimatedGlobalPose(int localMapId) const { return allData[localMapId]->estimatedGlobalPose; }

		bool resetTracking(int localMapId, const ORUtils::SE3Pose & pose);
		const ORUtils::SE3Pose* getTrackingPose(int localMapId) const { return getLocalMap(localMapId)->trackingState->pose_d; }

		int getLocalMapSize(int localMapId) const;
		bool getLocalMapBlockBounds(int localMapId, Vector3f &localMin, Vector3f &localMax) const;
		int countVisibleBlocks(int localMapId, int minBlockId, int maxBlockId, bool invertIDs) const;
		bool isLocalMapResident(int localMapId) const;
		int countResidentLocalMaps(void) const;
		int countPendingLocalMapSnapshots(void) const;
		int countPendingLocalMapPageIns(void) const;
		int64_t getResidentLocalMapEstimatedBytes(void) const;
		int64_t getPendingLocalMapSnapshotEstimatedBytes(void) const;
		int64_t getPendingLocalMapPageInEstimatedBytes(void) const;
		std::vector<int> getResidentLocalMapIds(void) const;
		void pageOutLocalMap(int localMapId, const std::string &directory, bool asyncSnapshot = false, bool keepCpuSnapshot = false);
		void pageInLocalMap(int localMapId);
		void prefetchLocalMap(int localMapId);
		bool tryCompleteLocalMapPageIn(int localMapId);
		const std::string &getLocalMapPageDirectory(int localMapId) const;
		void waitForLocalMapSnapshot(int localMapId);
		void waitForLocalMapPageIn(int localMapId);
		void adoptLocalMapPageSnapshot(int localMapId, const std::string &directory, int blockCount);

		ORUtils::SE3Pose findTransformation(int fromlocalMapId, int tolocalMapId) const;
	};
}

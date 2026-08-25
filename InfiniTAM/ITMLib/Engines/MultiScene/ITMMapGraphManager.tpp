// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMMapGraphManager.h"

//#include <queue>

namespace ITMLib
{
	template<class TVoxel, class TIndex>
	ITMVoxelMapGraphManager<TVoxel, TIndex>::ITMVoxelMapGraphManager(const ITMLibSettings *_settings, const ITMVisualisationEngine<TVoxel, TIndex> *_visualisationEngine, const ITMDenseMapper<TVoxel, TIndex> *_denseMapper, const Vector2i & _trackedImageSize)
		: settings(_settings), visualisationEngine(_visualisationEngine), denseMapper(_denseMapper), trackedImageSize(_trackedImageSize)
	{
	}

	template<class TVoxel, class TIndex>
	ITMVoxelMapGraphManager<TVoxel, TIndex>::~ITMVoxelMapGraphManager(void)
	{
		while (allData.size() > 0)
		{
			delete allData.back();
			allData.pop_back();
		}
	}

	template<class TVoxel, class TIndex>
	int ITMVoxelMapGraphManager<TVoxel, TIndex>::createNewLocalMap(void)
	{
		int newIdx = (int)allData.size();
		allData.push_back(new ITMLocalMap<TVoxel, TIndex>(settings, visualisationEngine, trackedImageSize));

		denseMapper->ResetScene(allData[newIdx]->scene);
		return newIdx;
	}

	template<class TVoxel, class TIndex>
	void ITMVoxelMapGraphManager<TVoxel, TIndex>::removeLocalMap(int localMapId)
	{
		if ((localMapId < 0) || ((unsigned)localMapId >= allData.size())) return;

		// make sure there are no relations anywhere pointing to the local map
		const ConstraintList & l = getConstraints(localMapId);
		for (ConstraintList::const_iterator it = l.begin(); it != l.end(); ++it) eraseRelation(it->first, localMapId);

		// delete the local map
		delete allData[localMapId];
		allData.erase(allData.begin() + localMapId);
	}

	template<class TVoxel, class TIndex>
	ITMPoseConstraint & ITMVoxelMapGraphManager<TVoxel, TIndex>::getRelation(int fromLocalMap, int toLocalMap)
	{
		ConstraintList & m = getLocalMap(fromLocalMap)->relations;
		return m[toLocalMap];
	}

	static const ITMPoseConstraint invalidPoseConstraint;

	template<class TVoxel, class TIndex>
	const ITMPoseConstraint & ITMVoxelMapGraphManager<TVoxel, TIndex>::getRelation_const(int fromLocalMap, int toLocalMap) const
	{
		if ((fromLocalMap < 0) || (fromLocalMap >= (int)allData.size())) return invalidPoseConstraint;

		const ConstraintList & m = getLocalMap(fromLocalMap)->relations;
		ConstraintList::const_iterator it = m.find(toLocalMap);
		if (it == m.end()) return invalidPoseConstraint;

		return it->second;
	}

	template<class TVoxel, class TIndex>
	void ITMVoxelMapGraphManager<TVoxel, TIndex>::eraseRelation(int fromLocalMap, int toLocalMap)
	{
		if ((fromLocalMap < 0) || (fromLocalMap >= (int)allData.size())) return;

		std::map<int, ITMPoseConstraint> & m = getLocalMap(fromLocalMap)->relations;
		m.erase(toLocalMap);
	}

	template<class TVoxel, class TIndex>
	bool ITMVoxelMapGraphManager<TVoxel, TIndex>::resetTracking(int localMapId, const ORUtils::SE3Pose & pose)
	{
		if ((localMapId < 0) || ((unsigned)localMapId >= allData.size())) return false;
		allData[localMapId]->trackingState->pose_d->SetFrom(&pose);
		allData[localMapId]->trackingState->age_pointCloud = -1;
		return true;
	}

	template<class TVoxel, class TIndex>
	int ITMVoxelMapGraphManager<TVoxel, TIndex>::getLocalMapSize(int localMapId) const
	{
		if ((localMapId < 0) || ((unsigned)localMapId >= allData.size())) return -1;

		ITMScene<TVoxel, TIndex> *scene = allData[localMapId]->scene;
		if (scene == NULL) return allData[localMapId]->cachedBlockCount;
		return scene->index.getNumAllocatedVoxelBlocks() - scene->localVBA.lastFreeBlockId - 1;
	}

	template<class TVoxel, class TIndex>
	bool ITMVoxelMapGraphManager<TVoxel, TIndex>::getLocalMapBlockBounds(int localMapId, Vector3f &localMin, Vector3f &localMax) const
	{
		if ((localMapId < 0) || ((unsigned)localMapId >= allData.size())) return false;
		ITMLocalMap<TVoxel, TIndex> *localMap = allData[localMapId];
		if (localMap->scene == NULL)
		{
			if (!localMap->cachedBlockBoundsValid) return false;
			localMin = localMap->cachedBlockBoundsMin;
			localMax = localMap->cachedBlockBoundsMax;
			return true;
		}

		TIndex cpuIndex(MEMORYDEVICE_CPU);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
		{
			cpuIndex.CopyFrom(localMap->scene->index,
				ORUtils::MemoryBlock<ITMHashEntry>::CUDA_TO_CPU,
				ORUtils::MemoryBlock<int>::CUDA_TO_CPU);
		}
		else
		{
			cpuIndex.CopyFrom(localMap->scene->index,
				ORUtils::MemoryBlock<ITMHashEntry>::CPU_TO_CPU,
				ORUtils::MemoryBlock<int>::CPU_TO_CPU);
		}

		const typename TIndex::IndexData *entries = cpuIndex.GetEntries();
		const float blockScale = settings->sceneParams.voxelSize * float(SDF_BLOCK_SIZE);
		bool found = false;
		Vector3f minBounds(0.0f), maxBounds(0.0f);
		for (int entryIdx = 0; entryIdx < TIndex::noTotalEntries; ++entryIdx)
		{
			const int ptr = entries[entryIdx].ptr;
			if (ptr < 0) continue;
			const Vector3s &pos = entries[entryIdx].pos;
			const Vector3f blockMin(float(pos.x) * blockScale,
				float(pos.y) * blockScale,
				float(pos.z) * blockScale);
			const Vector3f blockMax(float(pos.x + 1) * blockScale,
				float(pos.y + 1) * blockScale,
				float(pos.z + 1) * blockScale);
			if (!found)
			{
				minBounds = blockMin;
				maxBounds = blockMax;
				found = true;
			}
			else
			{
				minBounds.x = MIN(minBounds.x, blockMin.x);
				minBounds.y = MIN(minBounds.y, blockMin.y);
				minBounds.z = MIN(minBounds.z, blockMin.z);
				maxBounds.x = MAX(maxBounds.x, blockMax.x);
				maxBounds.y = MAX(maxBounds.y, blockMax.y);
				maxBounds.z = MAX(maxBounds.z, blockMax.z);
			}
		}
		if (!found)
		{
			localMap->cachedBlockBoundsValid = false;
			return false;
		}
		localMap->cachedBlockBoundsValid = true;
		localMap->cachedBlockBoundsMin = minBounds;
		localMap->cachedBlockBoundsMax = maxBounds;
		localMin = minBounds;
		localMax = maxBounds;
		return true;
	}

	template<class TVoxel, class TIndex>
	bool ITMVoxelMapGraphManager<TVoxel, TIndex>::isLocalMapResident(int localMapId) const
	{
		return localMapId >= 0 && (unsigned)localMapId < allData.size() && allData[localMapId]->isResident();
	}

	template<class TVoxel, class TIndex>
	int ITMVoxelMapGraphManager<TVoxel, TIndex>::countResidentLocalMaps(void) const
	{
		int result = 0;
		for (size_t i = 0; i < allData.size(); ++i) if (allData[i]->isResident()) ++result;
		return result;
	}

	template<class TVoxel, class TIndex>
	int ITMVoxelMapGraphManager<TVoxel, TIndex>::countPendingLocalMapSnapshots(void) const
	{
		int result = 0;
		for (size_t i = 0; i < allData.size(); ++i) if (allData[i]->hasPendingSnapshot()) ++result;
		return result;
	}

	template<class TVoxel, class TIndex>
	int ITMVoxelMapGraphManager<TVoxel, TIndex>::countPendingLocalMapPageIns(void) const
	{
		int result = 0;
		for (size_t i = 0; i < allData.size(); ++i) if (allData[i]->hasPendingPageIn()) ++result;
		return result;
	}

	template<class TVoxel, class TIndex>
	int64_t ITMVoxelMapGraphManager<TVoxel, TIndex>::getResidentLocalMapEstimatedBytes(void) const
	{
		int64_t result = 0;
		for (size_t i = 0; i < allData.size(); ++i)
			if (allData[i]->isResident()) result += allData[i]->estimatedSceneBytes();
		return result;
	}

	template<class TVoxel, class TIndex>
	int64_t ITMVoxelMapGraphManager<TVoxel, TIndex>::getPendingLocalMapSnapshotEstimatedBytes(void) const
	{
		int64_t result = 0;
		for (size_t i = 0; i < allData.size(); ++i)
			if (allData[i]->hasPendingSnapshot()) result += allData[i]->estimatedSceneBytes();
		return result;
	}

	template<class TVoxel, class TIndex>
	int64_t ITMVoxelMapGraphManager<TVoxel, TIndex>::getPendingLocalMapPageInEstimatedBytes(void) const
	{
		int64_t result = 0;
		for (size_t i = 0; i < allData.size(); ++i)
			if (allData[i]->hasPendingPageIn()) result += allData[i]->estimatedSceneBytes();
		return result;
	}

	template<class TVoxel, class TIndex>
	std::vector<int> ITMVoxelMapGraphManager<TVoxel, TIndex>::getResidentLocalMapIds(void) const
	{
		std::vector<int> result;
		for (size_t i = 0; i < allData.size(); ++i) if (allData[i]->isResident()) result.push_back((int)i);
		return result;
	}

	template<class TVoxel, class TIndex>
	void ITMVoxelMapGraphManager<TVoxel, TIndex>::pageOutLocalMap(int localMapId, const std::string &directory, bool asyncSnapshot, bool keepCpuSnapshot)
	{
		if (localMapId < 0 || (unsigned)localMapId >= allData.size()) throw std::runtime_error("Invalid local map page-out ID");
		Vector3f localMin, localMax;
		getLocalMapBlockBounds(localMapId, localMin, localMax);
		allData[localMapId]->pageOut(directory, asyncSnapshot, keepCpuSnapshot);
	}

	template<class TVoxel, class TIndex>
	void ITMVoxelMapGraphManager<TVoxel, TIndex>::pageInLocalMap(int localMapId)
	{
		if (localMapId < 0 || (unsigned)localMapId >= allData.size()) throw std::runtime_error("Invalid local map page-in ID");
		allData[localMapId]->pageIn();
	}

	template<class TVoxel, class TIndex>
	void ITMVoxelMapGraphManager<TVoxel, TIndex>::prefetchLocalMap(int localMapId)
	{
		if (localMapId < 0 || (unsigned)localMapId >= allData.size()) throw std::runtime_error("Invalid local map prefetch ID");
		allData[localMapId]->beginPageInAsync();
	}

	template<class TVoxel, class TIndex>
	bool ITMVoxelMapGraphManager<TVoxel, TIndex>::tryCompleteLocalMapPageIn(int localMapId)
	{
		if (localMapId < 0 || (unsigned)localMapId >= allData.size()) throw std::runtime_error("Invalid local map page-in completion ID");
		return allData[localMapId]->tryCompletePageIn();
	}

	template<class TVoxel, class TIndex>
	const std::string &ITMVoxelMapGraphManager<TVoxel, TIndex>::getLocalMapPageDirectory(int localMapId) const
	{
		if (localMapId < 0 || (unsigned)localMapId >= allData.size()) throw std::runtime_error("Invalid local map page-directory ID");
		return allData[localMapId]->pageDirectory;
	}

	template<class TVoxel, class TIndex>
	void ITMVoxelMapGraphManager<TVoxel, TIndex>::waitForLocalMapSnapshot(int localMapId)
	{
		if (localMapId < 0 || (unsigned)localMapId >= allData.size()) throw std::runtime_error("Invalid local map snapshot ID");
		allData[localMapId]->waitForSnapshot();
	}

	template<class TVoxel, class TIndex>
	void ITMVoxelMapGraphManager<TVoxel, TIndex>::waitForLocalMapPageIn(int localMapId)
	{
		if (localMapId < 0 || (unsigned)localMapId >= allData.size()) throw std::runtime_error("Invalid local map page-in wait ID");
		allData[localMapId]->waitForPageIn();
	}

	template<class TVoxel, class TIndex>
	void ITMVoxelMapGraphManager<TVoxel, TIndex>::adoptLocalMapPageSnapshot(int localMapId, const std::string &directory, int blockCount)
	{
		if (localMapId < 0 || (unsigned)localMapId >= allData.size()) throw std::runtime_error("Invalid local map snapshot ID");
		allData[localMapId]->adoptPageSnapshot(directory, blockCount);
	}

	template<class TVoxel, class TIndex>
	int ITMVoxelMapGraphManager<TVoxel, TIndex>::countVisibleBlocks(int localMapId, int minBlockId, int maxBlockId, bool invertIds) const
	{
		if ((localMapId < 0) || ((unsigned)localMapId >= allData.size())) return -1;
		const ITMLocalMap<TVoxel, TIndex> *localMap = allData[localMapId];
		if (!localMap->isResident()) return 0;

		if (invertIds) 
		{
			int tmp = minBlockId;
			minBlockId = localMap->scene->index.getNumAllocatedVoxelBlocks() - maxBlockId - 1;
			maxBlockId = localMap->scene->index.getNumAllocatedVoxelBlocks() - tmp - 1;
		}

		return visualisationEngine->CountVisibleBlocks(localMap->scene, localMap->renderState, minBlockId, maxBlockId);
	}

	struct LinkPathComparison 
	{
		bool operator()(const std::vector<int> & a, const std::vector<int> & b) { return a.size() > b.size(); }
	};

	template<class TVoxel, class TIndex>
	ORUtils::SE3Pose ITMVoxelMapGraphManager<TVoxel, TIndex>::findTransformation(int fromLocalMapId, int toLocalMapId) const
	{
		ORUtils::SE3Pose fromLocalMapPose, toLocalMapPose;
		if ((fromLocalMapId >= 0) && ((size_t)fromLocalMapId < allData.size())) fromLocalMapPose = allData[fromLocalMapId]->estimatedGlobalPose;
		if ((toLocalMapId >= 0) && ((size_t)toLocalMapId < allData.size())) toLocalMapPose = allData[toLocalMapId]->estimatedGlobalPose;
		return ORUtils::SE3Pose(toLocalMapPose.GetM() * fromLocalMapPose.GetInvM());
	}
}

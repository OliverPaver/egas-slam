// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <filesystem>
#include <vector>

#include "ITMLocalVBA.h"
#include "ITMGlobalCache.h"
#include "../../Utils/ITMSceneParams.h"

namespace ITMLib
{
	/** \brief
	Represents the 3D world model as a hash of small voxel
	blocks
	*/
	template <class TVoxel, class TIndex>
	class ITMScene
	{
	public:
		/** Scene parameters like voxel size etc. */
		const ITMSceneParams *sceneParams;

		/** Hash table to reference the 8x8x8 blocks */
		TIndex index;

		/** Current local content of the 8x8x8 voxel blocks -- stored host or device */
		ITMLocalVBA<TVoxel> localVBA;

		/** Global content of the 8x8x8 voxel blocks -- stored on host only */
		ITMGlobalCache<TVoxel> *globalCache;

        // Per-frame rejected CUDA allocation requests, copied with the existing
        // allocation-status transfer. These diagnostics are not persisted state.
#ifndef GPS_SLAM_DISABLE_BASIC_CAPACITY_INSTRUMENTATION
        int lastFailedVoxelAllocations = 0;
        int lastFailedExcessAllocations = 0;
        bool allocationFailureCountersAvailable = false;
#endif

		void SaveToDirectory(const std::string &outputDirectory) const
		{
			localVBA.SaveToDirectory(outputDirectory);
			index.SaveToDirectory(outputDirectory);
		}

		void SaveSparseToDirectory(const std::string &outputDirectory) const
		{
			localVBA.SaveSparseToDirectory(outputDirectory, usedBlockPtrs());
			index.SaveToDirectory(outputDirectory);
		}

		void LoadFromDirectory(const std::string &outputDirectory)
		{
			if (!localVBA.LoadSparseFromDirectory(outputDirectory))
				localVBA.LoadFromDirectory(outputDirectory);
			index.LoadFromDirectory(outputDirectory);
		}

		void CopyFromCUDA(const ITMScene<TVoxel, TIndex> &source)
		{
			localVBA.CopyFrom(source.localVBA,
				ORUtils::MemoryBlock<TVoxel>::CUDA_TO_CPU,
				ORUtils::MemoryBlock<int>::CUDA_TO_CPU);
			index.CopyFrom(source.index,
				ORUtils::MemoryBlock<ITMHashEntry>::CUDA_TO_CPU,
				ORUtils::MemoryBlock<int>::CUDA_TO_CPU);
		}

		void CopyFromCPU(const ITMScene<TVoxel, TIndex> &source, MemoryDeviceType targetMemoryType)
		{
			if (targetMemoryType == MEMORYDEVICE_CUDA)
			{
				localVBA.CopyFrom(source.localVBA,
					ORUtils::MemoryBlock<TVoxel>::CPU_TO_CUDA,
					ORUtils::MemoryBlock<int>::CPU_TO_CUDA);
				index.CopyFrom(source.index,
					ORUtils::MemoryBlock<ITMHashEntry>::CPU_TO_CUDA,
					ORUtils::MemoryBlock<int>::CPU_TO_CUDA);
			}
			else
			{
				localVBA.CopyFrom(source.localVBA,
					ORUtils::MemoryBlock<TVoxel>::CPU_TO_CPU,
					ORUtils::MemoryBlock<int>::CPU_TO_CPU);
				index.CopyFrom(source.index,
					ORUtils::MemoryBlock<ITMHashEntry>::CPU_TO_CPU,
					ORUtils::MemoryBlock<int>::CPU_TO_CPU);
			}
		}

		std::vector<int> usedBlockPtrs(void) const
		{
			const auto *entries = index.GetEntries();
			const int totalBlocks = index.getNumAllocatedVoxelBlocks();
			std::vector<unsigned char> seen(static_cast<size_t>(totalBlocks), 0);
			std::vector<int> usedBlockPtrs;
			const int estimatedUsedBlocks = totalBlocks - localVBA.lastFreeBlockId - 1;
			if (estimatedUsedBlocks > 0) usedBlockPtrs.reserve(static_cast<size_t>(estimatedUsedBlocks));

			for (int i = 0; i < TIndex::noTotalEntries; ++i) {
				const int ptr = entries[i].ptr;
				if (ptr < 0 || ptr >= totalBlocks || seen[static_cast<size_t>(ptr)]) continue;
				seen[static_cast<size_t>(ptr)] = 1;
				usedBlockPtrs.push_back(ptr);
			}
			return usedBlockPtrs;
		}

		ITMScene(const ITMSceneParams *_sceneParams, bool _useSwapping, MemoryDeviceType _memoryType)
			: sceneParams(_sceneParams), index(_memoryType), localVBA(_memoryType, index.getNumAllocatedVoxelBlocks(), index.getVoxelBlockSize())
		{
			if (_useSwapping)
				globalCache = new ITMGlobalCache<TVoxel>();
			else
				globalCache = NULL;
		}

		~ITMScene(void)
		{
			if (globalCache != NULL)
				delete globalCache;
		}

		// Suppress the default copy constructor and assignment operator
		ITMScene(const ITMScene &);
		ITMScene &operator=(const ITMScene &);
	};
}

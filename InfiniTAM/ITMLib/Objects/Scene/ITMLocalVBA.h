// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "../../../ORUtils/MemoryBlock.h"
#include "../../../ORUtils/MemoryBlockPersister.h"

namespace ITMLib
{
	/** \brief
	Stores the actual voxel content that is referred to by a
	ITMLib::ITMHashTable.
	*/
	template <class TVoxel>
	class ITMLocalVBA
	{
	private:
		ORUtils::MemoryBlock<TVoxel> *voxelBlocks;
		ORUtils::MemoryBlock<int> *allocationList;

		MemoryDeviceType memoryType;

	public:
		inline TVoxel *GetVoxelBlocks(void) { return voxelBlocks->GetData(memoryType); }
		inline const TVoxel *GetVoxelBlocks(void) const { return voxelBlocks->GetData(memoryType); }
		int *GetAllocationList(void) { return allocationList->GetData(memoryType); }
		const int *GetAllocationList(void) const { return allocationList->GetData(memoryType); }

		void CopyFrom(const ITMLocalVBA<TVoxel> &source, typename ORUtils::MemoryBlock<TVoxel>::MemoryCopyDirection voxelDirection,
			typename ORUtils::MemoryBlock<int>::MemoryCopyDirection allocationDirection)
		{
			voxelBlocks->SetFrom(source.voxelBlocks, voxelDirection);
			allocationList->SetFrom(source.allocationList, allocationDirection);
			lastFreeBlockId = source.lastFreeBlockId;
			allocatedSize = source.allocatedSize;
		}

#ifdef COMPILE_WITH_METAL
		const void *GetVoxelBlocks_MB() const { return voxelBlocks->GetMetalBuffer(); }
		const void *GetAllocationList_MB(void) const { return allocationList->GetMetalBuffer(); }
#endif
		int lastFreeBlockId;

		int allocatedSize;

		void SaveToDirectory(const std::string &outputDirectory) const
		{
			std::string VBFileName = outputDirectory + "voxel.dat";
			std::string ALFileName = outputDirectory + "alloc.dat";
			std::string AllocSizeFileName = outputDirectory + "vba.txt";

			ORUtils::MemoryBlockPersister::SaveMemoryBlock(VBFileName, *voxelBlocks, memoryType);
			ORUtils::MemoryBlockPersister::SaveMemoryBlock(ALFileName, *allocationList, memoryType);

			std::ofstream ofs(AllocSizeFileName.c_str());
			if (!ofs)
				throw std::runtime_error("Could not open " + AllocSizeFileName + " for writing");

			ofs << lastFreeBlockId << ' ' << allocatedSize;
		}

		void SaveSparseToDirectory(const std::string &outputDirectory, const std::vector<int> &usedBlockPtrs) const
		{
			const std::string ALFileName = outputDirectory + "alloc.dat";
			const std::string AllocSizeFileName = outputDirectory + "vba.txt";
			const std::string SparseFileName = outputDirectory + "vba_sparse.bin";

			const int totalBlocks = static_cast<int>(allocationList->dataSize);
			const int blockVoxelCount = totalBlocks > 0 ? allocatedSize / totalBlocks : 0;
			if (totalBlocks <= 0 || blockVoxelCount <= 0 || blockVoxelCount * totalBlocks != allocatedSize)
				throw std::runtime_error("Cannot write sparse VBA with invalid allocation geometry");

			ORUtils::MemoryBlockPersister::SaveMemoryBlock(ALFileName, *allocationList, memoryType);

			std::ofstream ofs(AllocSizeFileName.c_str());
			if (!ofs)
				throw std::runtime_error("Could not open " + AllocSizeFileName + " for writing");
			ofs << lastFreeBlockId << ' ' << allocatedSize;
			ofs.close();

			ORUtils::MemoryBlock<TVoxel> cpuVoxels(voxelBlocks->dataSize, MEMORYDEVICE_CPU);
			if (memoryType == MEMORYDEVICE_CUDA)
				cpuVoxels.SetFrom(voxelBlocks, ORUtils::MemoryBlock<TVoxel>::CUDA_TO_CPU);
			else
				cpuVoxels.SetFrom(voxelBlocks, ORUtils::MemoryBlock<TVoxel>::CPU_TO_CPU);
			const TVoxel *voxelData = cpuVoxels.GetData(MEMORYDEVICE_CPU);

			std::ofstream sparse(SparseFileName.c_str(), std::ios::binary);
			if (!sparse)
				throw std::runtime_error("Could not open " + SparseFileName + " for writing");

			const uint32_t magic = 0x31535647; // "GVS1" in little-endian builds.
			const uint32_t version = 1;
			const uint32_t usedCount = static_cast<uint32_t>(usedBlockPtrs.size());
			sparse.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
			sparse.write(reinterpret_cast<const char *>(&version), sizeof(version));
			sparse.write(reinterpret_cast<const char *>(&totalBlocks), sizeof(totalBlocks));
			sparse.write(reinterpret_cast<const char *>(&blockVoxelCount), sizeof(blockVoxelCount));
			sparse.write(reinterpret_cast<const char *>(&allocatedSize), sizeof(allocatedSize));
			sparse.write(reinterpret_cast<const char *>(&usedCount), sizeof(usedCount));
			if (!sparse)
				throw std::runtime_error("Could not write sparse VBA header");

			for (int ptr : usedBlockPtrs) {
				if (ptr < 0 || ptr >= totalBlocks)
					throw std::runtime_error("Cannot write sparse VBA with invalid voxel block pointer");
				sparse.write(reinterpret_cast<const char *>(&ptr), sizeof(ptr));
				sparse.write(reinterpret_cast<const char *>(voxelData + ptr * blockVoxelCount),
					blockVoxelCount * sizeof(TVoxel));
				if (!sparse)
					throw std::runtime_error("Could not write sparse VBA block data");
			}
		}

		void LoadFromDirectory(const std::string &inputDirectory)
		{
			std::string VBFileName = inputDirectory + "voxel.dat";
			std::string ALFileName = inputDirectory + "alloc.dat";
			std::string AllocSizeFileName = inputDirectory + "vba.txt";

			ORUtils::MemoryBlockPersister::LoadMemoryBlock(VBFileName, *voxelBlocks, memoryType);
			ORUtils::MemoryBlockPersister::LoadMemoryBlock(ALFileName, *allocationList, memoryType);

			std::ifstream ifs(AllocSizeFileName.c_str());
			if (!ifs)
				throw std::runtime_error("Could not open " + AllocSizeFileName + " for reading");

			ifs >> lastFreeBlockId >> allocatedSize;
		}

		bool LoadSparseFromDirectory(const std::string &inputDirectory)
		{
			const std::string SparseFileName = inputDirectory + "vba_sparse.bin";
			if (!std::filesystem::exists(SparseFileName)) return false;

			const std::string ALFileName = inputDirectory + "alloc.dat";
			const std::string AllocSizeFileName = inputDirectory + "vba.txt";

			ORUtils::MemoryBlockPersister::LoadMemoryBlock(ALFileName, *allocationList, memoryType);

			std::ifstream ifs(AllocSizeFileName.c_str());
			if (!ifs)
				throw std::runtime_error("Could not open " + AllocSizeFileName + " for reading");
			ifs >> lastFreeBlockId >> allocatedSize;

			std::ifstream sparse(SparseFileName.c_str(), std::ios::binary);
			if (!sparse)
				throw std::runtime_error("Could not open " + SparseFileName + " for reading");

			uint32_t magic = 0, version = 0, usedCount = 0;
			int totalBlocks = 0, blockVoxelCount = 0, fileAllocatedSize = 0;
			sparse.read(reinterpret_cast<char *>(&magic), sizeof(magic));
			sparse.read(reinterpret_cast<char *>(&version), sizeof(version));
			sparse.read(reinterpret_cast<char *>(&totalBlocks), sizeof(totalBlocks));
			sparse.read(reinterpret_cast<char *>(&blockVoxelCount), sizeof(blockVoxelCount));
			sparse.read(reinterpret_cast<char *>(&fileAllocatedSize), sizeof(fileAllocatedSize));
			sparse.read(reinterpret_cast<char *>(&usedCount), sizeof(usedCount));
			if (!sparse || magic != 0x31535647 || version != 1)
				throw std::runtime_error("Malformed sparse VBA header in " + SparseFileName);
			if (totalBlocks != static_cast<int>(allocationList->dataSize) || fileAllocatedSize != allocatedSize)
				throw std::runtime_error("Sparse VBA allocation geometry does not match runtime hash profile");
			if (blockVoxelCount <= 0 || blockVoxelCount * totalBlocks != allocatedSize)
				throw std::runtime_error("Sparse VBA block geometry is invalid");

			ORUtils::MemoryBlock<TVoxel> cpuVoxels(static_cast<size_t>(allocatedSize), MEMORYDEVICE_CPU);
			cpuVoxels.Clear();
			TVoxel *voxelData = cpuVoxels.GetData(MEMORYDEVICE_CPU);
			std::vector<unsigned char> seen(static_cast<size_t>(totalBlocks), 0);

			for (uint32_t i = 0; i < usedCount; ++i) {
				int ptr = -1;
				sparse.read(reinterpret_cast<char *>(&ptr), sizeof(ptr));
				if (!sparse || ptr < 0 || ptr >= totalBlocks || seen[static_cast<size_t>(ptr)])
					throw std::runtime_error("Malformed sparse VBA block pointer in " + SparseFileName);
				seen[static_cast<size_t>(ptr)] = 1;
				sparse.read(reinterpret_cast<char *>(voxelData + ptr * blockVoxelCount),
					blockVoxelCount * sizeof(TVoxel));
				if (!sparse)
					throw std::runtime_error("Malformed sparse VBA block payload in " + SparseFileName);
			}

			if (memoryType == MEMORYDEVICE_CUDA)
				voxelBlocks->SetFrom(&cpuVoxels, ORUtils::MemoryBlock<TVoxel>::CPU_TO_CUDA);
			else
				voxelBlocks->SetFrom(&cpuVoxels, ORUtils::MemoryBlock<TVoxel>::CPU_TO_CPU);
			return true;
		}

		ITMLocalVBA(MemoryDeviceType memoryType, int noBlocks, int blockSize)
		{
			this->memoryType = memoryType;

			allocatedSize = noBlocks * blockSize;

			voxelBlocks = new ORUtils::MemoryBlock<TVoxel>(allocatedSize, memoryType);
			allocationList = new ORUtils::MemoryBlock<int>(noBlocks, memoryType);
		}

		~ITMLocalVBA(void)
		{
			delete voxelBlocks;
			delete allocationList;
		}

		// Suppress the default copy constructor and assignment operator
		ITMLocalVBA(const ITMLocalVBA &);
		ITMLocalVBA &operator=(const ITMLocalVBA &);
	};
}

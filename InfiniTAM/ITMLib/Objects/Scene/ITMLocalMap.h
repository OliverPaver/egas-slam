// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <future>
#include <map>
#include <stdexcept>

#include "../../Engines/Visualisation/Interface/ITMVisualisationEngine.h"
#include "../../Objects/RenderStates/ITMRenderState.h"
#include "../../Objects/Scene/ITMScene.h"
#include "../../Objects/Tracking/ITMTrackingState.h"
#include "../../Utils/ITMLibSettings.h"

namespace ITMLib {
	struct ITMPoseConstraint
	{
	public:
		ITMPoseConstraint(void)
		{
			accu_num = 0;
		}

		void AddObservation(const ORUtils::SE3Pose & relative_pose, int weight = 1)
		{
			Matrix4f tmp = accu_poses.GetM() * (float)accu_num + relative_pose.GetM() * (float)weight;
			accu_num += weight;
			accu_poses.SetM(tmp / (float)accu_num);
			accu_poses.Coerce();
		}

		ORUtils::SE3Pose GetAccumulatedObservations(void) const { return accu_poses; }
		int GetNumAccumulatedObservations(void) const { return accu_num; }

	private:
		ORUtils::SE3Pose accu_poses;
		int accu_num;
	};

	typedef std::map<int, ITMPoseConstraint> ConstraintList;

	template<class TVoxel, class TIndex>
	class ITMLocalMap
	{
	private:
		struct ResidentData
		{
			ITMScene<TVoxel, TIndex> *scene;
			int64_t estimatedBytes;
			ResidentData(void)
				: scene(NULL), estimatedBytes(0)
			{}
		};

		static void deleteResidentData(ResidentData &data)
		{
			delete data.scene; data.scene = NULL;
			data.estimatedBytes = 0;
		}

	public:
		ITMScene<TVoxel, TIndex> *scene;
		ITMRenderState *renderState;
		ITMTrackingState *trackingState;
		ConstraintList relations;
		ORUtils::SE3Pose estimatedGlobalPose;
		bool resident;
		bool dirty;
		int cachedBlockCount;
		bool cachedBlockBoundsValid;
		Vector3f cachedBlockBoundsMin;
		Vector3f cachedBlockBoundsMax;
		int64_t cachedEstimatedSceneBytes;
		std::string pageDirectory;

		ITMLocalMap(const ITMLibSettings *settings, const ITMVisualisationEngine<TVoxel, TIndex> *visualisationEngine, const Vector2i & trackedImageSize)
			: settings(settings), visualisationEngine(visualisationEngine), trackedImageSize(trackedImageSize),
			  scene(NULL), renderState(NULL), trackingState(NULL), resident(false), dirty(false), cachedBlockCount(0),
			  cachedBlockBoundsValid(false), cachedBlockBoundsMin(0.0f), cachedBlockBoundsMax(0.0f),
			  cachedEstimatedSceneBytes(0), savedTrackingResult(ITMTrackingState::TRACKING_GOOD)
		{
			allocateResidentData();
		}
		~ITMLocalMap(void)
		{
			waitForSnapshot();
			if (pageInFuture.valid())
			{
				try {
					ResidentData data = pageInFuture.get();
					deleteResidentData(data);
				}
				catch (...) {}
			}
			releaseResidentData();
			deleteResidentData(cpuPageCache);
		}

		bool isResident(void) const { return resident; }
			bool hasPendingSnapshot(void) const { return snapshotFuture.valid(); }
			bool hasPendingPageIn(void) const { return pageInFuture.valid(); }
			bool hasCpuPageCache(void) const { return cpuPageCache.scene != NULL; }
			int64_t cpuPageCacheEstimatedBytes(void) const
			{
				return cpuPageCache.scene != NULL ? cpuPageCache.estimatedBytes : 0;
			}
			int64_t estimatedSceneBytes(void) const
			{
				if (scene != NULL)
					return estimateSceneBytes(scene);
			return cachedEstimatedSceneBytes;
		}
		void markDirty(void) { dirty = true; }
		void waitForSnapshot(void)
		{
			if (!snapshotFuture.valid()) return;
			ResidentData data = snapshotFuture.get();
			if (data.scene != NULL)
			{
				deleteResidentData(cpuPageCache);
				cpuPageCache = data;
				cachedEstimatedSceneBytes = data.estimatedBytes;
			}
		}
		void waitForPageIn(void)
		{
			completePendingPageIn();
		}
		bool tryCompletePageIn(void)
		{
			if (resident) return true;
			if (!pageInFuture.valid()) return false;
			if (pageInFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
				return false;
			completePendingPageIn();
			return resident;
		}

		void pageOut(const std::string &directory, bool asyncSnapshot = false, bool keepCpuSnapshot = false)
		{
			if (pageInFuture.valid()) completePendingPageIn();
			if (!resident) return;
			std::filesystem::path target(directory), staging(directory + ".tmp"), backup(directory + ".bak");
			cachedBlockCount = scene->index.getNumAllocatedVoxelBlocks() - scene->localVBA.lastFreeBlockId - 1;
			cachedEstimatedSceneBytes = estimateSceneBytes(scene);
			savedTrackingPose = *(trackingState->pose_d);
			savedTrackingResult = trackingState->trackerResult;
			if (keepCpuSnapshot)
			{
				waitForSnapshot();
				if (dirty || cpuPageCache.scene == NULL)
				{
					deleteResidentData(cpuPageCache);
					ITMScene<TVoxel, TIndex> *sourceScene = scene;
					ITMRenderState *sourceRenderState = renderState;
					ITMTrackingState *sourceTrackingState = trackingState;
					scene = NULL;
					renderState = NULL;
					trackingState = NULL;
					resident = false;
					pageDirectory = target.string();
					dirty = false;
					const ITMSceneParams sceneParams = settings->sceneParams;
					if (asyncSnapshot)
					{
						snapshotFuture = std::async(std::launch::async,
							[sourceScene, sourceRenderState, sourceTrackingState, sceneParams]() mutable {
							ResidentData data;
							try {
								data.scene = new ITMScene<TVoxel, TIndex>(&sceneParams, false, MEMORYDEVICE_CPU);
								data.scene->CopyFromCUDA(*sourceScene);
								data.estimatedBytes = estimateSceneBytes(data.scene);
								delete sourceScene; sourceScene = NULL;
								delete sourceRenderState; sourceRenderState = NULL;
								delete sourceTrackingState; sourceTrackingState = NULL;
								return data;
							}
							catch (...) {
								delete sourceScene;
								delete sourceRenderState;
								delete sourceTrackingState;
								deleteResidentData(data);
								throw;
							}
						});
						return;
					}
					ResidentData data;
					try {
						data.scene = new ITMScene<TVoxel, TIndex>(&sceneParams, false, MEMORYDEVICE_CPU);
						data.scene->CopyFromCUDA(*sourceScene);
						data.estimatedBytes = estimateSceneBytes(data.scene);
						delete sourceScene; sourceScene = NULL;
						delete sourceRenderState; sourceRenderState = NULL;
						delete sourceTrackingState; sourceTrackingState = NULL;
						cpuPageCache = data;
						cachedEstimatedSceneBytes = data.estimatedBytes;
					}
					catch (...) {
						delete sourceScene;
						delete sourceRenderState;
						delete sourceTrackingState;
						deleteResidentData(data);
						throw;
					}
					return;
				}
				pageDirectory = target.string();
				dirty = false;
				releaseResidentData();
				return;
			}
			if (dirty || pageDirectory.empty() || !std::filesystem::exists(pageDirectory)) {
				waitForSnapshot();
				deleteResidentData(cpuPageCache);
				ITMScene<TVoxel, TIndex> *sourceScene = scene;
				ITMRenderState *sourceRenderState = renderState;
				ITMTrackingState *sourceTrackingState = trackingState;
				scene = NULL;
				renderState = NULL;
				trackingState = NULL;
				resident = false;
				pageDirectory = target.string();
				dirty = false;
				const ITMSceneParams sceneParams = settings->sceneParams;
				if (asyncSnapshot) {
					snapshotFuture = std::async(std::launch::async,
						[sourceScene, sourceRenderState, sourceTrackingState, sceneParams, target, staging, backup]() mutable {
						ResidentData data;
						ITMScene<TVoxel, TIndex> *cpuScene = NULL;
						try {
							cpuScene = new ITMScene<TVoxel, TIndex>(&sceneParams, false, MEMORYDEVICE_CPU);
							cpuScene->CopyFromCUDA(*sourceScene);
							delete sourceScene; sourceScene = NULL;
							delete sourceRenderState; sourceRenderState = NULL;
							delete sourceTrackingState; sourceTrackingState = NULL;
							std::filesystem::remove_all(staging);
							std::filesystem::create_directories(staging);
							if (std::getenv("GPS_SLAM_DISABLE_SPARSE_TSDF_PAGES") != NULL)
								cpuScene->SaveToDirectory(staging.string() + "/");
							else
								cpuScene->SaveSparseToDirectory(staging.string() + "/");
							std::filesystem::remove_all(backup);
							if (std::filesystem::exists(target)) std::filesystem::rename(target, backup);
							try { std::filesystem::rename(staging, target); }
							catch (...) {
								if (!std::filesystem::exists(target) && std::filesystem::exists(backup))
									std::filesystem::rename(backup, target);
								throw;
							}
							std::filesystem::remove_all(backup);
							delete cpuScene;
							return data;
						}
						catch (...) {
							delete sourceScene;
							delete sourceRenderState;
							delete sourceTrackingState;
							delete cpuScene;
							throw;
						}
					});
					return;
				}
				ITMScene<TVoxel, TIndex> *cpuScene = NULL;
				try {
					cpuScene = new ITMScene<TVoxel, TIndex>(&sceneParams, false, MEMORYDEVICE_CPU);
					cpuScene->CopyFromCUDA(*sourceScene);
					delete sourceScene; sourceScene = NULL;
					delete sourceRenderState; sourceRenderState = NULL;
					delete sourceTrackingState; sourceTrackingState = NULL;
				}
				catch (...) {
					delete sourceScene;
					delete sourceRenderState;
					delete sourceTrackingState;
					delete cpuScene;
					throw;
				}
				snapshotFuture = std::async(std::launch::async,
					[cpuScene, target, staging, backup]() mutable {
					ResidentData data;
					try {
						std::filesystem::remove_all(staging);
						std::filesystem::create_directories(staging);
						if (std::getenv("GPS_SLAM_DISABLE_SPARSE_TSDF_PAGES") != NULL)
							cpuScene->SaveToDirectory(staging.string() + "/");
						else
							cpuScene->SaveSparseToDirectory(staging.string() + "/");
						std::filesystem::remove_all(backup);
						if (std::filesystem::exists(target)) std::filesystem::rename(target, backup);
						try { std::filesystem::rename(staging, target); }
						catch (...) {
							if (!std::filesystem::exists(target) && std::filesystem::exists(backup))
								std::filesystem::rename(backup, target);
							throw;
						}
						std::filesystem::remove_all(backup);
					}
					catch (...) {
						delete cpuScene;
						throw;
					}
					delete cpuScene;
					return data;
				});
				return;
			}
			releaseResidentData();
		}

		void pageIn(void)
		{
			if (resident) return;
			if (pageInFuture.valid())
			{
				completePendingPageIn();
				return;
			}
			waitForSnapshot();
			if (cpuPageCache.scene != NULL)
			{
				allocateResidentData();
				const MemoryDeviceType memoryType = settings->deviceType == ITMLibSettings::DEVICE_CUDA ? MEMORYDEVICE_CUDA : MEMORYDEVICE_CPU;
				try {
					scene->CopyFromCPU(*cpuPageCache.scene, memoryType);
					trackingState->pose_d->SetFrom(&savedTrackingPose);
					trackingState->trackerResult = savedTrackingResult;
					trackingState->age_pointCloud = -1;
					dirty = false;
					cachedEstimatedSceneBytes = cpuPageCache.estimatedBytes;
				}
				catch (...) {
					releaseResidentData();
					throw;
				}
				return;
			}
			if (pageDirectory.empty() || !std::filesystem::exists(pageDirectory))
				throw std::runtime_error("Cannot page in local map without a valid snapshot: " + pageDirectory);
			allocateResidentData();
			try {
				scene->LoadFromDirectory(pageDirectory + "/");
				trackingState->pose_d->SetFrom(&savedTrackingPose);
				trackingState->trackerResult = savedTrackingResult;
				trackingState->age_pointCloud = -1;
				dirty = false;
			}
			catch (...) {
				releaseResidentData();
				throw;
			}
		}

		void beginPageInAsync(void)
		{
			if (resident || pageInFuture.valid()) return;
			waitForSnapshot();
			if (cpuPageCache.scene != NULL) return;
			if (pageDirectory.empty() || !std::filesystem::exists(pageDirectory))
				throw std::runtime_error("Cannot prefetch local map without a valid snapshot: " + pageDirectory);
			const ITMSceneParams sceneParams = settings->sceneParams;
			const std::string directory = pageDirectory;
			pageInFuture = std::async(std::launch::async,
				[sceneParams, directory]() mutable {
				ResidentData data;
				try {
					data.scene = new ITMScene<TVoxel, TIndex>(&sceneParams, false, MEMORYDEVICE_CPU);
					data.scene->LoadFromDirectory(directory + "/");
					data.estimatedBytes = estimateSceneBytes(data.scene);
					return data;
				}
				catch (...) {
					deleteResidentData(data);
					throw;
				}
			});
		}

		void adoptPageSnapshot(const std::string &directory, int blockCount)
		{
			const int64_t estimate = estimatedSceneBytes();
			if (resident) releaseResidentData();
			waitForSnapshot();
			deleteResidentData(cpuPageCache);
			pageDirectory = directory;
			cachedBlockCount = blockCount;
			cachedEstimatedSceneBytes = estimate;
			dirty = false;
		}

			void saveCpuPageCacheToDirectory(const std::string &outputDirectory) const
			{
				if (cpuPageCache.scene == NULL)
					throw std::runtime_error("Cannot save missing CPU page cache");
				if (std::getenv("GPS_SLAM_DISABLE_SPARSE_TSDF_PAGES") != NULL)
					cpuPageCache.scene->SaveToDirectory(outputDirectory);
				else
					cpuPageCache.scene->SaveSparseToDirectory(outputDirectory);
			}

			int64_t spillCpuPageCacheToDirectory(const std::string &directory)
			{
				waitForSnapshot();
				if (cpuPageCache.scene == NULL) return 0;
				const int64_t spilledBytes = cpuPageCache.estimatedBytes;
				std::filesystem::path target(directory), staging(directory + ".tmp"), backup(directory + ".bak");
				std::filesystem::remove_all(staging);
				std::filesystem::create_directories(staging);
				try
				{
					saveCpuPageCacheToDirectory(staging.string() + "/");
					std::filesystem::remove_all(backup);
					if (std::filesystem::exists(target)) std::filesystem::rename(target, backup);
					try { std::filesystem::rename(staging, target); }
					catch (...)
					{
						if (!std::filesystem::exists(target) && std::filesystem::exists(backup))
							std::filesystem::rename(backup, target);
						throw;
					}
					std::filesystem::remove_all(backup);
				}
				catch (...)
				{
					std::filesystem::remove_all(staging);
					throw;
				}
				pageDirectory = target.string();
				deleteResidentData(cpuPageCache);
				return spilledBytes;
			}

		private:
		const ITMLibSettings *settings;
		const ITMVisualisationEngine<TVoxel, TIndex> *visualisationEngine;
		Vector2i trackedImageSize;
		ORUtils::SE3Pose savedTrackingPose;
		ITMTrackingState::TrackingResult savedTrackingResult;
		std::future<ResidentData> snapshotFuture;
		std::future<ResidentData> pageInFuture;
		ResidentData cpuPageCache;

		static int64_t estimateSceneBytes(const ITMScene<TVoxel, TIndex> *scene)
		{
			if (scene == NULL) return 0;
			const int voxelBlocks = scene->index.getNumAllocatedVoxelBlocks();
			return static_cast<int64_t>(scene->localVBA.allocatedSize) * static_cast<int64_t>(sizeof(TVoxel))
				+ static_cast<int64_t>(voxelBlocks) * static_cast<int64_t>(sizeof(int))
				+ static_cast<int64_t>(TIndex::noTotalEntries) * static_cast<int64_t>(sizeof(ITMHashEntry))
				+ static_cast<int64_t>(SDF_EXCESS_LIST_SIZE) * static_cast<int64_t>(sizeof(int));
		}

		void allocateResidentData(void)
		{
			if (resident) return;
			MemoryDeviceType memoryType = settings->deviceType == ITMLibSettings::DEVICE_CUDA ? MEMORYDEVICE_CUDA : MEMORYDEVICE_CPU;
			scene = new ITMScene<TVoxel, TIndex>(&settings->sceneParams, settings->swappingMode == ITMLibSettings::SWAPPINGMODE_ENABLED, memoryType);
			renderState = visualisationEngine->CreateRenderState(scene, trackedImageSize);
			trackingState = new ITMTrackingState(trackedImageSize, memoryType);
			resident = true;
			cachedEstimatedSceneBytes = estimateSceneBytes(scene);
		}

		void releaseResidentData(void)
		{
			delete scene; scene = NULL;
			delete renderState; renderState = NULL;
			delete trackingState; trackingState = NULL;
			resident = false;
		}

		void completePendingPageIn(void)
		{
			if (!pageInFuture.valid()) return;
			ResidentData data = pageInFuture.get();
			if (resident)
			{
				deleteResidentData(data);
				return;
			}
			allocateResidentData();
			const MemoryDeviceType memoryType = settings->deviceType == ITMLibSettings::DEVICE_CUDA ? MEMORYDEVICE_CUDA : MEMORYDEVICE_CPU;
			try {
				scene->CopyFromCPU(*data.scene, memoryType);
				trackingState->pose_d->SetFrom(&savedTrackingPose);
				trackingState->trackerResult = savedTrackingResult;
				trackingState->age_pointCloud = -1;
				dirty = false;
				cachedEstimatedSceneBytes = data.estimatedBytes;
			}
			catch (...) {
				releaseResidentData();
				deleteResidentData(data);
				throw;
			}
			deleteResidentData(data);
		}

	public:
	};
}

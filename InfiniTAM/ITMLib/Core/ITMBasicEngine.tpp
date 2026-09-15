// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMBasicEngine.h"

#include "../Engines/LowLevel/ITMLowLevelEngineFactory.h"
#include "../Engines/Meshing/ITMMeshingEngineFactory.h"
#include "../Engines/ViewBuilding/ITMViewBuilderFactory.h"
#include "../Engines/Visualisation/ITMVisualisationEngineFactory.h"
#include "../Objects/RenderStates/ITMRenderStateFactory.h"
#include "../Objects/RenderStates/ITMRenderState_VH.h"
#include "../Trackers/ITMTrackerFactory.h"
#include "../Utils/ITMTrackingFailurePolicy.h"

#include "../../ORUtils/NVTimer.h"
#include "../../ORUtils/FileUtils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <stdexcept>

// #define OUTPUT_TRAJECTORY_QUATERNIONS

using namespace ITMLib;

static bool basicEngineIsFiniteMatrix(const ORUtils::Matrix4<float> &matrix)
{
	for (int i = 0; i < 16; ++i)
	{
		if (!std::isfinite(matrix.m[i])) return false;
	}
	return true;
}

static bool basicEngineIsFinitePose(const ORUtils::SE3Pose &pose)
{
	return basicEngineIsFiniteMatrix(pose.GetM()) && basicEngineIsFiniteMatrix(pose.GetInvM());
}

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::configureCapacityDiagnostics(const std::string &csvPath)
{
#ifdef GPS_SLAM_DISABLE_BASIC_CAPACITY_INSTRUMENTATION
    (void)csvPath;
    throw std::runtime_error("Capacity diagnostics were compiled out of this performance-control binary");
#else
    if constexpr (!std::is_same<TIndex, ITMVoxelBlockHash>::value)
        throw std::runtime_error("Capacity diagnostics require the voxel hash index");
    capacityDiagnosticLog.open(csvPath);
    if (!capacityDiagnosticLog)
        throw std::runtime_error("Cannot open capacity diagnostics: " + csvPath);
    capacityDiagnosticLog << "frame,raw_result,policy_result,did_fusion,pose_finite,"
        "blocks_capacity,blocks_used,blocks_free,hash_buckets,excess_capacity,excess_free,"
        "visible_blocks,counters_available,failed_voxel_requests,failed_excess_requests\n";
#endif
}

#ifndef GPS_SLAM_DISABLE_BASIC_CAPACITY_INSTRUMENTATION
template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::logCapacityDiagnostics(int frameNo, bool didFusion)
{
    if (!capacityDiagnosticLog.is_open()) return;
    if constexpr (std::is_same<TIndex, ITMVoxelBlockHash>::value)
    {
        const int freeBlocks = scene->localVBA.lastFreeBlockId + 1;
        const auto *renderState = static_cast<const ITMRenderState_VH *>(renderState_live);
        capacityDiagnosticLog << frameNo << ',' << lastRawTrackingResult << ','
            << lastPolicyTrackingResult << ',' << didFusion << ','
            << basicEngineIsFinitePose(*(trackingState->pose_d)) << ','
            << SDF_LOCAL_BLOCK_NUM << ',' << SDF_LOCAL_BLOCK_NUM - freeBlocks << ','
            << freeBlocks << ',' << SDF_BUCKET_NUM << ',' << SDF_EXCESS_LIST_SIZE << ','
            << scene->index.GetLastFreeExcessListId() + 1 << ',' << renderState->noVisibleEntries << ','
            << scene->allocationFailureCountersAvailable << ',' << scene->lastFailedVoxelAllocations << ','
            << scene->lastFailedExcessAllocations << '\n';
        // Flush once after the complete buffered row, not after each field.
        // The legacy application does not destroy its heap-allocated engine
        // at normal exit. This also preserves every completed row on early
        // exit without relying on its stream destructor or an EOF callback.
        capacityDiagnosticLog.flush();
        if (!capacityDiagnosticLog)
            throw std::runtime_error("Cannot write capacity diagnostics");
    }
}

#endif

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::configureTrackingDiagnostics(int startFrame, int endFrame, const std::string &csvPath)
{
	trackingDiagnosticStartFrame = startFrame;
	trackingDiagnosticEndFrame = endFrame;
	if (trackingDiagnosticEndFrame < trackingDiagnosticStartFrame ||
		trackingDiagnosticStartFrame < 0 || csvPath.empty())
	{
		if (trackingDiagnosticLog.is_open()) trackingDiagnosticLog.close();
		trackingDiagnosticStartFrame = -1;
		trackingDiagnosticEndFrame = -1;
		return;
	}
	trackingDiagnosticLog.open(csvPath.c_str());
	if (!trackingDiagnosticLog)
	{
		fprintf(stdout, "[BASIC TRACKING DIAG] failed_to_open path=%s\n", csvPath.c_str());
		trackingDiagnosticStartFrame = -1;
		trackingDiagnosticEndFrame = -1;
		return;
	}
	trackingDiagnosticLog
		<< "frame,stage,tracker_result,did_fusion,pose_finite,pose_translation_norm,"
		<< "depth_valid,depth_nonfinite,depth_min,depth_max,depth_mean,"
		<< "depth_normal_finite,depth_normal_nonfinite,"
		<< "raycast_finite,raycast_nonfinite,raycast_valid_w,raycast_min_z,raycast_max_z,raycast_mean_z\n";
	trackingDiagnosticLog.flush();
}

template <typename TVoxel, typename TIndex>
bool ITMBasicEngine<TVoxel, TIndex>::shouldLogTrackingDiagnostics(int frameNo) const
{
	return trackingDiagnosticLog.is_open() &&
		frameNo >= trackingDiagnosticStartFrame &&
		frameNo <= trackingDiagnosticEndFrame;
}

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::logTrackingDiagnostics(int frameNo, const char *stage,
	ITMTrackingState::TrackingResult trackerResult, bool didFusion)
{
	if (!shouldLogTrackingDiagnostics(frameNo)) return;

	int depthValid = 0, depthNonFinite = 0;
	float depthMin = std::numeric_limits<float>::infinity();
	float depthMax = 0.0f;
	double depthSum = 0.0;
	if (view != NULL && view->depth != NULL)
	{
		view->depth->UpdateHostFromDevice();
		const float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
		if (depth != NULL)
		{
			for (size_t i = 0; i < view->depth->dataSize; ++i)
			{
				const float d = depth[i];
				if (!std::isfinite(d))
				{
					++depthNonFinite;
					continue;
				}
				if (d <= 0.0f) continue;
				++depthValid;
				depthSum += d;
				depthMin = std::min(depthMin, d);
				depthMax = std::max(depthMax, d);
			}
		}
	}
	if (depthValid == 0) depthMin = 0.0f;
	const double depthMean = depthValid > 0 ? depthSum / double(depthValid) : 0.0;

	int normalFinite = 0, normalNonFinite = 0;
	if (view != NULL && view->depthNormal != NULL)
	{
		view->depthNormal->UpdateHostFromDevice();
		const Vector4f *normals = view->depthNormal->GetData(MEMORYDEVICE_CPU);
		if (normals != NULL)
		{
			for (size_t i = 0; i < view->depthNormal->dataSize; ++i)
			{
				const Vector4f &n = normals[i];
				if (std::isfinite(n.x) && std::isfinite(n.y) &&
					std::isfinite(n.z) && std::isfinite(n.w))
					++normalFinite;
				else
					++normalNonFinite;
			}
		}
	}

	int raycastFinite = 0, raycastNonFinite = 0, raycastValidW = 0;
	float raycastMinZ = std::numeric_limits<float>::infinity();
	float raycastMaxZ = 0.0f;
	double raycastSumZ = 0.0;
	if (renderState_live != NULL && renderState_live->raycastResult != NULL)
	{
		ITMFloat4Image raycastCpu(renderState_live->raycastResult->noDims, true, false);
		const ORUtils::MemoryBlock<Vector4f>::MemoryCopyDirection copyDirection =
			settings->deviceType == ITMLibSettings::DEVICE_CUDA
				? ORUtils::MemoryBlock<Vector4f>::CUDA_TO_CPU
				: ORUtils::MemoryBlock<Vector4f>::CPU_TO_CPU;
		raycastCpu.SetFrom(renderState_live->raycastResult, copyDirection);
		const Vector4f *raycast = raycastCpu.GetData(MEMORYDEVICE_CPU);
		if (raycast != NULL)
		{
			for (size_t i = 0; i < raycastCpu.dataSize; ++i)
			{
				const Vector4f &p = raycast[i];
				if (std::isfinite(p.x) && std::isfinite(p.y) &&
					std::isfinite(p.z) && std::isfinite(p.w))
				{
					++raycastFinite;
					if (p.w > 0.0f)
					{
						++raycastValidW;
						raycastSumZ += p.z;
						raycastMinZ = std::min(raycastMinZ, p.z);
						raycastMaxZ = std::max(raycastMaxZ, p.z);
					}
				}
				else
				{
					++raycastNonFinite;
				}
			}
		}
	}
	if (raycastValidW == 0) raycastMinZ = 0.0f;
	const double raycastMeanZ = raycastValidW > 0 ? raycastSumZ / double(raycastValidW) : 0.0;

	const bool poseFinite = basicEngineIsFinitePose(*(trackingState->pose_d));
	const ORUtils::Matrix4<float> c2w = trackingState->pose_d->GetInvM();
	const float tx = c2w.m[3];
	const float ty = c2w.m[7];
	const float tz = c2w.m[11];
	const float poseTranslationNorm = std::sqrt(tx * tx + ty * ty + tz * tz);

	trackingDiagnosticLog << frameNo << ","
		<< (stage == NULL ? "" : stage) << ","
		<< trackerResult << ","
		<< (didFusion ? 1 : 0) << ","
		<< (poseFinite ? 1 : 0) << ","
		<< poseTranslationNorm << ","
		<< depthValid << ","
		<< depthNonFinite << ","
		<< depthMin << ","
		<< depthMax << ","
		<< depthMean << ","
		<< normalFinite << ","
		<< normalNonFinite << ","
		<< raycastFinite << ","
		<< raycastNonFinite << ","
		<< raycastValidW << ","
		<< raycastMinZ << ","
		<< raycastMaxZ << ","
		<< raycastMeanZ << "\n";
	trackingDiagnosticLog.flush();
}

template <typename TVoxel, typename TIndex>
ITMBasicEngine<TVoxel, TIndex>::ITMBasicEngine(const ITMLibSettings *settings, const ITMRGBDCalib &calib, Vector2i imgSize_rgb, Vector2i imgSize_d)
{
	this->settings = settings;

	if ((imgSize_d.x == -1) || (imgSize_d.y == -1))
		imgSize_d = imgSize_rgb;

	MemoryDeviceType memoryType = settings->GetMemoryType();
	this->scene = new ITMScene<TVoxel, TIndex>(&settings->sceneParams, settings->swappingMode == ITMLibSettings::SWAPPINGMODE_ENABLED, memoryType);

	const ITMLibSettings::DeviceType deviceType = settings->deviceType;

	lowLevelEngine = ITMLowLevelEngineFactory::MakeLowLevelEngine(deviceType);
	viewBuilder = ITMViewBuilderFactory::MakeViewBuilder(calib, deviceType);
	visualisationEngine = ITMVisualisationEngineFactory::MakeVisualisationEngine<TVoxel, TIndex>(deviceType);

	meshingEngine = NULL;
	if (settings->createMeshingEngine)
		meshingEngine = ITMMeshingEngineFactory::MakeMeshingEngine<TVoxel, TIndex>(deviceType);

	denseMapper = new ITMDenseMapper<TVoxel, TIndex>(settings);
	denseMapper->ResetScene(scene);

	imuCalibrator = new ITMIMUCalibrator_iPad();
	tracker = ITMTrackerFactory::Instance().Make(imgSize_rgb, imgSize_d, settings, lowLevelEngine, imuCalibrator, scene->sceneParams);
	trackingController = new ITMTrackingController(tracker, settings);

	Vector2i trackedImageSize = trackingController->GetTrackedImageSize(imgSize_rgb, imgSize_d);

	renderState_live = ITMRenderStateFactory<TIndex>::CreateRenderState(trackedImageSize, scene->sceneParams, memoryType);
	renderState_freeview = ITMRenderStateFactory<TIndex>::CreateRenderState(trackedImageSize, scene->sceneParams, settings->GetMemoryType());

	trackingState = new ITMTrackingState(trackedImageSize, memoryType);
	tracker->UpdateInitialPose(trackingState);

	view = NULL; // will be allocated by the view builder

	if (settings->behaviourOnFailure == settings->FAILUREMODE_RELOCALISE)
		relocaliser = new FernRelocLib::Relocaliser<float>(imgSize_d, Vector2f(settings->sceneParams.viewFrustum_min, settings->sceneParams.viewFrustum_max), 0.2f, 500, 4);
	else
		relocaliser = NULL;

	kfRaycast = new ITMUChar4Image(imgSize_d, memoryType);

	trackingActive = true;
	fusionActive = true;
	mainProcessingActive = true;
	trackingInitialised = false;
	relocalisationCount = 0;
	framesProcessed = 0;
}

template <typename TVoxel, typename TIndex>
ITMBasicEngine<TVoxel, TIndex>::~ITMBasicEngine()
{
	delete renderState_live;
	if (renderState_freeview != NULL)
		delete renderState_freeview;

	delete scene;

	delete denseMapper;
	delete trackingController;

	delete tracker;
	delete imuCalibrator;

	delete lowLevelEngine;
	delete viewBuilder;

	delete trackingState;
	if (view != NULL)
		delete view;

	delete visualisationEngine;

	if (relocaliser != NULL)
		delete relocaliser;
	delete kfRaycast;

	if (meshingEngine != NULL)
		delete meshingEngine;
}

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::SaveSceneToMesh(const char *fileName)
{
	if (meshingEngine == NULL)
		return;

	ITMMesh *mesh = new ITMMesh(settings->GetMemoryType());

	meshingEngine->MeshScene(mesh, scene);
	mesh->WritePLY(fileName);
	// mesh->WriteSTL(objFileName);
	// mesh->WriteOBJ(objFileName);
	delete mesh;
}

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::SaveToFile(const std::string &saveOutputDirectory)
{
	// throws error if any of the saves fail

	// std::string saveOutputDirectory = "State/";
	std::string relocaliserOutputDirectory = saveOutputDirectory + "Relocaliser/", sceneOutputDirectory = saveOutputDirectory + "Scene/";

	MakeDir(saveOutputDirectory.c_str());
	MakeDir(relocaliserOutputDirectory.c_str());
	MakeDir(sceneOutputDirectory.c_str());

	if (relocaliser)
		relocaliser->SaveToDirectory(relocaliserOutputDirectory);

	scene->SaveToDirectory(sceneOutputDirectory);
}

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::LoadFromFile(const std::string &saveInputDirectory)
{
	// std::string saveInputDirectory = "State/";
	std::string relocaliserInputDirectory = saveInputDirectory + "Relocaliser/", sceneInputDirectory = saveInputDirectory + "Scene/";

	////TODO: add factory for relocaliser and rebuild using config from relocaliserOutputDirectory + "config.txt"
	////TODO: add proper management of case when scene load fails (keep old scene or also reset relocaliser)

	this->resetAll();

	// try // load relocaliser
	// {
	// 	FernRelocLib::Relocaliser<float> *relocaliser_temp = new FernRelocLib::Relocaliser<float>(view->depth->noDims, Vector2f(settings->sceneParams.viewFrustum_min, settings->sceneParams.viewFrustum_max), 0.2f, 500, 4);

	// 	relocaliser_temp->LoadFromDirectory(relocaliserInputDirectory);

	// 	delete relocaliser;
	// 	relocaliser = relocaliser_temp;
	// }
	// catch (std::runtime_error &e)
	// {
	// 	throw std::runtime_error("Could not load relocaliser: " + std::string(e.what()));
	// }

	try // load scene
	{
		scene->LoadFromDirectory(sceneInputDirectory);
	}
	catch (std::runtime_error &e)
	{
		denseMapper->ResetScene(scene);
		throw std::runtime_error("Could not load scene:" + std::string(e.what()));
	}
}

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::resetAll()
{
	denseMapper->ResetScene(scene);
	trackingState->Reset();
}

#ifdef OUTPUT_TRAJECTORY_QUATERNIONS
static int QuaternionFromRotationMatrix_variant(const double *matrix)
{
	int variant = 0;
	if ((matrix[4] > -matrix[8]) && (matrix[0] > -matrix[4]) && (matrix[0] > -matrix[8]))
	{
		variant = 0;
	}
	else if ((matrix[4] < -matrix[8]) && (matrix[0] > matrix[4]) && (matrix[0] > matrix[8]))
	{
		variant = 1;
	}
	else if ((matrix[4] > matrix[8]) && (matrix[0] < matrix[4]) && (matrix[0] < -matrix[8]))
	{
		variant = 2;
	}
	else if ((matrix[4] <
			  matrix[8]) &&
			 (matrix[0] < -matrix[4]) && (matrix[0] < matrix[8]))
	{
		variant = 3;
	}
	return variant;
}

static void QuaternionFromRotationMatrix(const double *matrix, double *q)
{
	/* taken from "James Diebel. Representing Attitude: Euler
	Angles, Quaternions, and Rotation Vectors. Technical Report, Stanford
	University, Palo Alto, CA."
	*/

	// choose the numerically best variant...
	int variant = QuaternionFromRotationMatrix_variant(matrix);
	double denom = 1.0;
	if (variant == 0)
	{
		denom += matrix[0] + matrix[4] + matrix[8];
	}
	else
	{
		int tmp = variant * 4;
		denom += matrix[tmp - 4];
		denom -= matrix[tmp % 12];
		denom -= matrix[(tmp + 4) % 12];
	}
	denom = sqrt(denom);
	q[variant] = 0.5 * denom;

	denom *= 2.0;
	switch (variant)
	{
	case 0:
		q[1] = (matrix[5] - matrix[7]) / denom;
		q[2] = (matrix[6] - matrix[2]) / denom;
		q[3] = (matrix[1] - matrix[3]) / denom;
		break;
	case 1:
		q[0] = (matrix[5] - matrix[7]) / denom;
		q[2] = (matrix[1] + matrix[3]) / denom;
		q[3] = (matrix[6] + matrix[2]) / denom;
		break;
	case 2:
		q[0] = (matrix[6] - matrix[2]) / denom;
		q[1] = (matrix[1] + matrix[3]) / denom;
		q[3] = (matrix[5] + matrix[7]) / denom;
		break;
	case 3:
		q[0] = (matrix[1] - matrix[3]) / denom;
		q[1] = (matrix[6] + matrix[2]) / denom;
		q[2] = (matrix[5] + matrix[7]) / denom;
		break;
	}

	if (q[0] < 0.0f)
		for (int i = 0; i < 4; ++i)
			q[i] *= -1.0f;
}
#endif

template <typename TVoxel, typename TIndex>
ITMTrackingState::TrackingResult ITMBasicEngine<TVoxel, TIndex>::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage, ITMIMUMeasurement *imuMeasurement)
{
	lastPoseRollback = false;
	lastFusionSuppressed = false;
#ifndef GPS_SLAM_DISABLE_BASIC_CAPACITY_INSTRUMENTATION
    scene->lastFailedVoxelAllocations = 0;
    scene->lastFailedExcessAllocations = 0;
#endif
	// prepare image and turn it into a depth image
	if (imuMeasurement == NULL)
		viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useBilateralFilter);
	else
		viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useBilateralFilter, imuMeasurement);
	const int inputFrameNo = framesProcessed;

	if (!mainProcessingActive)
		return ITMTrackingState::TRACKING_FAILED;

	// tracking
	ORUtils::SE3Pose oldPose(*(trackingState->pose_d));
	const bool oldPoseFinite = basicEngineIsFinitePose(oldPose);
	if (trackingActive)
		trackingController->Track(trackingState, view);
	else
	{
		trackingState->pose_d->SetInvM(*(gtC2wPoses[framesProcessed]));
		trackingState->pose_d->Coerce();
		trackingState->trackerResult = ITMTrackingState::TRACKING_GOOD;
	}
	if (rejectNonFiniteTrackingPose && !basicEngineIsFinitePose(*(trackingState->pose_d)))
	{
		trackingState->trackerResult = ITMTrackingState::TRACKING_FAILED;
		if (oldPoseFinite)
		{
			*(trackingState->pose_d) = oldPose;
			lastPoseRollback = true;
		}
		fprintf(stdout,
			"[BASIC TRACKING] non_finite_pose_reject frame=%d old_pose_finite=%d\n",
			framesProcessed, oldPoseFinite ? 1 : 0);
	}
	logTrackingDiagnostics(inputFrameNo, "post_track_pre_fusion",
		trackingState->trackerResult, false);

	lastRawTrackingResult = trackingState->trackerResult;
	ITMTrackingFailureDecision failureDecision =
		DecideTrackingFailure(trackingState->trackerResult, settings->behaviourOnFailure);
	ITMTrackingState::TrackingResult trackerResult = failureDecision.policyResult;
	if (failureDecision.rollbackPose)
	{
		if (oldPoseFinite)
		{
			*(trackingState->pose_d) = oldPose;
			lastPoseRollback = true;
		}
		else
		{
			trackerResult = ITMTrackingState::TRACKING_FAILED;
			failureDecision.policyResult = trackerResult;
		}
	}
	lastFusionSuppressed = failureDecision.suppressFusion;
	lastPolicyTrackingResult = trackerResult;

	// relocalisation
	int addKeyframeIdx = -1;
	if (settings->behaviourOnFailure == ITMLibSettings::FAILUREMODE_RELOCALISE)
	{
		if (trackerResult == ITMTrackingState::TRACKING_GOOD && relocalisationCount > 0)
			relocalisationCount--;

		int NN;
		float distances;
		view->depth->UpdateHostFromDevice();

		// find and add keyframe, if necessary
		bool hasAddedKeyframe = relocaliser->ProcessFrame(view->depth, trackingState->pose_d, 0, 1, &NN, &distances, trackerResult == ITMTrackingState::TRACKING_GOOD && relocalisationCount == 0);

		// frame not added and tracking failed -> we need to relocalise
		if (!hasAddedKeyframe && trackerResult == ITMTrackingState::TRACKING_FAILED)
		{
			relocalisationCount = 10;

			// Reset previous rgb frame since the rgb image is likely different than the one acquired when setting the keyframe
			view->rgb_prev->Clear();

			const FernRelocLib::PoseDatabase::PoseInScene &keyframe = relocaliser->RetrievePose(NN);
			trackingState->pose_d->SetFrom(&keyframe.pose);

			denseMapper->UpdateVisibleList(view, trackingState, scene, renderState_live, true);
			trackingController->Prepare(trackingState, scene, view, visualisationEngine, renderState_live);
			trackingController->Track(trackingState, view);

			trackerResult = trackingState->trackerResult;
			failureDecision = DecideTrackingFailure(trackerResult, settings->behaviourOnFailure);
			lastFusionSuppressed = failureDecision.suppressFusion;
			lastPolicyTrackingResult = trackerResult;
		}
	}

	// std::cout << "estimated_pose:\n"
	// 		  << trackingState->pose_d->GetInvM() << std::endl;
	bool didFusion = false;
	const bool bootstrapFusionAllowed = !trackingInitialised && !lastFusionSuppressed;
	if ((trackerResult == ITMTrackingState::TRACKING_GOOD || bootstrapFusionAllowed) &&
		(fusionActive) && (relocalisationCount == 0) && !lastFusionSuppressed)
	{
		// fusion
		denseMapper->ProcessFrame(view, trackingState, scene, renderState_live);
		didFusion = true;
		if (framesProcessed > 50)
			trackingInitialised = true;

	}

	if (trackerResult == ITMTrackingState::TRACKING_GOOD || trackerResult == ITMTrackingState::TRACKING_POOR)
	{
		if (!didFusion)
			denseMapper->UpdateVisibleList(view, trackingState, scene, renderState_live);

		// raycast to renderState_live for tracking and free visualisation
		trackingController->Prepare(trackingState, scene, view, visualisationEngine, renderState_live);
		logTrackingDiagnostics(inputFrameNo, "post_prepare", trackerResult, didFusion);

		if (addKeyframeIdx >= 0)
		{
			ORUtils::MemoryBlock<Vector4u>::MemoryCopyDirection memoryCopyDirection =
				settings->deviceType == ITMLibSettings::DEVICE_CUDA ? ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CUDA : ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU;

			kfRaycast->SetFrom(renderState_live->raycastImage, memoryCopyDirection);
		}
	}
	else
		*trackingState->pose_d = oldPose;

#ifdef OUTPUT_TRAJECTORY_QUATERNIONS
	const ORUtils::SE3Pose *p = trackingState->pose_d;
	double t[3];
	double R[9];
	double q[4];
	for (int i = 0; i < 3; ++i)
		t[i] = p->GetInvM().m[3 * 4 + i];
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			R[r * 3 + c] = p->GetM().m[c * 4 + r];
	QuaternionFromRotationMatrix(R, q);
	fprintf(stderr, "%f %f %f %f %f %f %f\n", t[0], t[1], t[2], q[1], q[2], q[3], q[0]);
#endif
	// 记录相机内外参
	camPoses.emplace_back(*(trackingState->pose_d));
	camIntrincs.emplace_back(view->calib.intrinsics_d);
#ifndef GPS_SLAM_DISABLE_BASIC_CAPACITY_INSTRUMENTATION
    logCapacityDiagnostics(inputFrameNo, didFusion);
#endif
	framesProcessed++;
	return trackerResult;
}

template <typename TVoxel, typename TIndex>
Vector2i ITMBasicEngine<TVoxel, TIndex>::GetImageSize(void) const
{
	return renderState_live->raycastImage->noDims;
}

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::GetImage(ITMUChar4Image *out, GetImageType getImageType, ORUtils::SE3Pose *pose, ITMIntrinsics *intrinsics)
{
	if (view == NULL)
		return;

	out->Clear();
	switch (getImageType)
	{
	case ITMBasicEngine::InfiniTAM_IMAGE_ORIGINAL_RGB:
		out->ChangeDims(view->rgb->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			out->SetFrom(view->rgb, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		else
			out->SetFrom(view->rgb, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		break;
	case ITMBasicEngine::InfiniTAM_IMAGE_ORIGINAL_DEPTH:
		out->ChangeDims(view->depth->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			view->depth->UpdateHostFromDevice();
		ITMVisualisationEngine<TVoxel, TIndex>::DepthToUchar4(out, view->depth);

		break;
	case ITMBasicEngine::InfiniTAM_IMAGE_SCENERAYCAST:
	case ITMBasicEngine::InfiniTAM_IMAGE_COLOUR_FROM_VOLUME:
	case ITMBasicEngine::InfiniTAM_IMAGE_COLOUR_FROM_NORMAL:
	case ITMBasicEngine::InfiniTAM_IMAGE_COLOUR_FROM_CONFIDENCE:
	{
		// use current raycast or forward projection?
		IITMVisualisationEngine::RenderRaycastSelection raycastType;
		if (trackingState->age_pointCloud <= 0)
			raycastType = IITMVisualisationEngine::RENDER_FROM_OLD_RAYCAST;
		else
			raycastType = IITMVisualisationEngine::RENDER_FROM_OLD_FORWARDPROJ;

		// what sort of image is it?
		IITMVisualisationEngine::RenderImageType imageType;
		switch (getImageType)
		{
		case ITMBasicEngine::InfiniTAM_IMAGE_COLOUR_FROM_CONFIDENCE:
			imageType = IITMVisualisationEngine::RENDER_COLOUR_FROM_CONFIDENCE;
			break;
		case ITMBasicEngine::InfiniTAM_IMAGE_COLOUR_FROM_NORMAL:
			imageType = IITMVisualisationEngine::RENDER_COLOUR_FROM_NORMAL;
			break;
		case ITMBasicEngine::InfiniTAM_IMAGE_COLOUR_FROM_VOLUME:
			imageType = IITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME;
			break;
		default:
			imageType = IITMVisualisationEngine::RENDER_SHADED_GREYSCALE_IMAGENORMALS;
		}

		visualisationEngine->RenderImage(scene, trackingState->pose_d, &view->calib.intrinsics_d, renderState_live, renderState_live->raycastImage, imageType, raycastType);

		// printf("check raycast results...\n");
		// auto raycast_results = renderState_live->raycastResult;
		// printf("dim: (%d, %d)\n", raycast_results->noDims.x, raycast_results->noDims.y);
		// ITMFloat4Image *out_points = new ITMFloat4Image(raycast_results->noDims, true, false);
		// out_points->SetFrom(raycast_results, ORUtils::MemoryBlock<Vector4f>::CUDA_TO_CPU);
		// Vector4f *pointsRay = out_points->GetData(MEMORYDEVICE_CPU);
		// printf("(%f, %f, %f, %f)\n", pointsRay[0].x, pointsRay[0].y, pointsRay[0].z, pointsRay[0].w);

		ORUtils::Image<Vector4u> *srcImage = NULL;
		if (relocalisationCount != 0)
			srcImage = kfRaycast;
		else
			srcImage = renderState_live->raycastImage;

		out->ChangeDims(srcImage->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			out->SetFrom(srcImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		else
			out->SetFrom(srcImage, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		break;
	}
	case ITMBasicEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED:
	case ITMBasicEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME:
	case ITMBasicEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL:
	case ITMBasicEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_CONFIDENCE:
	{
		IITMVisualisationEngine::RenderImageType type = IITMVisualisationEngine::RENDER_SHADED_GREYSCALE;
		if (getImageType == ITMBasicEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME)
			type = IITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME;
		else if (getImageType == ITMBasicEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL)
			type = IITMVisualisationEngine::RENDER_COLOUR_FROM_NORMAL;
		else if (getImageType == ITMBasicEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_CONFIDENCE)
			type = IITMVisualisationEngine::RENDER_COLOUR_FROM_CONFIDENCE;

		if (renderState_freeview == NULL)
		{
			renderState_freeview = ITMRenderStateFactory<TIndex>::CreateRenderState(out->noDims, scene->sceneParams, settings->GetMemoryType());
		}

		visualisationEngine->FindVisibleBlocks(scene, pose, intrinsics, renderState_freeview);
		visualisationEngine->CreateExpectedDepths(scene, pose, intrinsics, renderState_freeview);
		visualisationEngine->RenderImage(scene, pose, intrinsics, renderState_freeview, renderState_freeview->raycastImage, type);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			out->SetFrom(renderState_freeview->raycastImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		else
			out->SetFrom(renderState_freeview->raycastImage, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		break;
	}
	case ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN:
		break;
	};
}

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::runRaycast(ORUtils::SE3Pose *pose, ITMIntrinsics *intrinsics)
{
	if (pose == nullptr && intrinsics == nullptr)
	{
		if (view == NULL)
			return;
		// use current raycast or forward projection?
		IITMVisualisationEngine::RenderRaycastSelection raycastType;
		if (trackingState->age_pointCloud <= 0)
			raycastType = IITMVisualisationEngine::RENDER_FROM_OLD_RAYCAST;
		else
			raycastType = IITMVisualisationEngine::RENDER_FROM_OLD_FORWARDPROJ;

		// what sort of image is it?
		IITMVisualisationEngine::RenderImageType imageType = IITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME;

		visualisationEngine->RenderImage(scene, trackingState->pose_d, &view->calib.intrinsics_d, renderState_live, renderState_live->raycastImage, imageType, raycastType);
	}
	else
	{
		IITMVisualisationEngine::RenderImageType type = IITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME;
		visualisationEngine->FindVisibleBlocks(scene, pose, intrinsics, renderState_freeview);
		visualisationEngine->CreateExpectedDepths(scene, pose, intrinsics, renderState_freeview);
		visualisationEngine->RenderImage(scene, pose, intrinsics, renderState_freeview, renderState_freeview->raycastImage, type);
	}
}

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::turnOnTracking() { trackingActive = true; }

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::turnOffTracking() { trackingActive = false; }

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::turnOnIntegration() { fusionActive = true; }

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::turnOffIntegration() { fusionActive = false; }

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::turnOnMainProcessing() { mainProcessingActive = true; }

template <typename TVoxel, typename TIndex>
void ITMBasicEngine<TVoxel, TIndex>::turnOffMainProcessing() { mainProcessingActive = false; }

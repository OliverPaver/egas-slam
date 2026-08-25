// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLibSettings.h"
#include "../Objects/Tracking/ITMTrackingState.h"

namespace ITMLib
{
	/** A side-effect-free description of how one tracker result is contained.
	 *
	 * FAILUREMODE_IGNORE intentionally preserves the historical InfiniTAM
	 * behaviour for baseline comparisons. STOP_INTEGRATION keeps the pipeline
	 * alive on a failed frame, but never permits that frame to move or update
	 * the map.
	 */
	struct ITMTrackingFailureDecision
	{
		ITMTrackingState::TrackingResult policyResult;
		bool rollbackPose;
		bool suppressFusion;
	};

	inline ITMTrackingFailureDecision DecideTrackingFailure(
		ITMTrackingState::TrackingResult rawResult,
		ITMLibSettings::FailureMode failureMode)
	{
		if (failureMode == ITMLibSettings::FAILUREMODE_IGNORE)
			return {ITMTrackingState::TRACKING_GOOD, false, false};

		if (failureMode == ITMLibSettings::FAILUREMODE_STOP_INTEGRATION)
		{
			if (rawResult == ITMTrackingState::TRACKING_FAILED)
				return {ITMTrackingState::TRACKING_POOR, true, true};
			if (rawResult == ITMTrackingState::TRACKING_POOR)
				return {ITMTrackingState::TRACKING_POOR, false, true};
			return {ITMTrackingState::TRACKING_GOOD, false, false};
		}

		if (failureMode == ITMLibSettings::FAILUREMODE_CONTAIN_FAILED)
		{
			if (rawResult == ITMTrackingState::TRACKING_FAILED)
				return {ITMTrackingState::TRACKING_POOR, true, true};
			// The extended tracker labels recoverable poses POOR. Preserve the
			// established fusion path for those poses while keeping the raw
			// result visible to diagnostics and submap-boundary policy.
			return {ITMTrackingState::TRACKING_GOOD, false, false};
		}

		// Relocalisation owns its recovery path. Its raw result must remain
		// visible and only good poses may be fused.
		return {rawResult, false, rawResult != ITMTrackingState::TRACKING_GOOD};
	}
}

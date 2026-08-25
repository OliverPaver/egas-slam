#pragma once

#include <functional>
#include <iostream>
#include "InputSource/ImageSourceEngine.h"
#include "ITMLib/Core/ITMMainEngine.h"
#include "ITMLib/Utils/ITMLibSettings.h"
#include "ORUtils/FileUtils.h"
#include "ORUtils/NVTimer.h"

namespace InfiniTAM
{
    namespace Engine
    {
        class CLIEngine
        {
            static CLIEngine *instance;

            // InputSource::ImageSourceEngine *imageSource;
            std::vector<ITMUChar4Image *> rgb_images;
            std::vector<ITMShortImage *> depth_images;
            std::function<bool(int, ITMUChar4Image *, ITMShortImage *)> frame_provider;
            Vector2i rgb_size;
            Vector2i depth_size;
            int frame_count = 0;
            bool on_demand = false;
            // std::vector<ORUtils::SE3Pose> cam_poses;
            // std::vector<ITMLib::ITMIntrinsics> cam_intrincs;
            ITMLib::ITMLibSettings internalSettings;
            ITMLib::ITMMainEngine *mainEngine;
            ITMLib::ITMTrackingState::TrackingResult lastTrackingResult;

            StopWatchInterface *timer_instant;
            StopWatchInterface *timer_average;

        private:
            ITMUChar4Image *inputRGBImage;
            ITMShortImage *inputRawDepthImage;

            int currentFrameNo;

        public:
            static CLIEngine *Instance(void)
            {
                if (instance == NULL)
                    instance = new CLIEngine();
                return instance;
            }

            float processedTime;

            void Initialise(std::vector<ITMUChar4Image *> rgb_images,
                            std::vector<ITMShortImage *> depth_images,
                            ITMLib::ITMMainEngine *mainEngine);
            void InitialiseOnDemand(Vector2i rgb_size, Vector2i depth_size, int frame_count,
                                    std::function<bool(int, ITMUChar4Image *, ITMShortImage *)> frame_provider,
                                    ITMLib::ITMMainEngine *mainEngine);
            void Shutdown();

            void Run();
            bool ProcessFrame();
            ITMLib::ITMTrackingState::TrackingResult GetLastTrackingResult() const
            {
                return lastTrackingResult;
            }

            Vector2i GetDepthSize()
            {
                return depth_size;
            }

            Vector2i GetRGBSize()
            {
                return rgb_size;
            }

            ITMLib::ITMMainEngine *getMainEngine()
            {
                return mainEngine;
            }
        };
    }
}

#include "TsdfFusion/CLIEngine.h"

#include <string.h>

#include "ORUtils/FileUtils.h"

using namespace InfiniTAM::Engine;
using namespace InputSource;
using namespace ITMLib;

CLIEngine *CLIEngine::instance;

void CLIEngine::Initialise(std::vector<ITMUChar4Image *> rgb_images,
                           std::vector<ITMShortImage *> depth_images,
                           ITMMainEngine *mainEngine)
{
    this->rgb_images = rgb_images;
    this->depth_images = depth_images;
    this->rgb_size = rgb_images[0]->noDims;
    this->depth_size = depth_images[0]->noDims;
    this->frame_count = rgb_images.size();
    this->on_demand = false;
    this->mainEngine = mainEngine;
    this->currentFrameNo = 0;
    this->lastTrackingResult = ITMLib::ITMTrackingState::TRACKING_GOOD;

    inputRGBImage = nullptr;
    inputRawDepthImage = nullptr;

    sdkCreateTimer(&timer_instant);
    sdkCreateTimer(&timer_average);
    sdkResetTimer(&timer_average);

    printf("initialised.\n");
}

void CLIEngine::InitialiseOnDemand(Vector2i rgb_size, Vector2i depth_size, int frame_count,
                                   std::function<bool(int, ITMUChar4Image *, ITMShortImage *)> frame_provider,
                                   ITMMainEngine *mainEngine)
{
    this->rgb_images.clear();
    this->depth_images.clear();
    this->rgb_size = rgb_size;
    this->depth_size = depth_size;
    this->frame_count = frame_count;
    this->frame_provider = std::move(frame_provider);
    this->on_demand = true;
    this->mainEngine = mainEngine;
    this->currentFrameNo = 0;
    this->lastTrackingResult = ITMLib::ITMTrackingState::TRACKING_GOOD;
    inputRGBImage = new ITMUChar4Image(rgb_size, true, false);
    inputRawDepthImage = new ITMShortImage(depth_size, true, false);

    sdkCreateTimer(&timer_instant);
    sdkCreateTimer(&timer_average);
    sdkResetTimer(&timer_average);
    printf("initialised on-demand input.\n");
}

bool CLIEngine::ProcessFrame()
{
    if (currentFrameNo >= frame_count)
        return false;
    if (on_demand)
    {
        if (!frame_provider(currentFrameNo, inputRGBImage, inputRawDepthImage))
            return false;
    }
    else
    {
        inputRGBImage = rgb_images[currentFrameNo];
        inputRawDepthImage = depth_images[currentFrameNo];
    }

    sdkResetTimer(&timer_instant);
    sdkStartTimer(&timer_instant);
    sdkStartTimer(&timer_average);
    // actual processing on the mailEngine
    lastTrackingResult = mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage);

    sdkStopTimer(&timer_instant);
    sdkStopTimer(&timer_average);

    float processedTime_inst = sdkGetTimerValue(&timer_instant);
    float processedTime_avg = sdkGetAverageTimerValue(&timer_average);

    // printf("frame %i: time %.2f, avg %.2f\n", currentFrameNo, processedTime_inst, processedTime_avg);

    currentFrameNo++;

    return true;
}

void CLIEngine::Run()
{
    while (true)
    {
        if (!ProcessFrame())
            break;
    }
}

void CLIEngine::Shutdown()
{
    sdkDeleteTimer(&timer_instant);
    sdkDeleteTimer(&timer_average);

    if (on_demand)
    {
        delete inputRGBImage;
        delete inputRawDepthImage;
    }
    delete instance;
}

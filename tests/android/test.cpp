#include <android/log.h>
#include <android_native_app_glue.h>
#include <sstream>
#include "isaac/array.h"
#include "isaac/tools/to_string.hpp"

void android_main(struct android_app* state)
{
    app_dummy(); // Make sure glue isn't stripped
    __android_log_print(ANDROID_LOG_INFO, "IsaacAndroidTest", "This is a test");
    isaac::array test(5, isaac::FLOAT_TYPE);
    __android_log_print(ANDROID_LOG_INFO, "IsaacAndroidTest", isaac::tools::to_string(test).c_str());
    ANativeActivity_finish(state->activity);
}

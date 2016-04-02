#include <android/log.h>
#include <android_native_app_glue.h>
#include <sstream>
#include "isaac/array.h"

void android_main(struct android_app* state)
{
    app_dummy(); // Make sure glue isn't stripped
    __android_log_print(ANDROID_LOG_INFO, "IsaacAndroidTest", "This is a test");
    isaac::array test(5, isaac::FLOAT_TYPE);
    std::ostringstream oss;
    oss << test;
    __android_log_print(ANDROID_LOG_INFO, "IsaacAndroidTest", oss.str().c_str());
    ANativeActivity_finish(state->activity);
}


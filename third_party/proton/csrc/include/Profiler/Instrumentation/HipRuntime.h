#include "Runtime.h"

namespace proton {

class HipRuntime : public Runtime {
 public:
	HipRuntime() : Runtime(DeviceType::HIP) {}
	~HipRuntime() override = default;

	void allocateHostBuffer(uint8_t **buffer) override;
	void freeHostBuffer(uint8_t *buffer) override;
	void *getPriorityStream() override;
	void processHostBuffer(uint8_t *hostBuffer, size_t hostBufferSize,
												 const uint8_t *deviceBuffer, size_t deviceBufferSize,
												 void *stream,
												 std::function<void(uint8_t *, size_t)> callback) override;
};

} // namespace proton
#ifndef PROTON_DRIVER_GPU_ROCTRACER_H_
#define PROTON_DRIVER_GPU_ROCTRACER_H_

#include <roctracer/roctracer.h>

namespace proton {

namespace roctracer {

template <bool CheckSuccess>
roctracer_status_t set_properties(roctracer_domain_t domain, void *properties);

template <bool CheckSuccess>
roctracer_status_t get_timestamp(roctracer_timestamp_t *timestamp);

void start();

void stop();

//
// Callbacks
//

template <bool CheckSuccess>
roctracer_status_t enable_domain_callback(activity_domain_t domain,
                                          activity_rtapi_callback_t callback,
                                          void *arg);

template <bool CheckSuccess>
roctracer_status_t disable_domain_callback(activity_domain_t domain);

template <bool CheckSuccess>
roctracer_status_t enable_op_callback(activity_domain_t domain, uint32_t op,
                                      activity_rtapi_callback_t callback,
                                      void *arg);

template <bool CheckSuccess>
roctracer_status_t disable_op_callback(activity_domain_t domain, uint32_t op);

//
// Activity
//

template <bool CheckSuccess>
roctracer_status_t open_pool(const roctracer_properties_t *properties);

template <bool CheckSuccess> roctracer_status_t close_pool();

template <bool CheckSuccess>
roctracer_status_t enable_op_activity(activity_domain_t domain, uint32_t op);

template <bool CheckSuccess>
roctracer_status_t enable_domain_activity(activity_domain_t domain);

template <bool CheckSuccess>
roctracer_status_t disable_op_activity(activity_domain_t domain, uint32_t op);

template <bool CheckSuccess>
roctracer_status_t disable_domain_activity(activity_domain_t domain);

template <bool CheckSuccess> roctracer_status_t flush_activity();

template <bool CheckSuccess>
roctracer_status_t next_record(const activity_record_t *record,
                               const activity_record_t **next);

char *op_string(uint32_t domain, uint32_t op, uint32_t kind);

//
// External correlation
//

template <bool CheckSuccess>
roctracer_status_t
activity_push_external_correlation_id(activity_correlation_id_t id);

template <bool CheckSuccess>
roctracer_status_t
activity_pop_external_correlation_id(activity_correlation_id_t *last_id);

} // namespace roctracer

} // namespace proton

#endif // PROTON_EXTERN_DISPATCH_H_

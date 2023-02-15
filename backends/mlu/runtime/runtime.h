// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifdef _POSIX_C_SOURCE
#include <time.h>
#endif
#include <cncl.h>
#include <cnnl.h>
#include <cnpapi.h>
#include <cnrt.h>
#include <mlu_op.h>

#include "glog/logging.h"
#include "paddle/phi/backends/device_base.h"
#include "paddle/phi/extension.h"
#include "runtime/cnpapi_data_process.h"

template <typename T>
struct CustomMLUStatusType {};

#define DEFINE_CUSTOM_MLU_STATUS_TYPE(type, success_value) \
  template <>                                              \
  struct CustomMLUStatusType<type> {                       \
    using Type = type;                                     \
    static constexpr Type kSuccess = success_value;        \
  }
#define ENV_Cat(x, y) x##y
#define ENV_Str(x) #x
#define ENV_Call(x, y) x(y)
#define ENV_DEFINE(type, name, value, parser)                        \
  type FLAGS_##name =                                                \
      getenv(ENV_Call(ENV_Str, ENV_Cat(FLAGS_, name)))               \
          ? parser(getenv(ENV_Call(ENV_Str, ENV_Cat(FLAGS_, name)))) \
          : value
#define ENV_uint64(x, value) ENV_DEFINE(uint64_t, x, value, std::stoul)
#define ENV_string(x, value) ENV_DEFINE(std::string, x, value, std::string)

#define CNPAPI_CALL(call)                                                    \
  do {                                                                       \
    cnpapiResult _status = call;                                             \
    if (_status != CNPAPI_SUCCESS) {                                         \
      const char *errstr;                                                    \
      cnpapiGetResultString(_status, &errstr);                               \
      LOG(ERROR) << "Function " << #call << " failed with error " << errstr; \
    }                                                                        \
  } while (0)

DEFINE_CUSTOM_MLU_STATUS_TYPE(cnrtRet_t, cnrtSuccess);
DEFINE_CUSTOM_MLU_STATUS_TYPE(cnnlStatus_t, CNNL_STATUS_SUCCESS);
DEFINE_CUSTOM_MLU_STATUS_TYPE(mluOpStatus_t, MLUOP_STATUS_SUCCESS);
DEFINE_CUSTOM_MLU_STATUS_TYPE(cnclResult_t, CNCL_RET_SUCCESS);

/*************** CNRT ERROR ***************/
inline bool is_error(cnrtRet_t e) { return e != cnrtSuccess; }

inline std::string build_mlu_error_msg(cnrtRet_t e) {
  std::ostringstream sout;
  sout << "MLU CNRT error(" << e << "), " << cnrtGetErrorName(e) << ": "
       << cnrtGetErrorStr(e);
  return sout.str();
}

/*************** CNNL ERROR ***************/
inline bool is_error(cnnlStatus_t stat) { return stat != CNNL_STATUS_SUCCESS; }

inline std::string build_mlu_error_msg(cnnlStatus_t stat) {
  std::ostringstream sout;
  sout << "MLU CNNL error(" << stat << "), " << cnnlGetErrorString(stat)
       << ". ";
  return sout.str();
}

/*************** MLUOP ERROR ***************/
inline bool is_error(mluOpStatus_t stat) {
  return stat != MLUOP_STATUS_SUCCESS;
}

inline std::string build_mlu_error_msg(mluOpStatus_t stat) {
  std::ostringstream sout;
  sout << "MLU OP error(" << stat << "), " << mluOpGetErrorString(stat) << ". ";
  return sout.str();
}

/*************** CNCL ERROR ***************/
inline bool is_error(cnclResult_t e) { return e != CNCL_RET_SUCCESS; }

inline std::string build_mlu_error_msg(cnclResult_t e) {
  std::ostringstream sout;
  sout << "MLU CNCL error(" << e << "), " << cnclGetErrorStr(e) << ". ";
  return sout.str();
}

#define PADDLE_ENFORCE_MLU_SUCCESS(COND)                    \
  do {                                                      \
    auto __cond__ = (COND);                                 \
    using __MLU_STATUS_TYPE__ = decltype(__cond__);         \
    constexpr auto __success_type__ =                       \
        CustomMLUStatusType<__MLU_STATUS_TYPE__>::kSuccess; \
    if (UNLIKELY(__cond__ != __success_type__)) {           \
      auto __summary__ = build_mlu_error_msg(__cond__);     \
      __THROW_ERROR_INTERNAL__(__summary__);                \
    }                                                       \
  } while (0)

struct CustomMLUStream {
  cnnlHandle_t handle;
  mluOpHandle_t op_handle;
  cnrtQueue_t queue;
};
typedef CustomMLUStream *mluStream_t;

inline cnnlHandle_t GetHandle(const C_Stream stream) {
  return reinterpret_cast<mluStream_t>(stream)->handle;
}
inline cnnlHandle_t GetHandle(void *stream) {
  return reinterpret_cast<mluStream_t>(stream)->handle;
}

inline mluOpHandle_t GetOpHandle(const C_Stream stream) {
  return reinterpret_cast<mluStream_t>(stream)->op_handle;
}
inline mluOpHandle_t GetOpHandle(void *stream) {
  return reinterpret_cast<mluStream_t>(stream)->op_handle;
}

inline cnrtQueue_t GetQueue(const C_Stream stream) {
  return reinterpret_cast<mluStream_t>(stream)->queue;
}
inline cnrtQueue_t GetQueue(void *stream) {
  return reinterpret_cast<mluStream_t>(stream)->queue;
}

C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size);
C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size);
C_Status MemCpyD2H(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size);
C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);
C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);
C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);

// Get system-wide realtime clock in nanoseconds
inline uint64_t PosixInNsec() {
#ifdef _POSIX_C_SOURCE
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return tp.tv_sec * 1000 * 1000 * 1000 + tp.tv_nsec;
#else
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return 1000 * (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
#endif
}

class MLUProfiler {
 public:
  static MLUProfiler &Instance() {
    static MLUProfiler instance;
    return instance;
  }

  MLUProfiler();

  ~MLUProfiler() {
    // destroy_step_info();
    // destroy_config();
  }

  void update_config(std::vector<uint32_t> device_ids) {
    devices_ids_ = device_ids;
    // metrics_ = metrics;
    // data_type_ = type;
  }

  void Prepare() { EnableCnpapiActivity(); }

  // void destroy_config() {
  //   if (config_) {
  //     ACL_CHECK(aclprofDestroyConfig(config_));
  //     config_ = nullptr;
  //   }
  // }

  // aclprofConfig *get_config() {
  //   if (config_ == nullptr) {
  //     config_ = aclprofCreateConfig(devices_ids_.data(),
  //                                   devices_ids_.size(),
  //                                   metrics_,
  //                                   nullptr,
  //                                   data_type_);
  //   }
  //   return config_;
  // }

  // aclprofStepInfo *get_step_info() {
  //   if (step_info_ == nullptr) {
  //     step_info_ = aclprofCreateStepInfo();
  //   }
  //   return step_info_;
  // }

  // void destroy_step_info() {
  //   if (step_info_) {
  //     aclprofDestroyStepInfo(step_info_);
  //     step_info_ = nullptr;
  //   }
  // }

  // void update_stream(aclrtStream stream) {
  //   if (stream_ == nullptr) {
  //     stream_ = stream;
  //     if (step_info_) {
  //       ACL_CHECK(aclprofGetStepTimestamp(step_info_, ACL_STEP_START,
  //       stream_));
  //     }
  //   }
  // }

  // aclrtStream get_stream() { return stream_; }

  // void clear_stream() { stream_ = nullptr; }

  void EnableCnpapiActivity() {
    CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_KERNEL));
    CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
    CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY_PTOP));
    CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMSET));
    CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_CNDRV_API));
    VLOG(3) << "enable cnpapi activity";
  }

  void DisableCnpapiActivity() {
    CNPAPI_CALL(cnpapiActivityFlushAll());
    CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_KERNEL));
    CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
    CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY_PTOP));
    CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMSET));
    CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_CNDRV_API));
    VLOG(3) << "disable cnpapi activity";
  }

  void start() {
    if (!start_) {
      tracing_start_ns_ = PosixInNsec();
      // ACL_CHECK(aclrtSynchronizeDevice());
      // ACL_CHECK(aclprofStart(MLUProfiler::Instance().get_config()));
      start_ = true;
    }
  }

  void stop() {
    if (start_) {
      DisableCnpapiActivity();
      // state_ = TracerState::STOPED;
      //     ACL_CHECK(aclrtSynchronizeDevice());
      //     ACL_CHECK(aclprofStop(MLUProfiler::Instance().get_config()));
      start_ = false;
    }
  }

  void ProcessCnpapiActivity(uint64_t *buffer, size_t valid_size) {
    cnpapiActivity *record = nullptr;
    while (true) {
      cnpapiResult status =
          cnpapiActivityGetNextRecord(buffer, valid_size, &record);
      if (status == CNPAPI_SUCCESS) {
        details::ProcessCnpapiActivityRecord(
            record, tracing_start_ns_, collector_);
      } else if (status == CNPAPI_ERROR_INSUFFICIENT_MEMORY ||
                 status == CNPAPI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        CNPAPI_CALL(status);
      }
    }
  }

  // void step_start() {
  //   get_step_info();
  //   if (stream_ && step_info_) {
  //     ACL_CHECK(aclprofGetStepTimestamp(step_info_, ACL_STEP_START,
  //     stream_));
  //   }
  // }

  // void step_stop() {
  //   if (stream_ && step_info_) {
  //     ACL_CHECK(aclprofGetStepTimestamp(step_info_, ACL_STEP_END, stream_));
  //   }
  //   destroy_step_info();
  // }

 private:
  uint64_t tracing_start_ns_ = UINT64_MAX;

  paddle::platform::TraceEventCollector *collector_;
  std::vector<uint32_t> devices_ids_;
  // aclprofAicoreMetrics metrics_;
  // uint64_t data_type_;

  // aclprofConfig *config_ = nullptr;
  // aclprofStepInfo *step_info_ = nullptr;
  // aclrtStream stream_ = nullptr;

  bool start_ = false;
};

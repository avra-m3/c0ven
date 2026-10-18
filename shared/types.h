// Fixed-width types with one spelling in C++20 and OpenCL C 3.0, for the headers both compile (ADR-0024, item 5).
#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

#ifdef __OPENCL_C_VERSION__
typedef uchar u8;
typedef char i8;
typedef ushort u16;
typedef short i16;
typedef uint u32;
typedef int i32;
typedef ulong u64;
typedef long i64;
#define SHARED_FN static inline
#define SHARED_ASSERT(cond, msg)
#else
#include <cstdint>
typedef uint8_t u8;
typedef int8_t i8;
typedef uint16_t u16;
typedef int16_t i16;
typedef uint32_t u32;
typedef int32_t i32;
typedef uint64_t u64;
typedef int64_t i64;
#define SHARED_FN static inline constexpr
#define SHARED_ASSERT(cond, msg) static_assert(cond, msg)
#endif

#endif

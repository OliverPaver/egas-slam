// Capacity-only experimental controls. The historical default remains unchanged.
#pragma once

#define SDF_BLOCK_SIZE 8
#define SDF_BLOCK_SIZE3 512

#if defined(GPS_SLAM_COMPACT_HASH)
#define SDF_LOCAL_BLOCK_NUM 0x10000
#define SDF_BUCKET_NUM 0x40000
#define SDF_EXCESS_LIST_SIZE 0x8000
#define GPS_SLAM_COMPILED_HASH_PROFILE "compact_64k"
#elif defined(GPS_SLAM_MEDIUM_HASH)
#define SDF_LOCAL_BLOCK_NUM 0x20000
#define SDF_BUCKET_NUM 0x80000
#define SDF_EXCESS_LIST_SIZE 0x10000
#define GPS_SLAM_COMPILED_HASH_PROFILE "medium_128k"
#elif defined(GPS_SLAM_CAPACITY_CONTROL)
#if GPS_SLAM_CAPACITY_CONTROL == 1
#define SDF_LOCAL_BLOCK_NUM 0x80000
#define SDF_BUCKET_NUM 0x100000
#define SDF_EXCESS_LIST_SIZE 0x20000
#define GPS_SLAM_COMPILED_HASH_PROFILE "legacy_pool_512k"
#elif GPS_SLAM_CAPACITY_CONTROL == 2
#define SDF_LOCAL_BLOCK_NUM 0x100000
#define SDF_BUCKET_NUM 0x400000
#define SDF_EXCESS_LIST_SIZE 0x80000
#define GPS_SLAM_COMPILED_HASH_PROFILE "legacy_pool_1m"
#elif GPS_SLAM_CAPACITY_CONTROL == 3
#define SDF_LOCAL_BLOCK_NUM 0x40000
#define SDF_BUCKET_NUM 0x400000
#define SDF_EXCESS_LIST_SIZE 0x80000
#define GPS_SLAM_COMPILED_HASH_PROFILE "legacy_256k_hash4m"
#elif GPS_SLAM_CAPACITY_CONTROL == 4
#define SDF_LOCAL_BLOCK_NUM 0x200000
#define SDF_BUCKET_NUM 0x800000
#define SDF_EXCESS_LIST_SIZE 0x100000
#define GPS_SLAM_COMPILED_HASH_PROFILE "legacy_pool_2m"
#elif GPS_SLAM_CAPACITY_CONTROL == 5
#define SDF_LOCAL_BLOCK_NUM 0x40000
#define SDF_BUCKET_NUM 0x800000
#define SDF_EXCESS_LIST_SIZE 0x100000
#define GPS_SLAM_COMPILED_HASH_PROFILE "legacy_256k_hash8m"
#else
#error Unknown GPS_SLAM_CAPACITY_CONTROL
#endif
#else
#define SDF_LOCAL_BLOCK_NUM 0x40000
#define SDF_BUCKET_NUM 0x100000
#define SDF_EXCESS_LIST_SIZE 0x20000
#define GPS_SLAM_COMPILED_HASH_PROFILE "legacy_large"
#endif

#define SDF_HASH_MASK (SDF_BUCKET_NUM - 1)

#if (defined(GPS_SLAM_COMPACT_HASH) + defined(GPS_SLAM_MEDIUM_HASH) + defined(GPS_SLAM_CAPACITY_CONTROL)) > 1
#error Select exactly one TSDF capacity profile
#endif
#if SDF_LOCAL_BLOCK_NUM <= 0 || SDF_BUCKET_NUM <= SDF_LOCAL_BLOCK_NUM || SDF_EXCESS_LIST_SIZE <= 0
#error Invalid TSDF capacity relationship
#endif
#if (SDF_BUCKET_NUM & (SDF_BUCKET_NUM - 1)) != 0
#error TSDF bucket count must be a power of two
#endif
// ITMLocalVBA::allocatedSize and CUDA voxel indices are signed 32-bit integers.
// In particular, a four-million-block pool would overflow at 2^31 voxels.
#if 1LL * SDF_LOCAL_BLOCK_NUM * SDF_BLOCK_SIZE3 > 2147483647LL
#error TSDF voxel pool exceeds signed 32-bit indexing
#endif
#if 1LL * SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE > 2147483647LL
#error TSDF hash table exceeds signed 32-bit indexing
#endif

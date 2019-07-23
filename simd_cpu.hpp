#pragma once
#include <vector>
#include <memory>

#include "simd_array.hpp"
#include "simd_array_avx512.hpp"

#include <thread>
#include "sync_line.hpp"

namespace simd
{
	namespace tag
	{
		struct Auto {};    // Default vectorization: #pragma omp simd
		struct Avx512 {};  // Force AVX512 via intrinsics
	}

	namespace cpu
	{
		static const int threads_auto = 0;

		template<class Scalar, int Z, class SIMDTag> class SIMDArray                         : public AlignedArray<Scalar, Z> {};
		template<class Scalar, int Z>                class SIMDArray<Scalar, Z, tag::Avx512> : public AlignedArrayAVX512<Scalar, Z> {};

		template<template<class DataBatch> typename Algorithm, int Z, class Scalar, int THREADS, int RO = 64, class SIMDTag = tag::Auto> class Dispatcher
		{
			using ThreadBatch = typename SIMDArray<Scalar, RO, SIMDTag>;
			std::vector<std::thread> workers;

			SyncLine<THREADS> mBarrier;

		public:

			class ExecutorMaster
			{
				Dispatcher *pOwner;
			public:
				ExecutorMaster(Dispatcher* pO) : pOwner(pO) {}

				template<class F> void SyncAndRun(const F& func)
				{
					pOwner->mBarrier.WaitMaster();
					func();
					pOwner->mBarrier.ReleaseMaster();
				}
			};

			class ExecutorSlave
			{
				Dispatcher *pOwner;
			public:
				ExecutorSlave(Dispatcher* pO) : pOwner(pO) {}

				template<class F> void SyncAndRun(const F& func)
				{
					pOwner->mBarrier.WaitSlave();
				}
			};

		public:
			Dispatcher()
			{}

			void RunWorker(int t)
			{
				for (int i = 0; i < (Z / (THREADS*RO)); ++i)
				{
					auto alg = std::make_unique<Algorithm<ThreadBatch>>();
					//Algorithm<ThreadBatch> alg;
					if (t == 0)
					{
						ExecutorMaster em{ this };
						(*alg)(&em);
					}
					else
					{
						ExecutorSlave es{ this };
						(*alg)(&es);
					}
				}
			}

			void Run()
			{
				for (int t = 0; t < THREADS; ++t)
				{
					workers.emplace_back(&Dispatcher::RunWorker, this, t);
				}

				for (auto& w : workers)
				{
					w.join();
				}

				workers.clear();
			}
		};
	}

}
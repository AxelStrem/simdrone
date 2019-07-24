#pragma once
#include <vector>
#include <memory>

#include "simd_array.hpp"
#include "simd_array_avx512.hpp"

#include <thread>
#include "sync_line.hpp"

#include <functional>
#include <type_traits>

namespace simd
{
	namespace tag
	{
		struct Auto {};    // Default vectorization: #pragma omp simd
		struct Avx512 {};  // Force AVX512 via intrinsics
	}

	class Step_Parallel {};
	class Step_Singlethreaded {};
	class Step_Accumulate {};
	
	template<int STEP, class Tag = Step_Parallel> class StepTag{};

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

			std::vector<std::function<int(int t, std::vector<Algorithm<ThreadBatch>>& alg)>> mSteps;

			class Accumulator
			{
			public:
				void reduce(Scalar* dst, ThreadBatch* src)
				{

				}
			};
		public:

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Parallel>{}))
				RunStep(int t, std::vector<Algorithm<ThreadBatch>>& alg)
			{
				int res = 0;
				for (int i = 0; i < (Z / (THREADS*RO)); ++i)
				{
					res = alg[i](StepTag<STEP, Step_Parallel>{});
				}

				return res;
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Singlethreaded>{}))
				RunStep(int t, std::vector<Algorithm<ThreadBatch>>& alg)
			{
				int res = 0;
				if (t == 0)
				{
					mBarrier.WaitMaster();
					res = alg[0](StepTag<STEP, Step_Singlethreaded>{});
					mBarrier.ReleaseMaster();
				}
				else
				{
					mBarrier.WaitSlave();
				}
				return res;
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Accumulate>{}, (Accumulator*)nullptr))
				RunStep(int t, std::vector<Algorithm<ThreadBatch>>& alg)
			{
				int res = 0;
				for (int i = 0; i < (Z / (THREADS*RO)); ++i)
				{
					res = alg[i](StepTag<STEP, Step_Accumulate>{}, nullptr);
				}

				return res;
			}

			void RunWorker(int t)
			{
				std::vector<Algorithm<ThreadBatch>> alg((Z / (THREADS*RO)));
				int step = 0;
				while (step >= 0)
				{
					step = mSteps[step](t, alg);
				}
			}

			template<int step> void FillSteps()
			{
				mSteps.push_back([this](int t, std::vector<Algorithm<ThreadBatch>>& alg)->int { return RunStep<step>(t, alg); });
				FillSteps<step + 1>();
			}

			template<> void FillSteps<Algorithm<ThreadBatch>::MaxStep + 1>()
			{}

		public:
			Dispatcher()
			{
				FillSteps<0>();
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
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

		template<template<class DataBatch> typename Algorithm, int Z, class Scalar, int THREADS, template<class DataBatch> typename AlgorithmPrimary = Algorithm, int RO = 64, class SIMDTag = tag::Auto> class Dispatcher
		{
			using ThreadBatch = typename SIMDArray<Scalar, RO, SIMDTag>;
			std::vector<std::thread> workers;

			SyncLine<THREADS> mBarrier;


			struct SlaveSet
			{
				Algorithm<ThreadBatch> alg[Z / (THREADS*RO)];
			};

			struct MasterSet
			{
				Algorithm<ThreadBatch> alg[(Z / (THREADS*RO)) - 1];
				AlgorithmPrimary<ThreadBatch> alg_master;
			};

			std::vector<std::function<int(int t, SlaveSet*)>> mStepsS;
			std::vector<std::function<int(int t, MasterSet*)>> mStepsM;
			
			class Accumulator
			{
			public:
				void reduce(Scalar* dst, ThreadBatch* src)
				{

				}
			};
		public:

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Parallel>{}))
				RunStep(int t, SlaveSet* set)
			{
				int res = 0;
				for (auto& instance : set->alg)
				{
					res = instance(StepTag<STEP, Step_Parallel>{});
				}
				return res;
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Parallel>{}))
				RunStep(int t, MasterSet* set)
			{
				for (auto& instance : set->alg)
				{
					instance(StepTag<STEP, Step_Parallel>{});
				}
				return set->alg_master(StepTag<STEP, Step_Parallel>{});
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Singlethreaded>{}))
				RunStep(int t, SlaveSet* set)
			{
				mBarrier.WaitSlave();
				return 0;
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Singlethreaded>{}))
				RunStep(int t, MasterSet* set)
			{
				int res = 0;
				mBarrier.WaitMaster();
				res = set->alg[0](StepTag<STEP, Step_Singlethreaded>{});
				mBarrier.ReleaseMaster();
				return res;
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Accumulate>{}, (Accumulator*)nullptr))
				RunStep(int t, SlaveSet* set)
			{
				int res = 0;
				for (int i = 0; i < (Z / (THREADS*RO)); ++i)
				{
					res = set->alg[i](StepTag<STEP, Step_Accumulate>{}, nullptr);
				}

				return res;
			}

			void RunWorkerS(int t)
			{
				auto alg = std::make_unique<SlaveSet>();
				int step = 0;
				while (step >= 0)
				{
					step = mStepsS[step](t, alg.get());
				}
			}

			void RunWorkerM(int t)
			{
				auto alg = std::make_unique<MasterSet>();
				int step = 0;
				while (step >= 0)
				{
					step = mStepsM[step](t, alg.get());
				}
			}

			template<int step> void FillSteps()
			{
				mStepsS.push_back([this](int t, SlaveSet* alg)->int { return RunStep<step>(t, alg); });
				mStepsM.push_back([this](int t, MasterSet* alg)->int { return RunStep<step>(t, alg); });

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
				workers.emplace_back(&Dispatcher::RunWorkerM, this, 0);

				for (int t = 1; t < THREADS; ++t)
				{
					workers.emplace_back(&Dispatcher::RunWorkerS, this, t);
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
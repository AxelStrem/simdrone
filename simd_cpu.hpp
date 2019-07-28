#pragma once
#include <vector>
#include <memory>

#include "simd_array.hpp"
#include "simd_array_avx512.hpp"

#include <thread>
#include "sync_line.hpp"

#include <functional>
#include <type_traits>
#include <array>

namespace simd
{
	template<class T> using int_t = int;

	namespace tag
	{
		struct Auto {};    // Default vectorization: #pragma omp simd
		struct Avx512 {};  // Force AVX512 via intrinsics
	}

	class Step_Parallel {};
	class Step_Singlethreaded {};
	class Step_Accumulate {};

	template<class DataBatch> struct Fold
	{
		int next_step;
		DataBatch*                         merge_source;
		typename DataBatch::ScalarType*    merge_target;
	};
	
	template<int STEP, class Tag = Step_Parallel> class StepTag{};

	namespace cpu
	{
		static const int threads_auto = 0;

		template<class Scalar, int Z, class SIMDTag> class SIMDArray                         : public AlignedArray<Scalar, Z> {};
		template<class Scalar, int Z>                class SIMDArray<Scalar, Z, tag::Avx512> : public AlignedArrayAVX512<Scalar, Z> {};

		template<template<class DataBatch> typename Algorithm, int Z, class Scalar, int THREADS, template<class DataBatch> typename AlgorithmPrimary = Algorithm, int RO = 64, class SIMDTag = tag::Auto> class Dispatcher
		{
			using ThreadBatch = typename SIMDArray<Scalar, RO, SIMDTag>;
			using SharedData = typename Algorithm<ThreadBatch>::Shared;

			std::vector<std::thread> workers;
			SharedData mShared;

			SyncLine<THREADS> mBarrier;
			std::array<ThreadBatch*, THREADS> merge_pointers;
			int master_res = 0;

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

			template<int STEP> int_t<decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Parallel>{}))> RunStep(int t, SlaveSet* set)
			{
				using ret_type = decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Parallel>{}));
				if constexpr (std::is_same<ret_type, Fold<ThreadBatch>>::value)
				{
					ret_type res = set->alg[0](StepTag<STEP, Step_Parallel>{});
					for (int i = 1; i < (Z / (THREADS*RO)); ++i)
					{
						*(res.merge_source) += *(set->alg[i](StepTag<STEP, Step_Parallel>{}).merge_source);
					}

					merge_pointers[t] = res.merge_source;
					mBarrier.WaitSlave();
					return res.next_step;
				}
				else
				{
					ret_type res = 0;
					for (auto& instance : set->alg)
					{
						res = instance(StepTag<STEP, Step_Parallel>{});
					}
					return res;
				}
			}

			template<int STEP> int_t<decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Parallel>{}))> RunStep(int t, MasterSet* set)
			{
				using ret_type = decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Parallel>{}));
				if constexpr (std::is_same<ret_type, Fold<ThreadBatch>>::value)
				{
					ret_type res = set->alg[0](StepTag<STEP, Step_Parallel>{});
					for (int i = 1; i < (Z / (THREADS*RO)); ++i)
					{
						*(res.merge_source) += *(set->alg[i](StepTag<STEP, Step_Parallel>{}).merge_source);
					}

					mBarrier.WaitMaster();

					for (int tn = 1; tn < THREADS; tn++)
					{
						*(res.merge_source) += *(merge_pointers[tn]);
					}

					*(res.merge_target) = res.merge_source->fold();

					mBarrier.ReleaseMaster();

					return res.next_step;
				}
				else
				{
					for (auto& instance : set->alg)
					{
						instance(StepTag<STEP, Step_Parallel>{});
					}
					return set->alg_master(StepTag<STEP, Step_Parallel>{});
				}
			}
			
			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Singlethreaded>{}))
				RunStep(int t, SlaveSet* set)
			{
				mBarrier.WaitSlave();
				return master_res;
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Singlethreaded>{}))
				RunStep(int t, MasterSet* set)
			{
				mBarrier.WaitMaster();
				master_res = set->alg_master(StepTag<STEP, Step_Singlethreaded>{});
				mBarrier.ReleaseMaster();
				return master_res;
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

				for (auto& a : alg->alg)
					a.init(&mShared);

				int step = 0;
				while (step >= 0)
				{
					step = mStepsS[step](t, alg.get());
				}
			}

			void RunWorkerM(int t)
			{
				auto alg = std::make_unique<MasterSet>();

				for (auto& a : alg->alg)
					a.init(&mShared);
				alg->alg_master.init(&mShared);

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
#pragma once
#include <vector>
#include <memory>

#include "simd.hpp"
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

	namespace cpu
	{
		static const int threads_auto = 0;

		template<class Scalar, int Z, class SIMDTag> struct SIMDArray
		{	using type = AlignedArray<Scalar, Z>;	};
		template<class Scalar, int Z> struct SIMDArray<Scalar, Z, tag::Avx512>
		{    using type = AlignedArrayAVX512<Scalar, Z>;	};

		template<template<class DataBatch> typename Algorithm, int Z, class Scalar, int THREADS, template<class DataBatch> typename AlgorithmPrimary = Algorithm, int RO = 64, class SIMDTag = tag::Auto> class Dispatcher
		{
			using ThreadBatch = typename SIMDArray<Scalar, RO, SIMDTag>::type;
			using SharedData  = typename Algorithm<ThreadBatch>::Shared;
			using Accumulator = typename Algorithm<ThreadBatch>::Accumulator;

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

			std::vector<std::function<int(int t, SlaveSet*, Accumulator*)>> mStepsS;
			std::vector<std::function<int(int t, MasterSet*, Accumulator*)>> mStepsM;
		public:

			template<int STEP> int_t<decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Parallel>{}))> RunStep(int t, SlaveSet* set, Accumulator* acc)
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
					if constexpr (std::is_same<ret_type, FoldAcc<ThreadBatch>>::value)
					{
						ret_type res = set->alg[0](StepTag<STEP, Step_Parallel>{});
						for (int i = 1; i < (Z / (THREADS*RO)); ++i)
						{
							res = set->alg[i](StepTag<STEP, Step_Parallel>{});
						}

						merge_pointers[t] = res.merge_source;
						mBarrier.WaitSlave();
						return res.next_step;
					}
					else
					{
						if constexpr (std::is_base_of<FoldMultiTag, ret_type>::value)
						{
							ret_type res = set->alg[0](StepTag<STEP, Step_Parallel>{});
							for (int i = 1; i < (Z / (THREADS*RO)); ++i)
							{
								res = set->alg[i](StepTag<STEP, Step_Parallel>{});
							}

							merge_pointers[t] = reinterpret_cast<ThreadBatch*>(res.merge_source);
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
				}
			}

			template<int STEP> int_t<decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Parallel>{}))> RunStep(int t, MasterSet* set, Accumulator* acc)
			{
				using ret_type = decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Parallel>{}));
				if constexpr (std::is_same<ret_type, Fold<ThreadBatch>>::value)
				{
					ret_type res = set->alg_master(StepTag<STEP, Step_Parallel>{});
					for (int i = 0; i < (Z / (THREADS*RO)) - 1; ++i)
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
					if constexpr (std::is_same<ret_type, FoldAcc<ThreadBatch>>::value)
					{
						ret_type res = set->alg_master(StepTag<STEP, Step_Parallel>{});
						for (int i = 0; i < (Z / (THREADS*RO)) - 1; ++i)
						{
							set->alg[i](StepTag<STEP, Step_Parallel>{});
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
						if constexpr (std::is_base_of<FoldMultiTag, ret_type>::value)
						{
							ret_type res = set->alg_master(StepTag<STEP, Step_Parallel>{});
							for (int i = 0; i < (Z / (THREADS*RO)) - 1; ++i)
							{
								set->alg[i](StepTag<STEP, Step_Parallel>{});
							}

							mBarrier.WaitMaster();

							for (int tn = 1; tn < THREADS; tn++)
							{
								auto mp = reinterpret_cast<decltype(res.merge_source)>(merge_pointers[tn]);

								//sum_gradients(res.merge_source, mp);
								traverse_accums(res.merge_source, mp, [](auto& lhs, const auto& rhs) { lhs += rhs; });
							}

							//traverse_gradients(res.merge_target, res.merge_source, [](auto& lhs, const auto& rhs) { lhs = rhs.fold(); });
							traverse_accums(res.merge_target, res.merge_source, [](auto& lhs, const auto& rhs) { lhs = rhs.fold(); });

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
				}
			}
			
			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Singlethreaded>{}))
				RunStep(int t, SlaveSet* set, Accumulator* acc)
			{
				mBarrier.WaitSlave();
				return master_res;
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Singlethreaded>{}))
				RunStep(int t, MasterSet* set, Accumulator* acc)
			{
				mBarrier.WaitMaster();
				master_res = set->alg_master(StepTag<STEP, Step_Singlethreaded>{});
				mBarrier.ReleaseMaster();
				return master_res;
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Accumulate>{}))
				RunStep(int t, SlaveSet* set, Accumulator* acc)
			{
				int res = 0;
				for (int i = 0; i < (Z / (THREADS*RO)); ++i)
				{
					res = set->alg[i](StepTag<STEP, Step_Accumulate>{});
				}

				return res;
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_AccReset>{})) RunStep(int t, SlaveSet* set, Accumulator* acc)
			{
				return set->alg[0](StepTag<STEP, Step_AccReset>{});
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_AccReset>{})) RunStep(int t, MasterSet* set, Accumulator* acc)
			{
				return set->alg_master(StepTag<STEP, Step_AccReset>{});
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Separate>{}))
				RunStep(int t, SlaveSet* set, Accumulator* acc)
			{
				using ret_type = decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Separate>{}));
				ret_type res = 0;
				int thread_offset = (Z / (THREADS*RO))*t;
				for (int i=0; i < Z / (THREADS*RO); ++i)
				{
					for (int j = 0; j < RO; ++j)
					{
						res = set->alg[i](StepTag<STEP, Step_Separate>{ (thread_offset + i)*RO + j, j });
					}
				}
				return res;
			}

			template<int STEP> decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Separate>{}))
				RunStep(int t, MasterSet* set, Accumulator* acc)
			{
				using ret_type = decltype(((Algorithm<ThreadBatch>*)nullptr)->operator()(StepTag<STEP, Step_Separate>{}));
				ret_type res = 0;
				int thread_offset = (Z / (THREADS*RO))*t;

				for (int j = 0; j < RO; ++j)
				{
					res = set->alg_master(StepTag<STEP, Step_Separate>{ (thread_offset)*RO + j, j });
				}

				for (int i = 0; i < Z / (THREADS*RO) - 1; ++i)
				{
					for (int j = 0; j < RO; ++j)
					{
						set->alg[i](StepTag<STEP, Step_Separate>{ (thread_offset + i + 1)*RO + j, j });
					}
				}
				return res;
			}

			void RunWorkerS(int t)
			{
				auto alg = std::make_unique<SlaveSet>();
				auto acc = std::make_unique<Accumulator>();

				for (auto& a : alg->alg)
					a.init(&mShared, acc.get());

				int step = 0;
				while (step >= 0)
				{
					step = mStepsS[step](t, alg.get(), acc.get());
				}
			}

			void RunWorkerM(int t)
			{
				auto alg = std::make_unique<MasterSet>();
				auto acc = std::make_unique<Accumulator>();

				for (auto& a : alg->alg)
					a.init(&mShared, acc.get());
				alg->alg_master.init(&mShared, acc.get());

				int step = 0;
				while (step >= 0)
				{
					step = mStepsM[step](t, alg.get(), acc.get());
				}
			}

			template<int step> void FillSteps()
			{
				mStepsS.push_back([this](int t, SlaveSet* alg, Accumulator* acc)->int { return RunStep<step>(t, alg, acc); });
				mStepsM.push_back([this](int t, MasterSet* alg, Accumulator* acc)->int { return RunStep<step>(t, alg, acc); });

				FillSteps<step + 1>();
			}

			template<> void FillSteps<Algorithm<ThreadBatch>::MaxStep + 1>()
			{}

		public:
			Dispatcher()
			{
				FillSteps<0>();
			}

			SharedData& Shared() { return mShared; }

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
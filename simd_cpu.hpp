#pragma once
#include <vector>
#include <memory>

#include "simd_array.hpp"
#include "simd_array_avx512.hpp"

#include <thread>
#include <condition_variable>

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

		template<template<class DataBatch> typename Algorithm, int Z, class Scalar, int THREADS, class SIMDTag = tag::Auto> class Dispatcher
		{
			using ThreadBatch = typename SIMDArray<Scalar, Z / THREADS, SIMDTag>;
			std::vector<std::thread> workers;

			std::mutex        m_slaves;
			std::mutex        m_master;

			std::condition_variable cv_slaves;
			std::condition_variable cv_master;

			int threads_waiting = 0;
			bool master_ready = false;

		public:
			void WaitMaster()
			{
				m_slaves.lock();
				std::unique_lock<std::mutex> lm(m_master);
				threads_waiting++;
				master_ready = false;
				while (threads_waiting < THREADS)
					cv_master.wait(lm);
			}

			void ReleaseMaster()
			{
				master_ready = true;
				m_slaves.unlock();
				cv_slaves.notify_all();
			}

			void WaitSlave()
			{
				m_master.lock();
				threads_waiting++;
				if (threads_waiting >= THREADS)
				{
					m_master.unlock();
					cv_master.notify_one();
				}
				else
				{
					m_master.unlock();
				}

				std::unique_lock<std::mutex> ls(m_slaves);
				while (!master_ready)
					cv_slaves.wait(ls);
			}

			class ExecutorMaster
			{
				Dispatcher *pOwner;
			public:
				ExecutorMaster(Dispatcher* pO) : pOwner(pO) {}

				template<class F> void SyncAndRun(const F& func)
				{
					pOwner->WaitMaster();
					func();
					pOwner->ReleaseMaster();
				}
			};

			class ExecutorSlave
			{
				Dispatcher *pOwner;
			public:
				ExecutorSlave(Dispatcher* pO) : pOwner(pO) {}

				template<class F> void SyncAndRun(const F& func)
				{
					pOwner->WaitSlave();
				}
			};

		public:
			Dispatcher()
			{}

			void RunWorker(int t)
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

		template<template<class DataBatch> typename Algorithm, int Z, class Scalar, int THREADS, int RO = 64, class SIMDTag = tag::Auto> class DispatcherRO
		{
			using ThreadBatch = typename SIMDArray<Scalar, RO, SIMDTag>;
			std::vector<std::thread> workers;

			std::mutex        m_slaves;
			std::mutex        m_master;

			std::condition_variable cv_slaves;
			std::condition_variable cv_master;

			int threads_waiting = 0;
			bool master_ready = false;

		public:
			void WaitMaster()
			{
				m_slaves.lock();
				std::unique_lock<std::mutex> lm(m_master);
				threads_waiting++;
				master_ready = false;
				while (threads_waiting < THREADS)
					cv_master.wait(lm);
			}

			void ReleaseMaster()
			{
				master_ready = true;
				m_slaves.unlock();
				cv_slaves.notify_all();
			}

			void WaitSlave()
			{
				m_master.lock();
				threads_waiting++;
				if (threads_waiting >= THREADS)
				{
					m_master.unlock();
					cv_master.notify_one();
				}
				else
				{
					m_master.unlock();
				}

				std::unique_lock<std::mutex> ls(m_slaves);
				while (!master_ready)
					cv_slaves.wait(ls);
			}

			class ExecutorMaster
			{
				DispatcherRO *pOwner;
			public:
				ExecutorMaster(DispatcherRO* pO) : pOwner(pO) {}

				template<class F> void SyncAndRun(const F& func)
				{
					pOwner->WaitMaster();
					func();
					pOwner->ReleaseMaster();
				}
			};

			class ExecutorSlave
			{
				DispatcherRO *pOwner;
			public:
				ExecutorSlave(DispatcherRO* pO) : pOwner(pO) {}

				template<class F> void SyncAndRun(const F& func)
				{
					pOwner->WaitSlave();
				}
			};

		public:
			DispatcherRO()
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
					workers.emplace_back(&DispatcherRO::RunWorker, this, t);
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
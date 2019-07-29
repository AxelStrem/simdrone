#pragma once

#include <thread>
#include <condition_variable>
#include <atomic>

template<int THREADS> class SyncLine
{
	std::mutex        m_slaves;
	std::mutex        m_master;
	std::mutex        m_enter;
	
	std::condition_variable cv_slaves;
	std::condition_variable cv_master;
	std::condition_variable cv_enter;

	std::atomic<int> threads_waiting = 0;
	std::atomic<int> threads_leaving = 0;
	bool master_ready = false;
public:
	void WaitMaster()
	{
		{
			std::unique_lock<std::mutex> ls(m_enter);
			while (threads_leaving > 0)
				cv_enter.wait(ls);
		}

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
		threads_waiting = 0;
		threads_leaving = THREADS - 1;
		m_slaves.unlock();
		cv_slaves.notify_all();
	}

	void WaitSlave()
	{
		{
			std::unique_lock<std::mutex> ls(m_enter);
			while (threads_leaving > 0)
				cv_enter.wait(ls);
		}

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

		m_enter.lock();
		if (--threads_leaving == 0)
		{
			master_ready = false;
			cv_enter.notify_all();
		}		
		m_enter.unlock();
	}
};
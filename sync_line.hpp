#pragma once

#include <thread>
#include <condition_variable>

template<int THREADS> class SyncLine
{
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
};
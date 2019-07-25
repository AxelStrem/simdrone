#pragma once
#include <numeric>
#include <algorithm>
#include <functional>

namespace simd
{
	template<class Derived> class ValArray
	{
	public:
		template<class F>
		Derived& apply(const F& func)
		{

			#pragma omp simd
			for (auto& x : *((Derived*)this))
			{
				x = func(x);
			}
			return *((Derived*)this);
		}

		template<class F>
		Derived& zip(const Derived& rhs, const F& func)
		{
			auto i1 = ((Derived*)(this))->begin();
			auto i2 = rhs.begin();

			#pragma omp simd
			for (; i1 != ((Derived*)(this))->end(); ++i1, ++i2)
			{
				*i1 = func(*i1, *i2);
			}
			return *((Derived*)this);
		}

		Derived& operator+=(const Derived& rhs)  {  return zip(rhs, std::plus<>{});  }
		Derived& operator-=(const Derived& rhs)  {  return zip(rhs, std::minus<>{}); }
		Derived& operator*=(const Derived& rhs)  {  return zip(rhs, std::multiplies<>{}); }
		Derived& operator/=(const Derived& rhs)  {  return zip(rhs, std::divides<>{}); }
		
		Derived& operator-() { return apply(std::negate<>{}); }
	};

	template<class Scalar, int Z> class alignas(64) AlignedArray : public ValArray<AlignedArray<Scalar, Z>>
	{
		Scalar data[Z];
	public:
		using ScalarType = Scalar;

		const Scalar* begin() const
		{
			return data;
		}

		const Scalar* end() const
		{
			return data + Z;
		}

		Scalar* begin()
		{
			return data;
		}

		Scalar* end()
		{
			return data + Z;
		}

		AlignedArray& operator=(const AlignedArray& rhs) = default;

		Scalar fold() const
		{
			Scalar res{};
			for (const Scalar& x : data) res += x;
			return res;
		}
	};

}
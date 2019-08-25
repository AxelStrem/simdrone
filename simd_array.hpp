#pragma once
#include <numeric>
#include <algorithm>
#include <functional>

namespace simd
{
	template<class Derived, class Scalar> class ValArray
	{
		struct func
		{
			struct clip_positive
			{
				Scalar operator()(const Scalar& a) const
				{
					return (a>0)?a:0;
				}
			};

			struct sign_positive
			{
				Scalar operator()(const Scalar& a) const
				{
					return (a>0) ? 1 : 0;
				}
			};
			
			struct fill
			{
				Scalar operator()(const Scalar& a, const Scalar& b) const { return b; }
			};

			struct abs
			{
				Scalar operator()(const Scalar& a) const
				{
					return (a>0) ? a : -a;
				}
			};
		};
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

		template<class F>
		Derived& zips(const Scalar& rhs, const F& func)
		{
			auto i1 = ((Derived*)(this))->begin();
			auto ie = ((Derived*)(this))->end();
			
			for (; i1 != ie; i1++)
			{
				i1[0] = func(i1[0], rhs);
			}
			return *((Derived*)this);
		}

		Derived& operator+=(const Derived& rhs)  {  return zip(rhs, std::plus<>{});  }
		Derived& operator-=(const Derived& rhs)  {  return zip(rhs, std::minus<>{}); }
		Derived& operator*=(const Derived& rhs)  {  return zip(rhs, std::multiplies<>{}); }
		Derived& operator/=(const Derived& rhs)  {  return zip(rhs, std::divides<>{}); }
		
		Derived& operator-() { return apply(std::negate<>{}); }

		Derived clone() const { return Derived{ *(reinterpret_cast<const Derived*>(this)) }; }

		Derived& operator=(const Scalar& rhs) { return this->zips(rhs, func::fill{}); }
		Derived& operator+=(const Scalar& rhs) { return this->zips(rhs, std::plus{}); }
		Derived& operator-=(const Scalar& rhs) { return this->zips(rhs, std::minus{}); }
		Derived& operator*=(const Scalar& rhs) { return this->zips(rhs, std::multiplies{}); }
		Derived& operator/=(const Scalar& rhs) { return this->zips(rhs, std::divides{}); }

		Derived operator+(const Derived& rhs) const { return clone() += rhs; }
		Derived operator-(const Derived& rhs) const { return clone() -= rhs; }
		Derived operator*(const Derived& rhs) const { return clone() *= rhs; }
		Derived operator/(const Derived& rhs) const { return clone() /= rhs; }

		Derived operator+(const Scalar& rhs) const { return clone() += rhs; }
		Derived operator-(const Scalar& rhs) const { return clone() -= rhs; }
		Derived operator*(const Scalar& rhs) const { return clone() *= rhs; }
		Derived operator/(const Scalar& rhs) const { return clone() /= rhs; }
		Derived& inverse(const Scalar& rhs) const { return clone().zips(rhs, avx512::divides_rev{}); }

		friend Derived operator*(const Scalar& lhs, const Derived& rhs) { return rhs * lhs; }
		friend Derived operator-(const Scalar& lhs, const Derived& rhs) { return -rhs + lhs; }
		friend Derived operator+(const Scalar& lhs, const Derived& rhs) { return rhs + lhs; }
		friend Derived operator/(const Scalar& lhs, const Derived& rhs) { return rhs.inverse(lhs); }

		friend Derived exp2(const Derived& x) { return x.clone().apply(std::exp2{}); }
		friend Derived clip_positive(const Derived& x) { return x.clone().apply(func::clip_positive{}); }
		friend Derived sign_positive(const Derived& x) { return x.clone().apply(func::sign_positive{}); }
		friend Derived abs(const Derived& x) { return x.clone().apply(func::abs{}); }
	};

	template<class Scalar, int Z> class alignas(64) AlignedArray : public ValArray<AlignedArray<Scalar, Z>, Scalar>
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
		AlignedArray& operator=(const Scalar& rhs) { return *((ValArray<AlignedArray<Scalar, Z>, Scalar>*)(this)) = rhs; }

		Scalar fold() const
		{
			Scalar res{};
			for (const Scalar& x : data) res += x;
			return res;
		}

		Scalar& operator[](int index)
		{
			return data[index];
		}
	};

}
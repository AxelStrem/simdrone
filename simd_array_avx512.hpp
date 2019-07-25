#pragma once

#pragma once
#include <numeric>
#include <algorithm>
#include <functional>

#include <immintrin.h>
#include <cstdlib>

namespace simd
{
	namespace avx512
	{
		template<class F> struct Value {};
		template<>        struct Value<float> 
		{ 
			using Type = __m512; 
			static __m512 fill(float x)
			{
				__m512 r;
				for (int i = 0; i < 16; ++i) r.m512_f32[i] = x;
				return r;
			}
		};
		template<>        struct Value<double>
		{ 
			using Type = __m512d;
			static __m512d fill(double x)
			{
				__m512d r;
				for (int i = 0; i < 8; ++i) r.m512d_f64[i] = x;
				return r;
			}
		};
		template<>        struct Value<int>
		{
			using Type = __m512i;
			static __m512i fill(int x)
			{
				__m512i r;
				for (int i = 0; i < 16; ++i) r.m512i_i32[i] = x;
				return r;
			}
		};

		struct plus
		{
			__m512  operator()(const __m512  a, const __m512  b) const { return _mm512_add_ps(a, b); }
			__m512d operator()(const __m512d a, const __m512d b) const { return _mm512_add_pd(a, b); }
			__m512i operator()(const __m512i a, const __m512i b) const { return _mm512_add_epi32(a, b); }
		};

		struct minus
		{
			__m512  operator()(const __m512  a, const __m512  b) const { return _mm512_sub_ps(a, b); }
			__m512d operator()(const __m512d a, const __m512d b) const { return _mm512_sub_pd(a, b); }
			__m512i operator()(const __m512i a, const __m512i b) const { return _mm512_sub_epi32(a, b); }
		};

		struct multiplies
		{
			__m512  operator()(const __m512  a, const __m512  b) const { return _mm512_mul_ps(a, b); }
			__m512d operator()(const __m512d a, const __m512d b) const { return _mm512_mul_pd(a, b); }
			__m512i operator()(const __m512i a, const __m512i b) const { return _mm512_mul_epi32(a, b); }
		};

		struct divides
		{
			__m512  operator()(const __m512  a, const __m512  b) const { return _mm512_div_ps(a, b); }
			__m512d operator()(const __m512d a, const __m512d b) const { return _mm512_div_pd(a, b); }
		};

		struct fill
		{
			__m512  operator()(const __m512  a, const __m512  b) const { return b; }
			__m512d operator()(const __m512d a, const __m512d b) const { return b; }
			__m512i operator()(const __m512i a, const __m512i b) const { return b; }
		};

		struct negate
		{
			__m512  operator()(const __m512  a) const { __m512  zero{};  return _mm512_sub_ps(zero, a); }
			__m512d operator()(const __m512d a) const { __m512d zero{};  return _mm512_sub_pd(zero, a); }
			__m512i operator()(const __m512i a) const { __m512i zero{};  return _mm512_sub_epi32(zero, a); }
		};
	}


	template<class Derived, class Scalar, int Size> class ValArrayAVX512_Unrolled
	{
	public:
		template<class F>
		Derived& apply(const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto ie = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->end());
			for (; i1 != ie; i1 += 4)
			{
				i1[0] = func(i1[0]);
				i1[1] = func(i1[1]);
				i1[2] = func(i1[2]);
				i1[3] = func(i1[3]);

			}
			return *((Derived*)this);
		}

		template<class F>
		Derived& zip(const Derived& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto i2 = reinterpret_cast<const typename avx512::Value<Scalar>::Type*>(rhs.begin());
			auto ie = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->end());
			for (; i1 != ie; i1 += 4, i2 += 4)
			{
				i1[0] = func(i1[0], i2[0]);
				i1[1] = func(i1[1], i2[1]);
				i1[2] = func(i1[2], i2[2]);
				i1[3] = func(i1[3], i2[3]);
			}
			return *((Derived*)this);
		}

		template<class F>
		Derived& zips(const Scalar& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto ie = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->end());
			auto v = avx512::Value<Scalar>::fill(rhs);

			for (; i1 != ie; i1 += 4)
			{
				i1[0] = func(i1[0], v);
				i1[1] = func(i1[1], v);
				i1[2] = func(i1[2], v);
				i1[3] = func(i1[3], v);
			}
			return *((Derived*)this);
		}
	};

	template<class Derived, class Scalar> class ValArrayAVX512_Unrolled<Derived, Scalar, 64> : public ValArrayAVX512_Unrolled<Derived, Scalar, 0>
	{
	public:
		template<class F>
		Derived& apply(const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			i1[0] = func(i1[0]);
			return *((Derived*)this);
		}

		template<class F>
		Derived& zip(const Derived& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto i2 = reinterpret_cast<const typename avx512::Value<Scalar>::Type*>(rhs.begin());
			i1[0] = func(i1[0], i2[0]);
			return *((Derived*)this);
		}
	};

	template<class Derived, class Scalar> class ValArrayAVX512_Unrolled<Derived, Scalar, 128> : public ValArrayAVX512_Unrolled<Derived, Scalar, 64>
	{
	public:
		template<class F>
		Derived& apply(const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			i1[0] = func(i1[0]);
			i1[1] = func(i1[1]);
			return *((Derived*)this);
		}

		template<class F>
		Derived& zip(const Derived& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto i2 = reinterpret_cast<const typename avx512::Value<Scalar>::Type*>(rhs.begin());
			i1[0] = func(i1[0], i2[0]);
			i1[1] = func(i1[1], i2[1]);
			return *((Derived*)this);
		}
	};

	template<class Derived, class Scalar> class ValArrayAVX512_Unrolled<Derived, Scalar, 256> : public ValArrayAVX512_Unrolled<Derived, Scalar, 128>
	{
	public:
		template<class F>
		Derived& apply(const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			i1[0] = func(i1[0]);
			i1[1] = func(i1[1]);
			i1[2] = func(i1[2]);
			i1[3] = func(i1[3]);
			return *((Derived*)this);
		}

		template<class F>
		Derived& zip(const Derived& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto i2 = reinterpret_cast<const typename avx512::Value<Scalar>::Type*>(rhs.begin());
			i1[0] = func(i1[0], i2[0]);
			i1[1] = func(i1[1], i2[1]);
			i1[2] = func(i1[2], i2[2]);
			i1[3] = func(i1[3], i2[3]);
			return *((Derived*)this);
		}
	};

	template<class Derived, class Scalar> class ValArrayAVX512_Unrolled<Derived, Scalar, 512> : public ValArrayAVX512_Unrolled<Derived, Scalar, 256>
	{
	public:
		template<class F>
		Derived& apply(const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			i1[0] = func(i1[0]);
			i1[1] = func(i1[1]);
			i1[2] = func(i1[2]);
			i1[3] = func(i1[3]);
			
			i1[4] = func(i1[4]);
			i1[5] = func(i1[5]);
			i1[6] = func(i1[6]);
			i1[7] = func(i1[7]);
			return *((Derived*)this);
		}

		template<class F>
		Derived& zip(const Derived& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto i2 = reinterpret_cast<const typename avx512::Value<Scalar>::Type*>(rhs.begin());
			i1[0] = func(i1[0], i2[0]);
			i1[1] = func(i1[1], i2[1]);
			i1[2] = func(i1[2], i2[2]);
			i1[3] = func(i1[3], i2[3]);
			
			i1[4] = func(i1[4], i2[4]);
			i1[5] = func(i1[5], i2[5]);
			i1[6] = func(i1[6], i2[6]);
			i1[7] = func(i1[7], i2[7]);
			return *((Derived*)this);
		}
	};

	template<class Derived, class Scalar> class ValArrayAVX512_Unrolled<Derived, Scalar, 1024> : public ValArrayAVX512_Unrolled<Derived, Scalar, 512>
	{
	public:
		template<class F>
		Derived& apply(const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			i1[0] = func(i1[0]);
			i1[1] = func(i1[1]);
			i1[2] = func(i1[2]);
			i1[3] = func(i1[3]);
			
			i1[4] = func(i1[4]);
			i1[5] = func(i1[5]);
			i1[6] = func(i1[6]);
			i1[7] = func(i1[7]);

			i1[8] = func(i1[8]);
			i1[9] = func(i1[9]);
			i1[10] = func(i1[10]);
			i1[11] = func(i1[11]);

			i1[12] = func(i1[12]);
			i1[13] = func(i1[13]);
			i1[14] = func(i1[14]);
			i1[15] = func(i1[15]);
			return *((Derived*)this);
		}

		template<class F>
		__forceinline Derived& zip(const Derived& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto i2 = reinterpret_cast<const typename avx512::Value<Scalar>::Type*>(rhs.begin());
			i1[0] = func(i1[0], i2[0]);
			i1[1] = func(i1[1], i2[1]);
			i1[2] = func(i1[2], i2[2]);
			i1[3] = func(i1[3], i2[3]);

			i1[4] = func(i1[4], i2[4]);
			i1[5] = func(i1[5], i2[5]);
			i1[6] = func(i1[6], i2[6]);
			i1[7] = func(i1[7], i2[7]);

			i1[8] = func(i1[8], i2[8]);
			i1[9] = func(i1[9], i2[9]);
			i1[10] = func(i1[10], i2[10]);
			i1[11] = func(i1[11], i2[11]);

			i1[12] = func(i1[12], i2[12]);
			i1[13] = func(i1[13], i2[13]);
			i1[14] = func(i1[14], i2[14]);
			i1[15] = func(i1[14], i2[15]);
			return *((Derived*)this);
		}
	};

	template<class Derived, class Scalar, int Z> class ValArrayAVX512 : public ValArrayAVX512_Unrolled<Derived, Scalar, Z*sizeof(Scalar)>
	{
	public:
		__forceinline Derived& operator+=(const Derived& rhs) { return this->zip(rhs, avx512::plus{}); }
		__forceinline Derived& operator-=(const Derived& rhs) { return this->zip(rhs, avx512::minus{}); }
		__forceinline Derived& operator*=(const Derived& rhs) { return this->zip(rhs, avx512::multiplies{}); }
		__forceinline Derived& operator/=(const Derived& rhs) { return this->zip(rhs, avx512::divides{}); }

		__forceinline Derived& operator-() { return this->apply(avx512::negate{}); }

		__forceinline Derived& operator=(const Scalar& rhs) { return this->zips(rhs, avx512::fill{}); }
		__forceinline Derived& operator+=(const Scalar& rhs) { return this->zips(rhs, avx512::plus{}); }
		__forceinline Derived& operator-=(const Scalar& rhs) { return this->zips(rhs, avx512::minus{}); }
		__forceinline Derived& operator*=(const Scalar& rhs) { return this->zips(rhs, avx512::multiplies{}); }
		__forceinline Derived& operator/=(const Scalar& rhs) { return this->zips(rhs, avx512::divides{}); }
	};

	template<class Scalar, int Z> class alignas(64) AlignedArrayAVX512 : public ValArrayAVX512<AlignedArrayAVX512<Scalar, Z>, Scalar, Z>
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

		AlignedArrayAVX512& operator=(const Scalar& rhs) { return ValArrayAVX512<AlignedArrayAVX512<Scalar, Z>, Scalar, Z>::operator=(rhs); };

		Scalar fold() const
		{
			Scalar res{};
			for (const Scalar& x : data) res += x;
			return res;
		}
	};

	template<class Scalar> class AlignedVectorAVX512 : public ValArrayAVX512<AlignedVectorAVX512<Scalar>, Scalar, 0>
	{
		Scalar* data;
		int Z;
	public:
		using ScalarType = Scalar;

		AlignedVectorAVX512(int sz) : Z(sz)
		{  data = std::aligned_alloc(64, Z*sizeof(Scalar)); }

		~AlignedVectorAVX512()
		{ std::free(data); }

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

		Scalar fold() const
		{
			Scalar res{};
			for (int i = 0; i < Z;++i) res += data[i];
			return res;
		}
	};

}
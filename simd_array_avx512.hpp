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
				__m512 r = {x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x };
				return r;
			}
		};
		template<>        struct Value<double>
		{ 
			using Type = __m512d;
			static __m512d fill(double x)
			{
				__m512d r = { x, x, x, x, x, x, x, x };
				return r;
			}
		};
		template<>        struct Value<int>
		{
			using Type = __m512i;
			static __m512i fill(int32_t x32)
			{
				int64_t x = (static_cast<int64_t>(x32) << 32) | x32;
				__m512i r = { x, x, x, x, x, x, x, x };
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

		struct divides_rev
		{
			__m512  operator()(const __m512  a, const __m512  b) const { return _mm512_div_ps(b, a); }
			__m512d operator()(const __m512d a, const __m512d b) const { return _mm512_div_pd(b, a); }
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


		__m512  load(const __m512* a) { return _mm512_loadu_ps(a); }
		__m512d load(const __m512d* a) { return _mm512_loadu_pd(a); }
		__m512i load(const __m512i* a) { return _mm512_loadu_epi32(a); }
	
		void store(__m512*  a, const __m512& v) { _mm512_storeu_ps(a, v); }
		void store(__m512d* a, const __m512d& v) { _mm512_storeu_pd(a, v); }
		void store(__m512i* a, const __m512i& v) { _mm512_storeu_epi32(a, v); }


		struct exp2
		{
			// not implemented
			__m512  operator()(const __m512  a) const
			{ 
				return a;
			}
			__m512d operator()(const __m512d a) const 
			{
				return a;
			}
		};

		struct abs
		{
			__m512  operator()(const __m512  a) const  { return _mm512_abs_ps(a);  }
			__m512d operator()(const __m512d a) const  { return _mm512_abs_pd(a);  }
		};

		struct clip_positive
		{
			__m512  operator()(const __m512  a) const 
			{
				__mmask16 mask = _mm512_fpclass_ps_mask(a, 0x40);
				constexpr __m512 zero{};
				return _mm512_mask_mov_ps(a, mask, zero);
			}
			__m512d  operator()(const __m512d  a) const
			{
				__mmask8 mask = _mm512_fpclass_pd_mask(a, 0x40);
				constexpr __m512d zero{};
				return _mm512_mask_mov_pd(a,mask,zero);
			}
		};

		struct sign_positive
		{
			__m512  operator()(const __m512  a) const
			{
				__mmask16 mask = _mm512_fpclass_ps_mask(a, 0x40);
				constexpr __m512 zero{};
				constexpr __m512 ones{1.f,1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
				return _mm512_mask_mov_ps(ones, mask, zero);
			}
			__m512d  operator()(const __m512d  a) const
			{
				__mmask8 mask = _mm512_fpclass_pd_mask(a, 0x40);
				constexpr __m512d zero{};
				constexpr __m512d ones{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
				return _mm512_mask_mov_pd(ones, mask, zero);
			}
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
				avx512::store(i1 + 0, func(avx512::load(i1 + 0)));
				avx512::store(i1 + 1, func(avx512::load(i1 + 1)));
				avx512::store(i1 + 2, func(avx512::load(i1 + 2)));
				avx512::store(i1 + 3, func(avx512::load(i1 + 3)));
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
				avx512::store(i1 + 0, func(avx512::load(i1 + 0), avx512::load(i2 + 0)));
				avx512::store(i1 + 1, func(avx512::load(i1 + 1), avx512::load(i2 + 1)));
				avx512::store(i1 + 2, func(avx512::load(i1 + 2), avx512::load(i2 + 2)));
				avx512::store(i1 + 3, func(avx512::load(i1 + 3), avx512::load(i2 + 3)));
			}
			return *((Derived*)this);
		}

		template<class F>
		Derived& zips(const Scalar& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto ie = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->end());
			auto v = avx512::Value<Scalar>::fill(rhs);

			for (; i1 != ie; i1 +=4)
			{
				avx512::store(i1 + 0, func(avx512::load(i1 + 0), v));
				avx512::store(i1 + 1, func(avx512::load(i1 + 1), v));
				avx512::store(i1 + 2, func(avx512::load(i1 + 2), v));
				avx512::store(i1 + 3, func(avx512::load(i1 + 3), v));
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
			avx512::store(i1 + 0, func(avx512::load(i1 + 0)));
			return *((Derived*)this);
		}

		template<class F>
		Derived& zip(const Derived& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto i2 = reinterpret_cast<const typename avx512::Value<Scalar>::Type*>(rhs.begin());
			avx512::store(i1 + 0, func(avx512::load(i1 + 0), avx512::load(i2 + 0)));
			return *((Derived*)this);
		}

		template<class F>
		Derived& zips(const Scalar& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto v = avx512::Value<Scalar>::fill(rhs);
			avx512::store(i1 + 0, func(avx512::load(i1 + 0), v));
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
			avx512::store(i1 + 0, func(avx512::load(i1 + 0)));
			avx512::store(i1 + 1, func(avx512::load(i1 + 1)));
			return *((Derived*)this);
		}

		template<class F>
		Derived& zip(const Derived& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto i2 = reinterpret_cast<const typename avx512::Value<Scalar>::Type*>(rhs.begin());
			avx512::store(i1 + 0, func(avx512::load(i1 + 0), avx512::load(i2 + 0)));
			avx512::store(i1 + 1, func(avx512::load(i1 + 1), avx512::load(i2 + 1)));
			return *((Derived*)this);
		}

		template<class F>
		Derived& zips(const Scalar& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto v = avx512::Value<Scalar>::fill(rhs);
			avx512::store(i1 + 0, func(avx512::load(i1 + 0), v));
			avx512::store(i1 + 1, func(avx512::load(i1 + 1), v));
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
			avx512::store(i1 + 0, func(avx512::load(i1 + 0)));
			avx512::store(i1 + 1, func(avx512::load(i1 + 1)));
			avx512::store(i1 + 2, func(avx512::load(i1 + 2)));
			avx512::store(i1 + 3, func(avx512::load(i1 + 3)));
			return *((Derived*)this);
		}

		template<class F>
		Derived& zip(const Derived& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto i2 = reinterpret_cast<const typename avx512::Value<Scalar>::Type*>(rhs.begin());
			avx512::store(i1 + 0, func(avx512::load(i1 + 0), avx512::load(i2 + 0)));
			avx512::store(i1 + 1, func(avx512::load(i1 + 1), avx512::load(i2 + 1)));
			avx512::store(i1 + 2, func(avx512::load(i1 + 2), avx512::load(i2 + 2)));
			avx512::store(i1 + 3, func(avx512::load(i1 + 3), avx512::load(i2 + 3)));
			return *((Derived*)this);
		}

		template<class F>
		Derived& zips(const Scalar& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto v = avx512::Value<Scalar>::fill(rhs);
			avx512::store(i1 + 0, func(avx512::load(i1 + 0), v));
			avx512::store(i1 + 1, func(avx512::load(i1 + 1), v));
			avx512::store(i1 + 2, func(avx512::load(i1 + 2), v));
			avx512::store(i1 + 3, func(avx512::load(i1 + 3), v));
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
			avx512::store(i1 + 0, func(avx512::load(i1 + 0)));
			avx512::store(i1 + 1, func(avx512::load(i1 + 1)));
			avx512::store(i1 + 2, func(avx512::load(i1 + 2)));
			avx512::store(i1 + 3, func(avx512::load(i1 + 3)));
			
			avx512::store(i1 + 4, func(avx512::load(i1 + 4)));
			avx512::store(i1 + 5, func(avx512::load(i1 + 5)));
			avx512::store(i1 + 6, func(avx512::load(i1 + 6)));
			avx512::store(i1 + 7, func(avx512::load(i1 + 7)));
			return *((Derived*)this);
		}

		template<class F>
		Derived& zip(const Derived& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto i2 = reinterpret_cast<const typename avx512::Value<Scalar>::Type*>(rhs.begin());
			avx512::store(i1 + 0, func(avx512::load(i1 + 0), avx512::load(i2 + 0)));
			avx512::store(i1 + 1, func(avx512::load(i1 + 1), avx512::load(i2 + 1)));
			avx512::store(i1 + 2, func(avx512::load(i1 + 2), avx512::load(i2 + 2)));
			avx512::store(i1 + 3, func(avx512::load(i1 + 3), avx512::load(i2 + 3)));
			
			avx512::store(i1 + 4, func(avx512::load(i1 + 4), avx512::load(i2 + 4)));
			avx512::store(i1 + 5, func(avx512::load(i1 + 5), avx512::load(i2 + 5)));
			avx512::store(i1 + 6, func(avx512::load(i1 + 6), avx512::load(i2 + 6)));
			avx512::store(i1 + 7, func(avx512::load(i1 + 7), avx512::load(i2 + 7)));
			return *((Derived*)this);
		}

		template<class F>
		Derived& zips(const Scalar& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto v = avx512::Value<Scalar>::fill(rhs);
			avx512::store(i1 + 0, func(avx512::load(i1 + 0), v));
			avx512::store(i1 + 1, func(avx512::load(i1 + 1), v));
			avx512::store(i1 + 2, func(avx512::load(i1 + 2), v));
			avx512::store(i1 + 3, func(avx512::load(i1 + 3), v));
			avx512::store(i1 + 4, func(avx512::load(i1 + 4), v));
			avx512::store(i1 + 5, func(avx512::load(i1 + 5), v));
			avx512::store(i1 + 6, func(avx512::load(i1 + 6), v));
			avx512::store(i1 + 7, func(avx512::load(i1 + 7), v));
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
			avx512::store(i1 + 0, func(avx512::load(i1 + 0)));
			avx512::store(i1 + 1, func(avx512::load(i1 + 1)));
			avx512::store(i1 + 2, func(avx512::load(i1 + 2)));
			avx512::store(i1 + 3, func(avx512::load(i1 + 3)));
			
			avx512::store(i1 + 4, func(avx512::load(i1 + 4)));
			avx512::store(i1 + 5, func(avx512::load(i1 + 5)));
			avx512::store(i1 + 6, func(avx512::load(i1 + 6)));
			avx512::store(i1 + 7, func(avx512::load(i1 + 7)));

			avx512::store(i1 + 8, func(avx512::load(i1 + 8)));
			avx512::store(i1 + 9, func(avx512::load(i1 + 9)));
			avx512::store(i1 + 10, func(avx512::load(i1 + 10)));
			avx512::store(i1 + 11, func(avx512::load(i1 + 11)));

			avx512::store(i1 + 12, func(avx512::load(i1 + 12)));
			avx512::store(i1 + 13, func(avx512::load(i1 + 13)));
			avx512::store(i1 + 14, func(avx512::load(i1 + 14)));
			avx512::store(i1 + 15, func(avx512::load(i1 + 15)));
			return *((Derived*)this);
		}

		template<class F>
		__forceinline Derived& zip(const Derived& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto i2 = reinterpret_cast<const typename avx512::Value<Scalar>::Type*>(rhs.begin());
			avx512::store(i1 + 0, func(avx512::load(i1 + 0), avx512::load(i2 + 0)));
			avx512::store(i1 + 1, func(avx512::load(i1 + 1), avx512::load(i2 + 1)));
			avx512::store(i1 + 2, func(avx512::load(i1 + 2), avx512::load(i2 + 2)));
			avx512::store(i1 + 3, func(avx512::load(i1 + 3), avx512::load(i2 + 3)));

			avx512::store(i1 + 4, func(avx512::load(i1 + 4), avx512::load(i2 + 4)));
			avx512::store(i1 + 5, func(avx512::load(i1 + 5), avx512::load(i2 + 5)));
			avx512::store(i1 + 6, func(avx512::load(i1 + 6), avx512::load(i2 + 6)));
			avx512::store(i1 + 7, func(avx512::load(i1 + 7), avx512::load(i2 + 7)));

			avx512::store(i1 + 8, func(avx512::load(i1 + 8), avx512::load(i2 + 8)));
			avx512::store(i1 + 9, func(avx512::load(i1 + 9), avx512::load(i2 + 9)));
			avx512::store(i1 + 10, func(avx512::load(i1 + 10), avx512::load(i2 + 10)));
			avx512::store(i1 + 11, func(avx512::load(i1 + 11), avx512::load(i2 + 11)));

			avx512::store(i1 + 12, func(avx512::load(i1 + 12), avx512::load(i2 + 12)));
			avx512::store(i1 + 13, func(avx512::load(i1 + 13), avx512::load(i2 + 13)));
			avx512::store(i1 + 14, func(avx512::load(i1 + 14), avx512::load(i2 + 14)));
			avx512::store(i1 + 15, func(avx512::load(i1 + 15), avx512::load(i2 + 15)));
			return *((Derived*)this);
		}

		template<class F>
		Derived& zips(const Scalar& rhs, const F& func)
		{
			auto i1 = reinterpret_cast<typename avx512::Value<Scalar>::Type*>(((Derived*)(this))->begin());
			auto v = avx512::Value<Scalar>::fill(rhs);
			avx512::store(i1 + 0, func(avx512::load(i1 + 0), v));
			avx512::store(i1 + 1, func(avx512::load(i1 + 1), v));
			avx512::store(i1 + 2, func(avx512::load(i1 + 2), v));
			avx512::store(i1 + 3, func(avx512::load(i1 + 3), v));

			avx512::store(i1 + 4, func(avx512::load(i1 + 4), v));
			avx512::store(i1 + 5, func(avx512::load(i1 + 5), v));
			avx512::store(i1 + 6, func(avx512::load(i1 + 6), v));
			avx512::store(i1 + 7, func(avx512::load(i1 + 7), v));
			
			avx512::store(i1 + 8, func(avx512::load(i1 + 8), v));
			avx512::store(i1 + 9, func(avx512::load(i1 + 9), v));
			avx512::store(i1 + 10, func(avx512::load(i1 + 10), v));
			avx512::store(i1 + 11, func(avx512::load(i1 + 11), v));

			avx512::store(i1 + 12, func(avx512::load(i1 + 12), v));
			avx512::store(i1 + 13, func(avx512::load(i1 + 13), v));
			avx512::store(i1 + 14, func(avx512::load(i1 + 14), v));
			avx512::store(i1 + 15, func(avx512::load(i1 + 15), v));
			return *((Derived*)this);
		}
	};

	template<class Derived, class Scalar, int Z> class ValArrayAVX512 : public ValArrayAVX512_Unrolled<Derived, Scalar, Z*sizeof(Scalar)>
	{
	public:
		Derived clone() const { return Derived{*(reinterpret_cast<const Derived*>(this))}; }

		__forceinline Derived& operator+=(const Derived& rhs) {	return this->zip(rhs, avx512::plus{});	}
		__forceinline Derived& operator-=(const Derived& rhs) { return this->zip(rhs, avx512::minus{}); }
		__forceinline Derived& operator*=(const Derived& rhs) { return this->zip(rhs, avx512::multiplies{}); }
		__forceinline Derived& operator/=(const Derived& rhs) { return this->zip(rhs, avx512::divides{}); }
		
		__forceinline Derived& operator-() const { return const_cast<ValArrayAVX512*>(this)->apply(avx512::negate{}); }

		__forceinline Derived& operator=(const Scalar& rhs) { return this->zips(rhs, avx512::fill{}); }
		__forceinline Derived& operator+=(const Scalar& rhs) { return this->zips(rhs, avx512::plus{}); }
		__forceinline Derived& operator-=(const Scalar& rhs) { return this->zips(rhs, avx512::minus{}); }
		__forceinline Derived& operator*=(const Scalar& rhs) { return this->zips(rhs, avx512::multiplies{}); }
		__forceinline Derived& operator/=(const Scalar& rhs) { return this->zips(rhs, avx512::divides{}); }

		__forceinline Derived operator+(const Derived& rhs) const { return clone() += rhs; }
		__forceinline Derived operator-(const Derived& rhs) const { return clone() -= rhs; }
		__forceinline Derived operator*(const Derived& rhs) const { return clone() *= rhs; }
		__forceinline Derived operator/(const Derived& rhs) const { return clone() /= rhs; }
		
		__forceinline Derived operator+(const Scalar& rhs) const { return clone() += rhs; }
		__forceinline Derived operator-(const Scalar& rhs) const { return clone() -= rhs; }
		__forceinline Derived operator*(const Scalar& rhs) const { return clone() *= rhs; }
		__forceinline Derived operator/(const Scalar& rhs) const { return clone() /= rhs; }
		__forceinline Derived& inverse(const Scalar& rhs) const { return clone().zips(rhs, avx512::divides_rev{}); }

		friend Derived operator*(const Scalar& lhs, const Derived& rhs) { return rhs * lhs; }
		friend Derived operator-(const Scalar& lhs, const Derived& rhs) { return -rhs + lhs; }
		friend Derived operator+(const Scalar& lhs, const Derived& rhs) { return rhs + lhs; }
		friend Derived operator/(const Scalar& lhs, const Derived& rhs) { return rhs.inverse(lhs); }
		
		friend Derived exp2(const Derived& x) { return x.clone().apply(avx512::exp2{}); }
		friend Derived clip_positive(const Derived& x) { return x.clone().apply(avx512::clip_positive{}); }
		friend Derived sign_positive(const Derived& x) { return x.clone().apply(avx512::sign_positive{}); }
		friend Derived abs(const Derived& x) { return x.clone().apply(avx512::abs{}); }
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

		Scalar& operator[](int index)
		{
			return data[index];
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

		Scalar& operator[](int index)
		{
			return data[index];
		}
	};

}
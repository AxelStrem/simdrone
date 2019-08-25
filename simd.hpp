#pragma once

namespace simd
{
	class Step_Parallel {};
	class Step_Separate {};
	class Step_Singlethreaded {};
	class Step_Accumulate {};
	class Step_AccReset {};

	template<class DataBatch> struct Fold
	{
		int next_step;
		DataBatch*                         merge_source;
		typename DataBatch::ScalarType*    merge_target;
	};

	template<class DataBatch> struct FoldAcc
	{
		int next_step;
		DataBatch*                         merge_source;
		typename DataBatch::ScalarType*    merge_target;
	};

	struct FoldMultiTag {};
	template<class DataBatch, class SourceContainer, class TargetContainer> struct FoldMulti : public FoldMultiTag
	{
		int next_step;
		SourceContainer* merge_source;
		TargetContainer* merge_target;
	};

	template<int STEP, class Tag = Step_Parallel> class StepTag {};
	template<int STEP> struct StepTag<STEP, Step_Separate>
	{
		int offset_global;
		int offset_local;
	};
}
#ifndef CAFFE_GROUPNORM_LAYER_HPP_
#define CAFFE_GROUPNORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	template <typename Dtype>
	class GroupNormLayer : public Layer<Dtype>{
	public :
		explicit GroupNormLayer(const LayerParameter& param) :Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "GroupNorm"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	
		Blob<Dtype> mean_, variance_, temp_, x_norm_;
		int group_ratio_ = 4;
		int group_num_;
		int channels_;
		int num_;
		Dtype eps_;
		
		Blob<Dtype> group_sum_multiplier_;
		Blob<Dtype> num_by_chans_;
		Blob<Dtype> spatial_sum_multiplier_;
		
	};



}



#endif
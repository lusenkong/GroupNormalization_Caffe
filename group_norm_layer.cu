#include <algorithm>
#include <vector>

#include "caffe/layers/group_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GroupNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	//int num_= bottom[0]->shape(0);
	int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0)*channels_);

	if (bottom[0] != top[0]) {
		caffe_copy(bottom[0]->count(), bottom_data, top_data);
	}

	// compute mean 
	caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num_, spatial_dim,
		1. / (num_ * spatial_dim), bottom_data,
		spatial_sum_multiplier_.gpu_data(), 0.,
		num_by_chans_.mutable_gpu_data());
	caffe_gpu_gemv<Dtype>(CblasNoTrans, num_* group_ratio_, group_num_,
		1. , num_by_chans_.gpu_data(), group_sum_multiplier_.gpu_data(), 0.,
		mean_.mutable_gpu_data());


	// subtract mean
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, group_num_, num_* group_ratio_, 1, 1,
		group_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
		num_by_chans_.mutable_gpu_data());
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num_,
		spatial_dim, 1, -1, num_by_chans_.gpu_data(),
		spatial_sum_multiplier_.gpu_data(), 1., top_data);

	// compute variance using var(X) = E((X-EX)^2)
	caffe_gpu_powx(top[0]->count(), top_data, Dtype(2),
		temp_.mutable_gpu_data());  // (X-EX)^2

	// E((X_EX)^2)
	caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num_, spatial_dim,
		1. / (num_ * spatial_dim), temp_.gpu_data(),
		spatial_sum_multiplier_.gpu_data(), 0.,
		num_by_chans_.mutable_gpu_data());
	caffe_gpu_gemv<Dtype>(CblasNoTrans, num_* group_ratio_, group_num_,
		1., num_by_chans_.gpu_data(), group_sum_multiplier_.gpu_data(), 0.,
		variance_.mutable_gpu_data());

	//normalize variance
	caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
	caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5), variance_.mutable_gpu_data());

	//replicate variance to input size
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, group_num_, num_* group_ratio_, 1, 1, group_sum_multiplier_.gpu_data(), variance_.gpu_data(), 0.,num_by_chans_.mutable_gpu_data());
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_* channels_, spatial_dim, 1, 1,
		num_by_chans_.gpu_data(), spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
	caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);

	caffe_copy(x_norm_.count(), top_data,
		x_norm_.mutable_gpu_data());
}

template <typename Dtype>
void GroupNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom) {
	
}

INSTANTIATE_LAYER_GPU_FUNCS(GroupNormLayer);
}
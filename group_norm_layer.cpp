#include <algorithm>
#include <vector>

#include "caffe/layers/group_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	
template <typename Dtype>
void GroupNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){

	if (bottom[0]->num_axes() == 1){
		channels_ = 1;
	}
	else{
		channels_ = bottom[0]->shape(1);
		group_num_ = channels_ /group_ratio_;
		num_ = bottom[0]->shape(0);
	}

	}

template <typename Dtype>
void GroupNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	if (bottom[0]->num_axes() >= 1)
		CHECK_EQ(bottom[0]->shape(1), channels_);
	top[0]->ReshapeLike(*bottom[0]);

	vector<int> sz;
	sz.push_back(group_ratio_ * num_);
	mean_.Reshape(sz);
	variance_.Reshape(sz);
	temp_.ReshapeLike(*bottom[0]);
	x_norm_.ReshapeLike(*bottom[0]);
	sz[0] = group_num_;
	group_sum_multiplier_.Reshape(sz);

	int spatial_dim = bottom[0]->count() / (channels_*bottom[0]->shape(0));
	
	if (spatial_sum_multiplier_.num_axes() == 0 || spatial_sum_multiplier_.shape(0) == spatial_dim){
		sz[0] = spatial_dim;
		spatial_sum_multiplier_.Reshape(sz);
		Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
		caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
	}

	int numbychans = channels_*bottom[0]->shape(0);
	if (num_by_chans_.num_axes() == 0 || num_by_chans_.shape(0) != numbychans){
			
		sz[0] = numbychans;
		num_by_chans_.Reshape(sz);
		caffe_set(group_sum_multiplier_.count(), Dtype(1), group_sum_multiplier_.mutable_cpu_data());
	}

}


template <typename Dtype>
void GroupNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	//int num_= bottom[0]->shape(0);
	int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0)*channels_);

	if (bottom[0] != top[0]) {
		caffe_copy(bottom[0]->count(), bottom_data, top_data);
	}

	// compute mean 
	caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num_, spatial_dim,
		1. / (group_num_ * spatial_dim), bottom_data,
		spatial_sum_multiplier_.cpu_data(), 0.,
		num_by_chans_.mutable_cpu_data());
	caffe_cpu_gemv<Dtype>(CblasNoTrans, num_* group_ratio_, group_num_,
		1. , num_by_chans_.cpu_data(), group_sum_multiplier_.cpu_data(), 0.,
		mean_.mutable_cpu_data());


	// subtract mean
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, group_num_, num_* group_ratio_, 1, 1,
		group_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
		num_by_chans_.mutable_cpu_data());
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num_,
		spatial_dim, 1, -1, num_by_chans_.cpu_data(),
		spatial_sum_multiplier_.cpu_data(), 1., top_data);

	// compute variance using var(X) = E((X-EX)^2)
	caffe_powx(top[0]->count(), top_data, Dtype(2),
		temp_.mutable_cpu_data());  // (X-EX)^2

	// E((X_EX)^2)
	caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num_, spatial_dim, 1. / (group_num_ * spatial_dim), temp_.cpu_data(), spatial_sum_multiplier_.cpu_data(), 0., num_by_chans_.mutable_cpu_data());
	caffe_cpu_gemv<Dtype>(CblasNoTrans, num_* group_ratio_, group_num_,1., num_by_chans_.cpu_data(), group_sum_multiplier_.cpu_data(), 0.,variance_.mutable_cpu_data());

	//normalize variance
	caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
	caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5), variance_.mutable_cpu_data());

	//replicate variance to input size
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, group_num_, num_* group_ratio_, 1, 1,group_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,num_by_chans_.mutable_cpu_data());
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_* channels_, spatial_dim, 1, 1,num_by_chans_.cpu_data(), spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
	caffe_div(temp_.count(), top_data, temp_.mutable_cpu_data(), top_data);

	caffe_copy(x_norm_.count(), top_data,
		x_norm_.mutable_cpu_data());



// void caffe_cpu_gemv
//(const CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha, const Dtype* A, const Dtype* x, const float beta, Dtype* y);
// y = alpha* op(A) * x + beta * y;
	//  A  : M x N
	//  x  : N x 1
	//  y  : M x 1


//void caffe_cpu_gemm
//(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const Dtype* A, const Dtype* B, const float beta, Dtype* C);
// C = alpha* op(A)* op(B) + beta* C;
	// C : M x N
	// common dim : K
	//compute mean


}

template <typename Dtype>
void GroupNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff;
	if (bottom[0] != top[0]){
		top_diff = top[0]->cpu_diff();
	}
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
}

//template <typename Dtype>
//void GroupNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//	const vector<Blob<Dtype>*>& top) {
//}
//
//template <typename Dtype>
//void GroupNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//	const vector<bool>& propagate_down,
//	const vector<Blob<Dtype>*>& bottom) {
//}

#ifdef CPU_ONLY
STUB_GPU(GroupNormLayer);
#endif

INSTANTIATE_CLASS(GroupNormLayer);
REGISTER_LAYER_CLASS(GroupNorm);
}
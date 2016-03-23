#ifndef CAFFE_FCN_CROP_LAYER_HPP_
#define CAFFE_FCN_CROP_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    template <typename Dtype>
    class FCNCropLayer : public Layer<Dtype> {
    public:
        explicit FCNCropLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "FCNCrop"; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 1; }
        virtual inline DiagonalAffineMap<Dtype> coord_map() {
            vector<pair<Dtype, Dtype> > coefs;
            coefs.push_back(make_pair((Dtype)1, (Dtype)-crop_h_));
            coefs.push_back(make_pair((Dtype)1, (Dtype)-crop_w_));
            return DiagonalAffineMap<Dtype>(coefs);
        }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        int crop_h_, crop_w_;
    };
}  // namespace caffe

#endif  // CAFFE_FCN_CROP_LAYER_HPP_


#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/layers/multi_stage_meanfield.hpp"

namespace caffe {

  template <typename TypeParam>
  class MultiStageMeanfieldTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
    
  protected:
    MultiStageMeanfieldTest() 
      : blob_bottom_unary_( new Blob<Dtype>(1,500,500,10)),  // softmax scores at each pixel: batchsize, height, width, nlabels
	blob_bottom_q0_( new Blob<Dtype>(1,500,500,10)),     // a copy of the above
	blob_bottom_img_( new Blob<Dtype>(1,500,500,3)),     // image data
	blob_top_pred_(new Blob<Dtype>(1,500,500,10))        // prediction of class labels at each pixel
    {}
    
    virtual void SetUp() {
    }
    
    
    virtual ~MultiStageMeanfieldTest() { 
    }
    
    void TestForward() {
      
      // set the parameters
      MultiStageMeanfieldParameter params;
      params.set_num_iterations(1);
      params.set_threshold(2);
      params.set_theta_alpha( 160 );
      params.set_theta_beta( 3 );
      params.set_theta_gamma( 3 );
      params.set_spatial_filter_weight( 3 );
      params.set_bilateral_filter_weight( 5 );
      
      // set the blob
      FillerParameter fillerpar;
      fillerpar.set_value( 0.0 );
      ConstantFiller filler( fillerpar );
      filler.Fill( blob_bottom_unary_ );
      filler.Fill( blob_bottom_q0_ );
      filler.Fill( blob_bottom_img_ );
      filler.Fill( blob_top_pred_ );

      // hard coded from above
      int nlabels = 10;
      int nchannels = 3;
      int cols = 500;
      int rows = 500;

      // doing something really simple at first: 
      // 1) all pixels are set to 1.0
      // 2) weight of all kernels set to 1.0
      // 3) all softmax vectors are set to (1,0,0,...)
      // 4) compatibilities set to set center pixel to 1.0

      std::vector<Dtype> score_data( cols*rows*nlabels, 0.0 );
      memset( score_data.data(), 1.0, sizeof(Dtype)*cols*rows ); // set first channel

      std::vector<Dtype> img_data( cols*rows*nchannels, 1.0 );

      blob_bottom_unary_->set_cpu_data( score_data.data() );
      blob_bottom_q0_->set_cpu_data( score_data.data() );
      blob_bottom_img_->set_cpu_data( img_data.data() );
      
      MultiStageMeanfieldLayer<Dtype> meanfield( params );
      
      EXPECT_EQ( 1, 1 );
    }
    
    Blob<Dtype>* blob_bottom_unary_;
    Blob<Dtype>* blob_bottom_q0_;
    Blob<Dtype>* blob_bottom_img_;
    Blob<Dtype>* blob_top_pred_;
  };
  
  
  TYPED_TEST_CASE(MultiStageMeanfieldTest, TestDtypesAndDevices);
  
  // TYPED_TEST(MultiStageMeanfieldTest, TestSetup) {}
  // TYPED_TEST(MultiStageMeanfieldTest, TestForward) {
  //   this->TestForward();
  //}
}

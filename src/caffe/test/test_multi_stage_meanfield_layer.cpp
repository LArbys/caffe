#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

  template <typename TypeParam>
  class MultiStageMeanfieldTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
    
  protected:
    MultiStageMeanfieldTest() 
      : blob_bottom_unary_( new Blob<Dtype>()), 
	blob_bottom_q0_( new Blob<Dtype>()),
	blob_bottom_img_( new Blob<Dtype>()),
	blob_top_pred_(new Blob<Dtype>())
    {}

    virtual void SetUp() {
    }


    virtual ~MultiStageMeanfieldTest() { 
    }

    void TestForward() {
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

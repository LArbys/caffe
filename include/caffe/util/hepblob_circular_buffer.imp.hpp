#ifndef HEPBLOB_CIRCULAR_BUFFER_IMP_HPP
#define HEPBLOB_CIRCULAR_BUFFER_IMP_HPP

#include "hepblob_circular_buffer.hpp"

template class caffe::hep_blob_circular_buffer< int    >;
template class caffe::hep_blob_circular_buffer< short  >;
template class caffe::hep_blob_circular_buffer< float  >;
template class caffe::hep_blob_circular_buffer< double >;

#endif

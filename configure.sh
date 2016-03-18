#!/usr/bin/env bash

# If CAFFE_DIR not set, try to guess                                          
if [[ -z $CAFFE_DIR ]]; then
    # Find the location of this script:                                         
    me="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    # Find the directory one above.                                             
    export CAFFE_DIR="$( cd "$( dirname "$me" )" && pwd )"
fi

case `uname -n` in 
    (wu)
	echo Setting up for wu...
	ln -s $CAFFE_DIR/Makefile.config.wu $CAFFE_DIR/Makefile.config
	;;
    (*)
	echo Unknown machine... Using default
	ln -s $CAFFE_DIR/Makefile.config.example $CAFFE_DIR/Makefile.config
	;;
esac


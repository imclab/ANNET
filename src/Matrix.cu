/*
 * Matrix.cpp
 *
 *  Created on: 05.04.2012
 *      Author: dgrat
 */

#include "include/gpgpu/Matrix.h"


namespace ANN {

/**
 * Matrix class implementation needs to be done by NVCC!
 */
Matrix::Matrix() : thrust::device_vector<float>(), iWidth(0), iHeight(0) {

}

Matrix::Matrix(unsigned int width, unsigned int height, float val) : thrust::device_vector<float>(width*height, val), iWidth(width), iHeight(height) {

}

Matrix::Matrix(unsigned int width, unsigned int height, thrust::host_vector<float> vec) : thrust::device_vector<float>(vec), iWidth(width), iHeight(height) {

}

thrust::device_vector<float> Matrix::getCol(const unsigned int x) const {
	assert(x < iWidth);
	thrust::device_vector<float> dvTmp(iHeight);
	for(unsigned int y = 0; y < iHeight; y++) {
		float fVal = (*this)[y*iWidth+x];
		dvTmp[y] = fVal;
	}
	return dvTmp;
}

}

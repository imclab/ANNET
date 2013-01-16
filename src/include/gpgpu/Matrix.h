/*
#-------------------------------------------------------------------------------
# Copyright (c) 2012 Daniel <dgrat> Frenzel.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Lesser Public License v2.1
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# 
# Contributors:
#     Daniel <dgrat> Frenzel - initial API and implementation
#-------------------------------------------------------------------------------
*/

#ifndef MATRIX_H_
#define MATRIX_H_

#include <thrust/device_vector.h>
#include <cassert>


namespace ANN {

/*
 * Host classes
 */
class Matrix : public thrust::device_vector<float> {
private:
	unsigned int iWidth;
	unsigned int iHeight;

public:
	Matrix();
	Matrix(unsigned int width, unsigned int height, float val);
	Matrix(unsigned int width, unsigned int height, thrust::host_vector<float> vec);

	thrust::device_vector<float> getCol(const unsigned int x) const;

	iterator getRowBegin(const unsigned int &y) {
		assert(y < iHeight);
		return begin()+y*iWidth;
	}
	iterator getRowEnd(const unsigned int &y) {
		assert(y < iHeight);
		return begin()+y*iWidth+iWidth;
	}

	const_iterator getRowBegin(const unsigned int &y) const {
		assert(y < iHeight);
		return begin()+y*iWidth;
	}
	const_iterator getRowEnd(const unsigned int &y) const {
		assert(y < iHeight);
		return begin()+y*iWidth+iWidth;
	}

	unsigned int getW() const {
		return iWidth;
	}
	unsigned int getH() const {
		return iHeight;
	}

	Matrix getInverse() {
		Matrix mat(iHeight, iWidth, 0);
		for(unsigned int y = 0; y < iWidth; y++) {
			mat.getRowBegin(y);
			thrust::device_vector<float> col = getCol(y);
			thrust::copy(col.begin(), col.end(), mat.getRowBegin(y) );
		}
		return mat;
	}
};

}

#endif /* MATRIX_H_ */

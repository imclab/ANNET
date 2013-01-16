/*
 * F2DArray.cpp
 *
 *  Created on: 28.01.2011
 *      Author: dgrat
 */

#include <cassert>
#include <string.h>
// own classes
#include "include/containers/2DArray.h"


using namespace ANN;


F2DArray::F2DArray() {
	m_iX 	= 0;
	m_iY 	= 0;
	m_pArray 	= NULL;
//	m_pSubArray = NULL;

	m_bAllocated = false;
}

#ifdef CUDA
/**
  * CUDA THRUST compatibility
  * host_vector<float>: Contains one row of the matrix
  * host_vector< host_vector<float> >: Contains all rows  of the matrix
  */
F2DArray::F2DArray(const Matrix &mat) {
	unsigned int iHeight 	= mat.getH();
	unsigned int iWidth 	= mat.getW();

	Alloc(iWidth, iHeight);

	for(unsigned int y = 0; y < iHeight; y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			m_pArray[y*iWidth+x] = mat[y*iWidth+x];
		}
	}
}

F2DArray::operator Matrix () {
	Matrix dmRes(GetW(), GetH(), 0.f);

	for(int y = 0; y < GetH(); y++) {
		for(int x = 0; x < GetW(); x++) {
			dmRes[y*GetW()+x] = m_pArray[y*GetW()+x];
		}
	}

	return dmRes;
}
#endif

F2DArray::F2DArray(float *pArray, const int &iSizeX, const int &iSizeY) {
	SetArray(pArray, iSizeX, iSizeY);
	SetArray(pArray, iSizeX, iSizeY);
	m_bAllocated = false;
}

F2DArray::~F2DArray() {
/*
	if(m_bAllocated) {
		if( m_pArray != NULL )
			delete [] m_pArray;
//		if(m_pSubArray != NULL)
//			delete [] m_pSubArray;
	}
*/
}

void F2DArray::Alloc(const int &iSize) {
	assert( iSize > 0 );
/*
	if(m_bAllocated) {
		if( m_pArray != NULL )
			delete [] m_pArray;
		if( m_pSubArray != NULL )
			delete [] m_pSubArray;
	}
*/
	m_iX 	= 0;
	m_iY 	= 0;
	m_pArray 	= new float[iSize];
//	m_pSubArray = NULL;
	memset( m_pArray, 0, iSize*sizeof(float) );
	m_bAllocated = true;
}

void F2DArray::Alloc(const int &iX, const int &iY) {
	assert( iY > 0 );
	assert( iX > 0 );
/*
	if( m_bAllocated ) {
		if( m_pArray != NULL )
			delete [] m_pArray;
		if( m_pSubArray != NULL )
			delete [] m_pSubArray;
	}
*/
	m_iX 	= iX;
	m_iY 	= iY;
	m_pArray 	= new float[iX*iY];
//	m_pSubArray = new float[iY];
	memset( m_pArray, 0, iX*iY*sizeof(float) );
	m_bAllocated = true;
}

const int &F2DArray::GetW() const {
	return m_iX;
}

const int &F2DArray::GetH() const {
	return m_iY;
}

int F2DArray::GetTotalSize() const {
	return m_iY * m_iX;
}

std::vector<float> F2DArray::GetSubArrayX(const int &iY) const {
	assert(iY < m_iY);

	std::vector<float> vSubArray(GetW() );
	std::copy(&m_pArray[iY*m_iX], &m_pArray[iY*m_iX]+GetW(), vSubArray.begin() );
	return vSubArray; //return &m_pArray[iY*m_iX];
}

std::vector<float> F2DArray::GetSubArrayY(const int &iX) const {
	assert(iX < m_iX);

	std::vector<float> vSubArray(GetH() );
	for(int y = 0; y < m_iY; y++) {
		vSubArray[y] = GetValue(iX, y);
	}
	return vSubArray;
}

F2DArray F2DArray::GetSubarray(const int &iX, const int &iY, const int &iSize) {
	assert(iY < m_iY);
	assert(iX < m_iX);

	F2DArray f2dArray;
	f2dArray.Alloc(iSize);

	int iC = iSize;
	for(int y = iY; y < m_iY; y++) {
		for(int x = iX; x < m_iX; x++) {
			if(iC > 0) {
				f2dArray.SetValue(this->GetValue(x, y), x, y);
			}
			else break;
			iC--;
		}
	}
	return f2dArray;
}

void F2DArray::SetValue(const float &fVal, const int &iX, const int &iY) {
	assert(iY < m_iY);
	assert(iX < m_iX);

	m_pArray[iX + iY*m_iX] = fVal;
}

float F2DArray::GetValue(const int &iX, const int &iY) const {
	assert(iY < m_iY);
	assert(iX < m_iX);

	return m_pArray[iX + iY*m_iX];
}

void F2DArray::SetArray(float *pArray, const int &iSizeX, const int &iSizeY) {
	assert( pArray != NULL );

	m_pArray = pArray;
	m_iX = iSizeX;
	m_iY = iSizeY;
}

float *F2DArray::GetArray() const {
	return m_pArray;
}


F2DArray::operator float*() {
	return m_pArray;
}

float *F2DArray::operator[] (const int &iY) const {
	return &m_pArray[iY*m_iX];
}

void F2DArray::GetOutput() {
	for(int y = 0; y < GetH(); y++) {
		for(int x = 0; x < GetW(); x++) {
			std::cout << "Array["<<x<<"]["<<y<<"]=" << GetValue(x, y) << std::endl;
		}
	}
}

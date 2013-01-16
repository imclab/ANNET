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

#ifndef PLAINNETARRAY_H_
#define PLAINNETARRAY_H_

#include <vector>

namespace ANN {

class F2DArray;


/**
 * \brief Pseudo 3D-array as a container for the weights of the network.
 *
 * @author Daniel "dgrat" Frenzel
 */
class F3DArray {
	friend class F2DArray;

private:
	int m_iX;	// nr. of neuron in layer m_iY
	int m_iY;	// nr. of layer in net
	int m_iZ;	// nr. of axon/weight of neuron m:iX in layer m_iY

public:
	// Public Access for CUDA
	/**
	 * Weights:
	 *
	 * m_fWeights[y] 		== LAYERS
	 * m_fWeights[y][x] 	== NEURONS
	 * m_fWeights[y][x][z] 	== EDGES
	 */
	float *m_pArray;

	// Standard C++ "conventions"
	F3DArray();
	~F3DArray();

	void Alloc(const int &iX, const int &iY, const int &iZ);

	const int &GetW() const;	// X
	const int &GetH() const;	// Y
	const int &GetD() const;	// Z
	int GetTotalSize() const; 	// X*Y*Z

	/* return a pointer to the subarray at: Y,X */
	F2DArray GetSubArrayYZ(const int &iX) const;
	F2DArray GetSubArrayXZ(const int &iY) const;
	F2DArray GetSubArrayXY(const int &iZ) const;

	void SetValue(const float &fVal,
			const int &iX, const int &iY, const int &iZ);
	float GetValue(const int &iX, const int &iY, const int &iZ) const;

//OPERATORS
	operator float*();
	F2DArray operator[] (const int &iX) const;
};

}

#endif /* PLAINNETARRAY_H_ */

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

#ifndef ANSOMNETGPU_H_
#define ANSOMNETGPU_H_

#include "../SOMNet.h"
#include "../gpgpu/Kernels.h"
#include "../gpgpu/Matrix.h"


namespace ANN {

class SOMNetGPU : public SOMNet {
private:
	ANN::Matrix m_EdgeMat;
	ANN::Matrix m_PosiMat;

	std::vector<SplittedNetExport> m_vSExp;

	int GetCudaDeviceCount() const;
	std::vector<SplittedNetExport> SplitDeviceData() const;
	void CombineDeviceData(const std::vector<SplittedNetExport> &SExp);

public:
	SOMNetGPU();
	SOMNetGPU(AbsNet *pNet);
	virtual ~SOMNetGPU();

	/**
	 * Trains the network with given input until iCycles is reached.
	 * @param iCycles Maximum number of training cycles.
	 */
	virtual void Training(const unsigned int &iCycles = 1000);
};

}

#endif /* ANSOMNETGPU_H_ */

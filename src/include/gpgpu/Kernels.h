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

#ifndef ANKERNELS_H_
#define ANKERNELS_H_

#include <cassert>
#include <vector>
#include "../containers/TrainingSet.h"
#include "../containers/2DArray.h"

#include <thrust/device_vector.h>

#include "../gpgpu/Matrix.h"
#include "../math/Functions.h"


class BMUExport {
public:
	BMUExport() {};
	BMUExport(unsigned int iUID, unsigned int iDID, thrust::host_vector<float> vPos) {
		iBMUID 			= iUID;
		iDeviceID 		= iDID;
		dvBMUPos 		= vPos;
	}

	unsigned int iBMUID;
	unsigned int iDeviceID;
	thrust::host_vector<float> dvBMUPos;
};

class SplittedNetExport {
public:
	SplittedNetExport() {};
	SplittedNetExport(const ANN::Matrix &mEdgeMat, const ANN::Matrix &mPosMat, const thrust::device_vector<float> &vConscience) {
		f2dEdges 		= mEdgeMat;
		f2dPositions 	= mPosMat;
		dvConscience 	= vConscience;
	}
  
	ANN::Matrix f2dEdges;
	ANN::Matrix f2dPositions;
	thrust::device_vector<float> dvConscience;
};

/*
 * BP kernels
 */
std::vector<float>
hostBPCalcDelta(const thrust::device_vector<float> &vNeurOut,
		const std::vector<float> &vTrainOut );

std::vector<thrust::device_vector<float> >
hostBPPropagateFW(const std::vector<ANN::Matrix> &vEdgeMatrices,
		const std::vector<ANN::Matrix> &vBiasEdgeMatrices,
		const std::vector<float> &vInput,
		const ANN::TransfFunction &function);

void
hostBPPropagateBW(std::vector<ANN::Matrix> &dvEdgeMatricesI,
		std::vector<ANN::Matrix> &dvMomentums,
		std::vector<thrust::device_vector<float> > &vErrorDeltas,
		const std::vector<thrust::device_vector<float> > &vNeuronValues,
		const float &fLearningRate,
		const float &fWeightDecay,
		const float &fMomentum,
		const ANN::TransfFunction &function);

/*
 * SOM kernels
 */
//////////////////////////////////////////////////////////////////////////////////////////////
float hostGetMax(const thrust::device_vector<float>& vec, unsigned int &ID);
float hostGetMin(const thrust::device_vector<float>& vec, unsigned int &ID);

//////////////////////////////////////////////////////////////////////////////////////////////
BMUExport
hostSOMFindBMNeuronID(std::vector<SplittedNetExport> &SExp,
		const thrust::device_vector<float> &InputVector,
		const float &fConscienceRate);

//////////////////////////////////////////////////////////////////////////////////////////////
void
hostSOMPropagateBW(std::vector<SplittedNetExport> &SExp,
		const thrust::device_vector<float> &dvInputVector,
		const BMUExport &,
		const float &fSigmaT,
		const float &fLearningRate );
		
void
hostSOMTraining( std::vector<SplittedNetExport> &SExp,
		const ANN::TrainingSet &InputSet,
		const unsigned int &iCycles,
		const float &fSigma0,
		const float &fLearningRate0,
		const float &fConscienceRate,
		float (*pfnDecay)(const float &, const float &, const float &) ) ;

#endif /* ANKERNELS_H_ */

/*
 * ANBPNetGPU.h
 *
 *  Created on: 14.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#ifndef ANBPNETGPU_H_
#define ANBPNETGPU_H_

#include "../BPNet.h"
#include "../gpgpu/Kernels.h"
#include "../gpgpu/Matrix.h"
#include "../math/Functions.h"

namespace ANN {

class BPNetGPU: public ANN::BPNet {
private:
	std::vector<ANN::Matrix> m_vEdgeMatricesI;
	std::vector<ANN::Matrix> m_vMomentums;
	std::vector<ANN::Matrix> m_vBiasEdges;
	std::vector<thrust::device_vector<float> > m_vNeuronVals;
	std::vector<thrust::device_vector<float> > m_dvOutDeltas;

public:
	void GetEdgeMatrices();
	void GetErrorDeltas();

	void RefreshEdges();
	void RefreshNeurons();

	void UpdateNeurons();		// only output layer (faster)
	void UpdateErrorDeltas(); 	// only output layer (faster)

	std::vector<float> GetCurrentInput();

public:
	BPNetGPU();
	virtual ~BPNetGPU();

	virtual void CreateNet(const ConTable &Net);

	virtual float SetOutput(const std::vector<float> &vOutArray);

	virtual void PropagateFW();
	virtual void PropagateBW();
	virtual std::vector<float> TrainFromData(const unsigned int &iCycles, const float &fTolerance, const bool &bBreak, float &fProgress);
};

}

#endif /* ANBPNETGPU_H_ */

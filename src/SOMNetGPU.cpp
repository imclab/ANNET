/*
 * ANSOMNetGPU.cpp
 *
 *  Created on: 01.04.2012
 *      Author: dgrat
 */

#include "include/gpgpu/SOMNetGPU.h"
#include "include/math/Functions.h"
#include "include/SOMLayer.h"
#include "include/base/AbsNeuron.h"
#include <cuda.h>


namespace ANN {

int SOMNetGPU::GetCudaDeviceCount() const {
	int iCount = 0;

	if(cudaGetDeviceCount(&iCount) != cudaSuccess)
		return 0;

	std::cout<<iCount<<" cuda-capable device(s) found."<<std::endl;
	return iCount;
}

std::vector<SplittedNetExport> SOMNetGPU::SplitDeviceData() const {
	std::vector<SplittedNetExport> vRes;
  
	unsigned int iStart 		= 0;
	unsigned int iStop 		= 0;
	unsigned int iSizeOfLayer 	= GetOPLayer()->GetNeurons().size();

	unsigned int iDeviceCount = GetCudaDeviceCount();
	for(unsigned int i = 0; i < iDeviceCount; i++) {
		if(cudaSetDevice(i) != cudaSuccess) {
			std::cout<<"SplitDeviceData(): Setting new cuda-capable device failed."<<std::endl;
			break;
		}

		iStart = i*(iSizeOfLayer/iDeviceCount);
		iStop = (i+1)*(iSizeOfLayer/iDeviceCount)-1;

		// Copy weights between neurons of the input and output layer
		ANN::Matrix f2dEdges 		= GetOPLayer()->ExpEdgesIn(iStart, iStop);
		// Copy positions of the neurons in the output layer
		ANN::Matrix f2dPositions 	= GetOPLayer()->ExpPositions(iStart, iStop);

		// Copy conscience information
		thrust::host_vector<float> hvConscience(iStop-iStart+1);
		thrust::device_vector<float> dvConscience;
		for(unsigned int j = 0; j <= iStop-iStart; j++) {
			hvConscience[j] = m_pOPLayer->GetNeuron(j+iStart)->GetValue();
		}
		dvConscience = hvConscience;

		SplittedNetExport SExp(f2dEdges, f2dPositions, dvConscience);
		vRes.push_back(SExp);
	}
	return vRes;
}

void SOMNetGPU::CombineDeviceData(const std::vector<SplittedNetExport> &SExp) {
	unsigned int iStart 		= 0;
	unsigned int iStop 		= 0;
	unsigned int iSizeOfLayer 	= GetOPLayer()->GetNeurons().size();

	unsigned int iDeviceCount = GetCudaDeviceCount();
	for(unsigned int i = 0; i < iDeviceCount; i++) {
		if(cudaSetDevice(i) != cudaSuccess) {
			std::cout<<"CombineDeviceData(): Setting new cuda-capable device failed."<<std::endl;
			break;
		}
		
		iStart = i*(iSizeOfLayer/iDeviceCount);
		iStop = (i+1)*(iSizeOfLayer/iDeviceCount)-1;
		
		// Copy weights between neurons of the input and output layer
		GetOPLayer()->ImpEdgesIn(SExp.at(i).f2dEdges, iStart, iStop);
		// Copy back conscience
		for(unsigned int j = 0; j <= iStop-iStart; j++) {
			m_pOPLayer->GetNeuron(j+iStart)->SetValue(SExp.at(i).dvConscience[j]);
		}
	}
}

SOMNetGPU::SOMNetGPU() {
	m_pIPLayer 		= NULL;
	m_pOPLayer 		= NULL;
	m_pBMNeuron 		= NULL;

	m_iCycle 		= 0;
	m_fSigma0 		= 0.f;
	m_fSigmaT 		= 0.f;
	m_fLearningRate 	= 0.5f;

	m_iWidthI 		= 0.f;
	m_iHeightI 		= 0.f;
	m_iWidthO 		= 0.f;
	m_iHeightO 		= 0.f;

	// Conscience mechanism
	m_fConscienceRate 	= 0.f;
	
	// mexican hat shaped function for this SOM
	SetDistFunction(&Functions::fcn_gaussian);

	m_fTypeFlag 	= ANNetSOM;
}

SOMNetGPU::SOMNetGPU(AbsNet *pNet) {
	if(pNet == NULL)
		return;

	std::vector<unsigned int> vDimI = ((SOMLayer*)(pNet->GetIPLayer() ))->GetDim();
	std::vector<unsigned int> vDimO = ((SOMLayer*)(pNet->GetOPLayer() ))->GetDim();

	// Copy weights between neurons of the input and output layer
	ANN::F2DArray f2dEdges = pNet->GetOPLayer()->ExpEdgesIn();
	// Copy positions of the neurons in the output layer
	ANN::F2DArray f2dPosistions = pNet->GetOPLayer()->ExpPositions();
	// Create the net finally
	CreateSOM(vDimI, vDimO, f2dEdges, f2dPosistions);
	// Copy training set
	SetTrainingSet(pNet->GetTrainingSet() );

	m_fTypeFlag 	= ANNetSOM;
}

SOMNetGPU::~SOMNetGPU() {

}

void SOMNetGPU::Training(const unsigned int &iCycles) {
	assert(iCycles > 0);
	assert(m_fSigma0 > 0.f);
	if(GetTrainingSet() == NULL) {
		std::cout<<"No training set available!"<<std::endl;
		return;
	}

	std::vector<SplittedNetExport> SExp = SplitDeviceData();
	hostSOMTraining(SExp,
		*GetTrainingSet(),
		iCycles,
		m_fSigma0,
		m_fLearningRate,
		m_fConscienceRate,
		&ANN::fcn_decay);

	std::cout<<"Training cycles finished properly"<<std::endl;
	// Write edge matrix back
	std::cout<<"Copy device memory back .."<<std::endl;
	// Copy data from device to host
	CombineDeviceData(SExp);	
	std::cout<<".. Finished"<<std::endl;
}

}

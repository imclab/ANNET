#ifndef _BPKERNELS_
#define _BPKERNELS_

#include "include/math/Functions.h"
#include "include/gpgpu/Kernels.h"
#include "include/math/Functions.h"


// Y <- A * X + Y
struct saxpy_functor {
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
	float operator()(const float& x, const float& y) const {
		return a * x + y;
	}
};

// Y <- A * X * Y
struct sax_functor {
    const float a;

    sax_functor(float _a) : a(_a) {}

    __host__ __device__
	float operator()(const float& x) const {
		return a * x;
	}
};
///////////////////////////////////////////////////////////////////////

inline void
SwitchTransfFunc(	std::vector<thrust::device_vector<float> > &vNeuronValues,
					thrust::device_vector<float> &dvLayer,
					thrust::device_vector<float> &dvBias,
					thrust::device_vector<float> &dvInput,
					const ANN::TransfFunction &function)
{
	// Run values through transfer function
	if (strcmp(function.name, "tanh") == 0) {
	    thrust::transform(
	    		dvLayer.begin(),
	    		dvLayer.end(),
	    		dvBias.begin(),
	    		dvLayer.begin(),
	    		ANN::tanTransferFcn() );
		// Now the input of the next layer will be the the previous one
	    dvInput = dvLayer;
		vNeuronValues.push_back(dvLayer);
		return;
	}
	if (strcmp(function.name, "log") == 0) {
	    thrust::transform(
	    		dvLayer.begin(),
	    		dvLayer.end(),
	    		dvBias.begin(),
	    		dvLayer.begin(),
	    		ANN::logTransferFcn() );
		// Now the input of the next layer will be the the previous one
	    dvInput = dvLayer;
		vNeuronValues.push_back(dvLayer);
		return;
	}
	if (strcmp(function.name, "binary") == 0) {
	    thrust::transform(
	    		dvLayer.begin(),
	    		dvLayer.end(),
	    		dvBias.begin(),
	    		dvLayer.begin(),
	    		ANN::binTransferFcn() );
		// Now the input of the next layer will be the the previous one
	    dvInput = dvLayer;
		vNeuronValues.push_back(dvLayer);
		return;
	}
	if (strcmp(function.name, "linear") == 0) {
	    thrust::transform(
	    		dvLayer.begin(),
	    		dvLayer.end(),
	    		dvBias.begin(),
	    		dvLayer.begin(),
	    		ANN::linTransferFcn() );
		// Now the input of the next layer will be the the previous one
	    dvInput = dvLayer;
		vNeuronValues.push_back(dvLayer);
		return;
	}
}

inline void
SwitchDevTransfFunc(thrust::device_vector<float> &dvNeurons,
					const std::vector<thrust::device_vector<float> > &vNeuronValues,
					const ANN::TransfFunction &function,
					const int &i)
{
	if (strcmp(function.name, "tanh") == 0) {
	    thrust::transform(
				vNeuronValues.at(i).begin(),
				vNeuronValues.at(i).end(),
				dvNeurons.begin(),
	    		ANN::devTanTransferFcn() );
	    return;
	}
	if (strcmp(function.name, "log") == 0) {
	    thrust::transform(
				vNeuronValues.at(i).begin(),
				vNeuronValues.at(i).end(),
				dvNeurons.begin(),
	    		ANN::devLogTransferFcn() );
	    return;
	}
	if (strcmp(function.name, "binary") == 0) {
				thrust::transform(
				vNeuronValues.at(i).begin(),
				vNeuronValues.at(i).end(),
				dvNeurons.begin(),
	    		ANN::devBinTransferFcn() );
	    return;
	}
	if (strcmp(function.name, "linear") == 0) {
	    thrust::transform(
				vNeuronValues.at(i).begin(),
				vNeuronValues.at(i).end(),
				dvNeurons.begin(),
	    		ANN::devLinTransferFcn() );
	    return;
	}
}
///////////////////////////////////////////////////////////////////////

std::vector<float>
hostBPCalcDelta(	const thrust::device_vector<float> &dvNeurOut,	// from forward run
					const std::vector<float> &vTrainOut ) 			// from training set
{
	thrust::device_vector<float> dvTrainOut (vTrainOut.begin(), vTrainOut.end() );
	thrust::device_vector<float> dvDelta	(vTrainOut.size(), 0.f);
    std::vector<float> vRes(vTrainOut.size() );

	// Calc error deltas of output layer
    thrust::transform(
    		dvTrainOut.begin(),
    		dvTrainOut.end(),
    		dvNeurOut.begin(),
    		dvDelta.begin(),
    		thrust::minus<float>() );

    thrust::copy(dvDelta.begin(), dvDelta.end(), vRes.begin());
    return vRes;
}
///////////////////////////////////////////////////////////////////////

std::vector<thrust::device_vector<float> >
hostBPPropagateFW(	const std::vector<ANN::Matrix> &vEdgeMatrices,
					const std::vector<ANN::Matrix> &vBiasEdgeMatrices,
					const std::vector<float> &vInput,
					const ANN::TransfFunction &function)
{
	std::vector<thrust::device_vector<float> > vNeuronValues(1, vInput);

	// Copy Input from vInput in device vector: vOutput
	thrust::device_vector<float> dvInput(vInput.begin(), vInput.end() ); 	// input
	thrust::device_vector<float> dvLayer;
	thrust::device_vector<float> dvBias;
	
	unsigned int iWidth 	= 0;
	unsigned int iHeight 	= 0;

	for(unsigned int i = 0; i < vEdgeMatrices.size(); i++) {	
		iWidth 		= vEdgeMatrices.at(i).getW();
		iHeight 	= vEdgeMatrices.at(i).getH();
		
		// Alloc memory
		dvLayer 	= thrust::device_vector<float>(iWidth, 0.f);
		dvBias 		= thrust::device_vector<float>(iWidth, 0.f);
	
		if(vBiasEdgeMatrices.at(i).getW() > 0) {
			dvLayer = thrust::device_vector<float>(vBiasEdgeMatrices.at(i).getRowBegin(0), vBiasEdgeMatrices.at(i).getRowEnd(0));
			dvBias 	= thrust::device_vector<float>(vBiasEdgeMatrices.at(i).getRowBegin(0), vBiasEdgeMatrices.at(i).getRowEnd(0));

			// initial bias term
			thrust::transform(dvBias.begin(),
				dvBias.end(),
				dvLayer.begin(),
				thrust::negate<float>() );

			// bias weights
		    thrust::transform( dvBias.begin(),
		    		dvBias.end(),
		    		dvLayer.begin(),
		    		dvLayer.begin(),
		    		saxpy_functor(1) );
		}

		// Calculate the result of the current layer
		for(unsigned int y = 0; y < iHeight; y++) {
		    // Y <- A * X + Y
		    thrust::transform( vEdgeMatrices.at(i).getRowBegin(y),
		    		vEdgeMatrices.at(i).getRowEnd(y),
		    		dvLayer.begin(),
		    		dvLayer.begin(),
		    		saxpy_functor(dvInput[y]) );
		}

		SwitchTransfFunc( vNeuronValues, dvLayer, dvBias, dvInput, function );
	}
	return vNeuronValues;
}
///////////////////////////////////////////////////////////////////////

inline void
AdadtEdges(	std::vector<ANN::Matrix> &vEdgeMatricesI,
			std::vector<thrust::device_vector<float> > &vErrors,
			std::vector<ANN::Matrix> &vMomentums,
			const std::vector<thrust::device_vector<float> > &vNeuronValues,
			const float &fLearningRate,
			const float &fWeightDecay,
			const float &fMomentum,
			const unsigned int iWidth, const unsigned int iHeight, const unsigned int i)
{
	/*
	 * Quick standard implementation
	 */
	if(fWeightDecay == 0.f && fMomentum == 0.f) {
		for(unsigned int y = 0; y < iHeight; y++) {
			thrust::transform( vErrors.at(i+1).begin(),
				vErrors.at(i+1).end(),
				vEdgeMatricesI.at(i).getRowBegin(y),
				vEdgeMatricesI.at(i).getRowBegin(y),
				saxpy_functor(fLearningRate*vNeuronValues.at(i)[y]) );
		}
		return;
	}

	/*
	 * Slower but more complex one
	 */
	thrust::device_vector<float> dvMomentums(iWidth, 0.f);
	ANN::Matrix matMomentums(iWidth, iHeight, 0);
	if(!vMomentums.size()) {
		vMomentums = std::vector<ANN::Matrix>(iHeight);
	}

	for(unsigned int y = 0; y < iHeight; y++) {
		// standard term
		thrust::transform( vErrors.at(i+1).begin(),
			vErrors.at(i+1).end(),
			dvMomentums.begin(),
			sax_functor(fLearningRate*vNeuronValues.at(i)[y]) );
		// weight decay
		if(fWeightDecay > 0.f) {
			thrust::transform( vEdgeMatricesI.at(i).getRowBegin(y),
				vEdgeMatricesI.at(i).getRowEnd(y),
				dvMomentums.begin(),
				dvMomentums.begin(),
				saxpy_functor(-fWeightDecay) );
		}
		// momentum term
		if(vMomentums.at(y).size() && fMomentum > 0.f) {
			thrust::transform( vMomentums.at(i).getRowBegin(y),
				vMomentums.at(i).getRowEnd(y),
				dvMomentums.begin(),
				dvMomentums.begin(),
				saxpy_functor(fMomentum) );

			thrust::copy(dvMomentums.begin(), dvMomentums.end(), matMomentums.getRowBegin(y) );
		}
		// .. belongs to standard term and updates weights
		thrust::transform( dvMomentums.begin(),
			dvMomentums.end(),
			vEdgeMatricesI.at(i).getRowBegin(y),
			vEdgeMatricesI.at(i).getRowBegin(y),
			thrust::plus<float>() );
	}
	// Safe momentums for the next run
	if(fMomentum > 0.f) {
		vMomentums[i] = matMomentums;
	}
}

void
hostBPPropagateBW(	std::vector<ANN::Matrix> &vEdgeMatricesI,
					std::vector<ANN::Matrix> &vMomentums,
					std::vector<thrust::device_vector<float> > &vErrors,
					const std::vector<thrust::device_vector<float> > &vNeuronValues,
					const float &fLearningRate,
					const float &fWeightDecay,
					const float &fMomentum,
					const ANN::TransfFunction &function )
{
	// All layers except output!
	for(int i = vEdgeMatricesI.size()-1; i >= 0; i--) {
		unsigned int iWidth 	= vEdgeMatricesI.at(i).getW();
		unsigned int iHeight 	= vEdgeMatricesI.at(i).getH();

		if(iWidth == 0 || iHeight == 0) {
			continue;
		}

		// errors of this layer
		assert(vErrors.at(i).size() == vNeuronValues.at(i).size());
		thrust::device_vector<float> dvErrors(vErrors.at(i).size(), 0);
		thrust::device_vector<float> dvNeurons(vNeuronValues.at(i).size(), 0);
		thrust::device_vector<float> dvEdges(iWidth, 0);

		// Calculate the result of the current layer
		for(unsigned int y = 0; y < iHeight; y++) {
			thrust::transform( vEdgeMatricesI.at(i).getRowBegin(y),
				vEdgeMatricesI.at(i).getRowEnd(y),
				vErrors.at(i+1).begin(),
				dvEdges.begin(),
				thrust::multiplies<float>() );

			dvErrors[y] = thrust::reduce(dvEdges.begin(), dvEdges.end(), (float) 0, thrust::plus<float>());
		}

		thrust::transform( vNeuronValues.at(i).begin(),
			vNeuronValues.at(i).end(),
			dvNeurons.begin(),
			ANN::devLogTransferFcn() );

		SwitchDevTransfFunc( dvNeurons, vNeuronValues, function, i );

		thrust::transform( dvNeurons.begin(),
			dvNeurons.end(),
			dvErrors.begin(),
			vErrors.at(i).begin(),
			thrust::multiplies<float>() );
	}

	// All layers except output ..
	for(int i = vEdgeMatricesI.size()-1; i >= 0 && vNeuronValues.size() > 0; i--) {
		unsigned int iWidth 	= vEdgeMatricesI.at(i).getW();
		unsigned int iHeight 	= vEdgeMatricesI.at(i).getH();

		AdadtEdges( vEdgeMatricesI, vErrors, vMomentums, vNeuronValues,
					fLearningRate, fWeightDecay, fMomentum,
					iWidth, iHeight, i );
	}
}

#endif

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

#ifndef SOMNET_H_
#define SOMNET_H_

#include "base/AbsNet.h"


namespace ANN {

class SOMNeuron;
class DistFunction;

class SOMNet : public AbsNet {
protected:
	const DistFunction 	*m_DistFunction;
	SOMNeuron 		*m_pBMNeuron;

	unsigned int 	m_iCycle;	// current cycle step in learning progress
	unsigned int 	m_iCycles;	// maximum of cycles
	float 			m_fSigma0;	// radius of the lattice at t0
	float 			m_fSigmaT;	// radius of the lattice at tx
	float 			m_fLambda;	// time constant
	float 			m_fLearningRateT;
	
	// Conscience mechanism
	float 			m_fConscienceRate;

	/* first Ctor */
	std::vector<unsigned int> m_vDimI; // dimensions of the input layer (Cartesian coordinates)
	std::vector<unsigned int> m_vDimO; // dimensions of the output layer (Cartesian coordinates)
	/* second Ctor */
	unsigned int 	m_iWidthI;	// width of the input layer
	unsigned int 	m_iHeightI;	// height of the input layer
	unsigned int 	m_iWidthO;	// width of the output layer
	unsigned int 	m_iHeightO; // height of the output layer

protected:
	/**
	 *
	 */
	void FindSigma0();		// size of the net

	/**
	 *
	 */
	void FindBMNeuron();	// best matching unit

	/**
	 * Implement to determine back propagation ( == learning ) behavior
	 */
	virtual void PropagateBW();

	/**
	 * TODO Implement to determine propagation behavior
	 */
	virtual void PropagateFW();

	/**
	 * Adds a layer to the network.
	 * @param iSize Number of neurons of the layer.
	 * @param flType Flag describing the type of the net.
	 */
	virtual void AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType);

public:
	/**
	 * Creates a self organizing map object.
	 */
	SOMNet();
	SOMNet(AbsNet *pNet);

	/**
	 * Creates a double layered network. Each layer with vDim[1] * vDim[2] * vDim[n+1] * .. neurons.
	 * @param vDimI vector inheriting the dimensions of the input layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 * @param vDimO vector inheriting the dimensions of the output layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 */
	SOMNet(const std::vector<unsigned int> &vDimI, const std::vector<unsigned int> &vDimO);

	/**
	 * Creates a double layered network.
	 * @param iWidthI Width of the input layer
	 * @param iHeightI Height of the input layer
	 * @param iWidthO Width of the output layer
	 * @param iHeightO Height of the output layer
	 */
	SOMNet(	const unsigned int &iWidthI, const unsigned int &iHeightI,
			const unsigned int &iWidthO, const unsigned int &iHeightO);

	virtual ~SOMNet();

	/*
	 *
	 */
	virtual void CreateNet(const ConTable &Net);

	SOMNet *GetNet();

	/**
	 * Creates a double layered network. Each layer with vDim[1] * vDim[2] * vDim[n+1] * .. neurons.
	 * The layers will get automatically connected properly, which means,
	 * every neuron in the output layer is connected to each neuron in the input layer.
	 * @param vDimI vector inheriting the dimensions of the input layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 * @param vDimO vector inheriting the dimensions of the output layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 */
	void CreateSOM(	const std::vector<unsigned int> &vDimI,
					const std::vector<unsigned int> &vDimO);

	void CreateSOM(	const std::vector<unsigned int> &vDimI,
					const std::vector<unsigned int> &vDimO,
					const F2DArray &f2dEdgeMat,
					const F2DArray &f2dNeurPos);

	/**
	 * Creates a double layered network.
	 * @param iWidthI Width of the input layer
	 * @param iHeightI Height of the input layer
	 * @param iWidthO Width of the output layer
	 * @param iHeightO Height of the output layer
	 */
	void CreateSOM(	const unsigned int &iWidthI, const unsigned int &iHeightI,
					const unsigned int &iWidthO, const unsigned int &iHeightO);

	/**
	 * Trains the network with given input until iCycles is reached.
	 * @param iCycles Maximum number of training cycles.
	 */
	virtual void Training(const unsigned int &iCycles = 1000);


	/**
	 * Sets learning rate scalar of the network.
	 * @param fVal New value of the learning rate. Recommended: 0.005f - 1.0f
	 */
	void SetLearningRate 	(const float &fVal);
	/**
	 * @return Return the learning rate of the net.
	 */
	float GetLearningRate() const;

	/**
	 * @param pFCN Kind of function the net has to use while back-/propagating.
	 */
	void SetDistFunction (const DistFunction *pFCN);

	/**
	 * @return Return the kind of function the net has to use while back-/propagating.
	 */
	const DistFunction *GetDistFunction() const;
	
	/**
	 * Sets the rate for the application of the conscience mechanism. 
	 * A value of zero leads to the standard kohonen implementation.
	 * Value must be: 0.f < fVal < 1.f
	 */
	void SetConscienceRate(const float &fVal);
	
	/**
	 * 
	 */
	float GetConscienceRate();
};

}

#endif /* SOMNET_H_ */

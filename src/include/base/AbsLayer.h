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

#ifndef ANBASICLAYER_H_
#define ANBASICLAYER_H_

#include <iostream>
#include <vector>
#include <stdint.h>
#include "../containers/2DArray.h"

#include <bzlib.h>

namespace ANN {

// own classes
class AbsNeuron;
class TransfFunction;
class ConTable;


enum {
	ANLayerInput 	= 1 << 0,	// type of layer
	ANLayerHidden 	= 1 << 1,	// type of layer
	ANLayerOutput 	= 1 << 2,	// type of layer

	ANBiasNeuron 	= 1 << 3	// properties of layer
};
typedef uint32_t LayerTypeFlag;

/**
 * \brief Represents a container for neurons in the network.
 */
class AbsLayer {
protected:
	/*
	 * Array of pointers to all neurons in this layer.
	 */
	std::vector<AbsNeuron *> m_lNeurons;

	/*
	 * ID of the layer
	 */
	int m_iID;

	/*
	 * Flag describing the kind of layer.
	 * (i. e. input, hidden or output possible)
	 */
	LayerTypeFlag m_fTypeFlag;

public:
	AbsLayer();
//	AbsLayer(const unsigned int &iNumber, int iShiftID = 0);
	virtual ~AbsLayer();

	/**
	 * Sets the current ID in the Network inheriting the layer.
	 * Useful for administration purposes.
	 */
	virtual void SetID(const int &iID);
	/**
	 * Returns the current ID in the Network inheriting the layer.
	 * Useful for administration purposes.
	 */
	virtual int GetID() const;

	/*
	 * TODO
	 */
	virtual void EraseAllEdges();
	/**
	 * Deletes the complete layer (all connections and all values).
	 */
	virtual void EraseAll();

	/**
	 * Resizes the layer. Deletes old neurons and adds new ones (initialized with random values).
	 * @param iSize New number of neurons.
	 * @param iShiftID When called each neuron created gets an ID defined in this function plus the value of iShiftID. Used for example in ANHFLayer, when creating 2d matrix.
	 */
	virtual void Resize(const unsigned int &iSize) = 0;

	/**
	 * Pointer to the neuron at index.
	 * @return Returns the pointer of the neuron at index iID
	 * @param iID Index of the neuron in m_lNeurons
	 */
	virtual AbsNeuron *GetNeuron(const unsigned int &iID) const;
	/**
	 * List of all neurons in this layer (not bias neuron).
	 * @return Returns an array with pointers of neurons in this layer.
	 */
	virtual const std::vector<AbsNeuron *> &GetNeurons() const;

	/**
	 *
	 */
	virtual void AddNeurons(const unsigned int &iSize) = 0;

	/**
	 * Defines the type of "activation" function the net has to use for back-/propagation.
	 * @param pFunction New "activation" function
	 */
	virtual void SetNetFunction 	(const TransfFunction *pFunction);

	/**
	 * Sets the type of the layer (input, hidden or output layer)
	 * @param fType Flag describing the type of the layer.
	 * Flag: "ANBiasNeuron" will automatically add a bias neuron.
	 */
	virtual void SetFlag(const LayerTypeFlag &fType);
	/**
	 * Adds a flag if not already set.
	 * @param fType Flag describing the type of the layer.
	 * Flag: "ANBiasNeuron" will automatically add a bias neuron.
	 */
	virtual void AddFlag(const LayerTypeFlag &fType);
	/**
	 * Type of the layer
	 * @return Returns the flag describing the type of the layer.
	 */
	virtual LayerTypeFlag GetFlag() const;

	/**
	 * Save layer's content to filesystem
	 */
	virtual void ExpToFS(BZFILE* bz2out, int iBZ2Error);
	/**
	 * Load layer's content to filesystem
	 * @return The ID of the current layer.
	 */
	virtual int ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable &Table);

	// FRIEND
	friend void SetEdgesToValue(AbsLayer *pSrcLayer, AbsLayer *pDestLayer, const float &fVal, const bool &bAdaptState = false);

	/** \brief:
	 * NEURON1	 			: edge1, edge2, edge[n < iWidth] ==> directing to input neuron 1
	 * NEURON2 				: edge1, edge2, edge[n < iWidth] ==> directing to input neuron 2
	 * NEURON3	 			: edge1, edge2, edge[n < iWidth] ==> directing to input neuron 3
	 * NEURON[i < iHeight] 	: edge1, edge2, edge[n < iWidth] ==> directing to input neuron i
	 * ..
	 * @return Returns a matrix: width=size_of_this_layer; height=size_previous_layer
	 */
	virtual F2DArray ExpEdgesIn() const;
	virtual void ImpEdgesIn(const F2DArray &);

	/** \brief:
	 * NEURON[iStart]		: edge1, edge2, edge[n < iWidth] ==> directing to input neuron 1
	 * NEURON[iStart+1]		: edge1, edge2, edge[n < iWidth] ==> directing to input neuron 2
	 * NEURON[..]	 		: edge1, edge2, edge[n < iWidth] ==> directing to input neuron 3
	 * NEURON[iStop] 		: edge1, edge2, edge[n < iWidth] ==> directing to input neuron i
	 * ..
	 * @return Returns a matrix: width=size_of_this_layer; height=iStop-iStart
	 */
	virtual F2DArray ExpEdgesIn(int iStart, int iStop) const;
	virtual void ImpEdgesIn(const F2DArray &, int iStart, int iStop);

	/** \brief:
	 * NEURON1				: edge1, edge1, edge[n < iWidth] ==> directing to next neuron 1, 2, n
	 * NEURON2				: edge2, edge2, edge[n < iWidth] ==> directing to next neuron 1, 2, n
	 * NEURON3				: edge3, edge3, edge[n < iWidth] ==> directing to next neuron 1, 2, n
	 * NEURON[i < iHeight] 	: edge4, edge4, edge[n < iWidth] ==> directing to next neuron 1, 2, n
	 * ..
	 * @return Returns a matrix: width=size_this_layer; height=size_of_next_layer
	 */
	virtual F2DArray ExpEdgesOut() const;
	virtual void ImpEdgesOut(const F2DArray &);

	/**
	 * pPositions:
	 * NEURON1				: X, Y, POS[n < iWidth] ==> directing to input
	 * NEURON2				: X, Y, POS[n < iWidth] ==> directing to input
	 * NEURON3				: X, Y, POS[n < iWidth] ==> directing to input
	 * NEURON[i < iHeight] 	: X, Y, POS[n < iWidth] ==> directing to input
	 */
	virtual F2DArray ExpPositions() const;
	virtual void ImpPositions(const F2DArray &f2dPos);
	
	/**
	 * pPositions:
	 * NEURON1				: X, Y, POS[n < iWidth] ==> directing to input
	 * NEURON2				: X, Y, POS[n < iWidth] ==> directing to input
	 * NEURON3				: X, Y, POS[n < iWidth] ==> directing to input
	 * NEURON[i < iHeight] 	: X, Y, POS[n < iWidth] ==> directing to input
	 */
	virtual F2DArray ExpPositions(int iStart, int iStop) const;
	virtual void ImpPositions(const F2DArray &f2dPos, int iStart, int iStop);
};

}
#endif /* ANBASICLAYER_H_ */

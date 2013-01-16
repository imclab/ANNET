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

#ifndef EDGE_H_
#define EDGE_H_

namespace ANN {

class AbsNeuron;


/**
 * \brief Represents an edge in the network. Always connecting two neurons.
 *
 * Each egde connects two neurons and has a specific value which might gets changed by connected neurons.
 *
 * @author Daniel "dgrat" Frenzel
 */
class Edge {
private:
	float m_fWeight;
	float m_fMomentum;

	AbsNeuron *m_pNeuronFirst;
	AbsNeuron *m_pNeuronSecond;

	bool m_bAllowAdaptation;

public:
	/**
	 * Creating a "weight" connecting two neurons.
	 */
	Edge();

	/**
	 * Creating a "weight" with the properties of: *pEdge
	 * this constructor can't copy connections (edges), because they normally have dependencies to other neurons.
	 */
	Edge(Edge *pEdge);
	/**
	 * Creating a "weight" connecting two neurons.
	 * @param pFirst Pointer directing to neuron number one.
	 * @param pSecond Pointer directing to neuron number two.
	 */
	Edge(AbsNeuron *pFirst, AbsNeuron *pSecond);
	/**
	 * Creating a "weight" connecting two neurons.
	 * @param pFirst Pointer directing to neuron number one.
	 * @param pSecond Pointer directing to neuron number two.
	 * @param fValue Value of the new Edge.
	 * @param fMomentum Value of the current momentum of the edge.
	 * @param bAdapt Allows to change the weight.
	 */
	Edge(AbsNeuron *pFirst, AbsNeuron *pSecond, float fValue, float fMomentum = 0.f, bool bAdapt = true);

	/**
	 * Looking from neuron pSource. Is returning a pointer to the other neuron.
	 * @param pSource Pointer to a neuron building this edge.
	 * @return Returns a pointer to the neuron != pSource.
	 */
	AbsNeuron *GetDestination(AbsNeuron *pSource) const;
	/**
	 * Looking from neuron pSource. Is returning an index to the other neuron.
	 * @param pSource Pointer to a neuron building this edge.
	 * @return Returns the index of the neuron != pSource.
	 */
	int GetDestinationID(AbsNeuron *pSource) const;

	/**
	 * Value of this edge.
	 * @return Returns the value of this edge.
	 */
	const float &GetValue() const;
	/**
	 * Sets the value of this edge.
	 * @param fValue New value of this edge.
	 */
	void SetValue(float fValue);

	/**
	 * Momentum of this edge.
	 * @return Returns the momentum of this edge.
	 */
	const float &GetMomentum() const;
	/**
	 * Sets the momentum of this edge.
	 * @param fValue New momentum of this edge.
	 */
	void SetMomentum(float fValue);

	/**
	 * Returns whether weight is changeable or not.
	 */
	bool GetAdaptationState() const;
	/**
	 * Switch the state whether changeable or not.
	 */
	void SetAdaptationState(const bool &bAdapt);

	/**
	 * Returns the value of the Edge.
	 */
	operator float() const;
};

}
#endif /* EDGE_H_ */

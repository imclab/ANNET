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

#ifndef TRANSFERFUNCTIONS_H_
#define TRANSFERFUNCTIONS_H_

#include <cmath>
#include <stdio.h>
#include <string.h>

namespace ANN {

//////////////////////////////////////////////////////////////////////////////////////////////
/** Some basic functions of the neuronal net
 * All of the functions could get used with CUDA
 * and there the declaration must be in the same file as the implementation
 */
//////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Transfer functions for backpropagation networks
 */
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_tanh_normal (const float& in, const float& theta) {
	return (tanh (in - theta));
}

#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_tanh_derivate (const float& in, const float& theta) {
	return (1.f - pow (tanh (in - theta), 2.f));
}
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_log_normal (const float& in, const float& theta) {
	return (1.f / (1.f + exp (theta - in)));
}

#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_log_derivate (const float& in, const float& theta) {
	float e_val;
	e_val = exp (theta - in);
	return (e_val / pow (e_val + 1.f, 2.f));
}
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_linear_normal (const float& in, const float& theta) {
	return (in - theta);
}

#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_linear_derivate (const float& in, const float& theta) {
	return (1.f);
}
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_binary_normal (const float& in, const float& theta) {
	if (in >= theta) {
		return (1.f);
	}
	return (-1.f);
}

#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_binary_derivate (const float& in, const float& theta) {
	return (1.f);
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	struct tanTransferFcn {
		__host__ __device__
		float operator()(const float& fVal, const float& fBias) const {
			return ANN::fcn_tanh_normal(fVal, fBias);
		}
	};

	struct devTanTransferFcn {
		__host__ __device__
		float operator()(const float& fVal) const {
			return ANN::fcn_tanh_derivate(fVal, 0);
		}
	};

//////////////////////////////////////////////////////////////////////////////////////////////
	struct binTransferFcn {
		__host__ __device__
		float operator()(const float& fVal, const float& fBias) const {
			return ANN::fcn_binary_normal(fVal, fBias);
		}
	};

	struct devBinTransferFcn {
		__host__ __device__
		float operator()(const float& fVal) const {
			return ANN::fcn_binary_derivate(fVal, 0);
		}
	};

//////////////////////////////////////////////////////////////////////////////////////////////
	struct linTransferFcn {
		__host__ __device__
		float operator()(const float& fVal, const float& fBias) const {
			return ANN::fcn_linear_normal(fVal, fBias);
		}
	};

	struct devLinTransferFcn {
		__host__ __device__
		float operator()(const float& fVal) const {
			return ANN::fcn_linear_derivate(fVal, 0);
		}
	};

//////////////////////////////////////////////////////////////////////////////////////////////
	struct logTransferFcn {
		__host__ __device__
		float operator()(const float& fVal, const float& fBias) const {
			return ANN::fcn_log_normal(fVal, fBias);
		}
	};

	struct devLogTransferFcn {
		__host__ __device__
		float operator()(const float& fVal) const {
			return ANN::fcn_log_derivate(fVal, 0);
		}
	};
#endif

//////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Distance functions for self organizing maps
 */
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_gaussian_bell (const float& dist, const float& sigmaT) {
	return exp(-pow(dist, 2.f)/(2.f*pow(sigmaT, 2.f)));
}
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_mexican_hat (const float& dist, const float& sigmaT) {
	return (2.f/sqrt(3.f) * pow(M_PI, -0.25f) ) * (1.f - pow(dist, 2.f)) * fcn_gaussian_bell(dist, sigmaT);
}
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_decay (const float& sigma0, const float& T, const float& lambda) {
	return sigma0*exp(-T/lambda);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/** \brief Represents an activation function.
  *
  * Complete definition of the function and it's derivate.
  */
class TransfFunction {
public:
	/** \brief The symbolic name of the function. */
	char * name;

	/** \brief Plain function itself for backpropagation networks.
	  *
	  * The first parameter gives the x-value,
	  * the second one is the theta value, taken from the neuron.
	  */
	float (* normal)(const float&, const float&);

	/** \brief The derivative function for backpropagation networks.
	  *
	  * Used for the backpropagation algorithm.
	  */
	float (* derivate)(const float&, const float&);
};

class DistFunction {
public:
	/** \brief The symbolic name of the function. */
	char * name;

	/**  \brief The distance function for SOMs
	 *
	 * Used for the determination of the excitation of a neuron.
	 */
	float (* distance)(const float&, const float&);

	/**  \brief The decay function for SOMs
	 *
	 * Calculates the decay after each epoch. \n
	 * \f$
	 * \\ \sigma(t) = \sigma_0e^{-\frac{t}{\lambda}}
	 * \\
	 * \\ \mbox{The Greek letter sigma (} \sigma_0 \mbox{) denotes the width of the lattice at time t(0) }
	 * \\ \mbox{and the Greek letter lambda (} \lambda \mbox{) denotes a time constant. }
	 * \\ \mbox{t is the current time-step (iteration of the loop). }
	 * \f$
	 */
	float (* decay)(const float& sigma, const float& t, const float& lambda);
};

/** \class Functions
 ** \brief List of activation functions that are available to the
 **        Network.
 */
class Functions {
public:
	/** \brief Resolve an activation function by symbolic name.
	  *
	  * \param  name The function name, as given in the function structure.
	  * \return NULL on failure, pointer to structure on success.
	  */
	static const TransfFunction* ResolveTransfFByName (const char *name);
	static const DistFunction*	 ResolveDistFByName (const char *name);

	 /**
	  * \brief The sigmoid tanh function.
	  *
	  * \f$f_{act} (x, \Theta) = tanh (x - \Theta)\f$
	  */
	static const TransfFunction fcn_tanh;
	 /**
	  * \brief The sigmoid log function.
	  *
	  * \f$f_{act} (x, \Theta) = \frac{1}{1 + e^{-(x - \Theta)}}\f$
	  */
	static const TransfFunction fcn_log;
	 /**
	  * \brief A linear activation function.
	  *
	  * \f$f_{act} (x, \Theta) = x - \Theta\f$
	  */
	static const TransfFunction fcn_linear;
	 /**
	  * \brief A binary activation function.
	  *
	  * \f$f_{act} (x, \Theta) = \left\{\begin{array}{cl}1.0 & x \geq
	  * \Theta\\-1.0 & x < \Theta\end{array}\right.\f$
	  */
	static const TransfFunction fcn_binary;

	/**
	 * \brief A gaussian distance function.
	 *
	 * \f$
	 *  \\ \sigma(t) = \sigma_0e^{-\frac{t}{\lambda}}
	 *  \\ h(t) = {e^{-\frac{dist^2}{2\sigma(t)^2}}}
	 * 	\\
	 *  \\ \mbox{The Greek letter sigma (} \sigma_0 \mbox{) denotes the width of the lattice at time t(0)}
	 *  \\ \mbox{and the Greek letter lambda (} \lambda \mbox{) denotes a time constant.}
	 *  \\ \mbox{t is the current time-step (iteration of the loop).}
	 * \f$
	 */
	static const DistFunction fcn_gaussian;

	/**
	 * \brief A gaussian distance function.
	 *
	 * \f$
	 * \\ \sigma(t) = \sigma_0e^{-\frac{t}{\lambda}}
	 * \\ h(t) = (\frac{2}{\sqrt{3}}\pi^{-\frac{1}{4}})(1-dist^2)({e^{-\frac{dist^2}{2\sigma(t)^2}}})
	 * \\
	 * \\ \mbox{The Greek letter sigma (} \sigma_0 \mbox{) denotes the width of the lattice at time t(0)}
	 * \\ \mbox{and the Greek letter lambda (} \lambda \mbox{) denotes a time constant.}
	 * \\ \mbox{t is the current time-step (iteration of the loop).}
	 * \\ \mbox{dist is the distance of a node to the BMU}
	 * \\ \mbox{and } \sigma \mbox{, is the width of the neighborhood function.}
	 * \f$
	 */
	static const DistFunction fcn_mexican;
};

};

#endif /* TRANSFERFUNCTIONS_H_ */

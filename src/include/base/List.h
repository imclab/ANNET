/*
 * ANVector.h
 *
 *  Created on: 24.11.2012
 *      Author: dgrat
 */

#include <list>
#include <cassert>


#ifndef ANNLIST_H_
#define ANNLIST_H_

namespace ANN {

template<class T>
class list : public std::list<T> {
public:
	list()  : std::list<T>() {};
	list(int n, const T& value = T() ) : std::list<T>(n, value) {};

	T &at(const unsigned int &iPos) {
		assert(iPos < this->size() );
		unsigned int i = iPos;
		for(typename std::list<T>::iterator it = this->begin(); it != this->end(); it++) {
			if(i == 0) {
				return *it;
			}
			i--;
		}
	}

	const T at(const unsigned int &iPos) const {
		assert(iPos < this->size() );
		unsigned int i = iPos;
		for(typename std::list<T>::const_iterator it = this->begin(); it != this->end(); it++) {
			if(i == 0) {
				return *it;
			}
			i--;
		}
		return T();	// error return
	}

	T& operator[](const unsigned int& iPos) {
		return this->at(iPos);
	}

	const T& operator[](const unsigned int& iPos) const {
		return this->at(iPos);
	}
};

}

#endif /* ANNLIST_H_ */

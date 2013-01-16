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

#ifndef RANDOMIZER_H_
#define RANDOMIZER_H_

#include <cstdlib>
#include <cmath>
#include <ctime>


#ifdef __linux__
	#include <sys/times.h>
	/*
	 * not defined in unix os but windows
	 */
	inline long getTickCount() {
		struct tms tm;
		return times(&tm);
	}
#endif /*__linux__*/

namespace ANN {
/*
 * predeclaration of some functions
 */
inline float RandFloat(float begin, float end);
inline int RandInt(int x,int y);

inline void InitTime();
#define INIT_TIME InitTime();

#ifdef WIN32
	/*
	 * for getTickCount()
	 */
	typedef unsigned long 	DWORD;
	typedef unsigned short 	WORD;
	typedef unsigned int 		UNINT32;

	#include <windows.h>
#endif /*WIN32*/

void InitTime() {
	time_t t;
	time(&t);
	srand((unsigned int)t);
}
/*
 * Returns a random number
 * Call of getTickCount() necessary
 */
float RandFloat(float begin, float end) {
	float temp;
	/* swap low & high around if the user makes no sense */
	if (begin > end) {
		temp = begin;
		begin = end;
		end = temp;
	}

	/* calculate the random number & return it */
	return rand() / (RAND_MAX + 1.f) * (end - begin) + begin;
}

//returns a random integer between x and y
int RandInt(int x,int y) {
	return rand()%(y-x+1)+x;
}

}

#endif /* RANDOMIZER_H_ */

#ifndef STRUCTURETENSOR_H_INCLUDED
#define STRUCTURETENSOR_H_INCLUDED


#include <sstream>
#include <string>
#include "DGtal/math/linalg/EigenDecomposition.h"

#include "DGtal/base/Common.h"
#include "DGtal/helpers/StdDefs.h"
#include "DGtal/images/ImageSelector.h"

// RealFFT
#include "DGtal/kernel/domains/HyperRectDomain.h"
#include "DGtal/kernel/SpaceND.h"
#include "DGtal/images/ImageContainerBySTLVector.h"
#include "RealFFT.h" 
#include "VTKWriter.h"



/** Compute the structure tensor.
	* @param 	im 		The image over which the structure tensor will be computed;
	*					T			The structure tensor;
	*					sizeT	= image size;
	*					s 		The first convolution parameter (default_value=3.0);
	*					r 		The second convolution parameter (default_value=2.0).
	*/

template <typename ImageMatrix, typename ImageDouble>
void structureTensor( ImageMatrix& T, const ImageDouble& imDouble,
		  double const& s = 3.0, double const& r = 2.0);


#include "structureTensor.ih"
#endif // STRUCTURETENSOR_H_INCLUDED

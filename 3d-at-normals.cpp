/**
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **/
/**
 * @file 3d-at-normals.cpp
 * @author Jacques-Olivier Lachaud (\c jacques-olivier.lachaud@univ-savoie.fr)
 * LAboratoire de MAthématiques - LAMA (CNRS, UMR 5127), Université de Savoie, France
 *
 * @date 2016/03/10
 *
 * Vol file viewer, with normals regularized by Ambrosio-Tortorelli functionnal
 *
 * Uses IntegralInvariantCurvatureEstimation
 * @see related article:
 *       Coeurjolly, D.; Lachaud, J.O; Levallois, J., (2013). Integral based Curvature
 *       Estimators in Digital Geometry. DGCI 2013. Retrieved from
 *       https://liris.cnrs.fr/publis/?id=5866
 *
 * This file is part of the DGtalTools.
 */

///////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include "DGtal/base/Common.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

// Shape constructors
#include "DGtal/io/readers/GenericReader.h"
#include "DGtal/images/ImageSelector.h"
#include "DGtal/images/imagesSetsUtils/SetFromImage.h"
#include "DGtal/images/IntervalForegroundPredicate.h"
#include "DGtal/topology/SurfelAdjacency.h"
#include "DGtal/topology/helpers/Surfaces.h"
#include "DGtal/topology/LightImplicitDigitalSurface.h"
#include <DGtal/topology/SetOfSurfels.h>

#include "DGtal/images/ImageHelper.h"
#include "DGtal/topology/DigitalSurface.h"
#include "DGtal/graph/DepthFirstVisitor.h"
#include "DGtal/graph/GraphVisitorRange.h"

// Noise
#include "DGtal/geometry/volumes/KanungoNoise.h"

// Integral Invariant includes
#include "DGtal/geometry/surfaces/estimation/IIGeometricFunctors.h"
#include "DGtal/geometry/surfaces/estimation/IntegralInvariantVolumeEstimator.h"
#include "DGtal/geometry/surfaces/estimation/IntegralInvariantCovarianceEstimator.h"

// Drawing
#include "DGtal/io/boards/Board3D.h"
#include "DGtal/io/colormaps/GradientColorMap.h"

#ifdef WITH_VISU3D_QGLVIEWER
#include "DGtal/io/viewers/Viewer3D.h"
#endif

using namespace DGtal;
using namespace functors;

/**
 * Missing parameter error message.
 *
 * @param param
 */
void missingParam( std::string param )
{
  trace.error() << " Parameter: " << param << " is required.";
  trace.info() << std::endl;
}

namespace po = boost::program_options;

int main( int argc, char** argv )
{
  // parse command line ----------------------------------------------
  po::options_description general_opt("Allowed options are");
  general_opt.add_options()
    ("help,h", "display this message")
    ("input,i", po::value< std::string >(), ".vol file")
    ("radius,r",  po::value< double >(), "Kernel radius for IntegralInvariant estimator" )
    ("noise,k",  po::value< double >()->default_value(0.5), "Level of Kanungo noise ]0;1[" )
    ("threshold,t",  po::value< unsigned int >()->default_value(8), "Min size of SCell boundary of an object" )
    ("minImageThreshold,l",  po::value<  int >()->default_value(0), "set the minimal image threshold to define the image object (object defined by the voxel with intensity belonging to ]minImageThreshold, maxImageThreshold ] )." )
    ("maxImageThreshold,u",  po::value<  int >()->default_value(255), "set the minimal image threshold to define the image object (object defined by the voxel with intensity belonging to ]minImageThreshold, maxImageThreshold] )." );

  bool parseOK = true;
  po::variables_map vm;
  try
    {
      po::store( po::parse_command_line( argc, argv, general_opt ), vm );
    }
  catch( const std::exception & ex )
    {
      parseOK = false;
      trace.error() << " Error checking program options: " << ex.what() << std::endl;
    }
  bool neededArgsGiven=true;

  if (parseOK && !(vm.count("input"))){
    missingParam("--input");
    neededArgsGiven=false;
  }
  if (parseOK && !(vm.count("radius"))){
    missingParam("--radius");
    neededArgsGiven=false;
  }

  double noiseLevel = vm["noise"].as< double >();
  if( noiseLevel < 0.0 || noiseLevel > 1.0 )
    {
      parseOK = false;
      trace.error() << "The noise level should be in the interval: [0, 1]"<< std::endl;
    }

  if(!neededArgsGiven || !parseOK || vm.count("help") || argc <= 1 )
    {
      trace.info()<< "Vol file viewer, with normals regularized by Ambrosio-Tortorelli functionnal" <<std::endl
                  << general_opt << "\n"
                  << "Basic usage: "<<std::endl
                  << "\t at-3d-normals -i file.vol -r 5 --noise 0.1"<<std::endl
                  << std::endl;
      return 0;
    }

  QApplication application(argc,argv);

  unsigned int threshold = vm["threshold"].as< unsigned int >();
  int minImageThreshold =  vm["minImageThreshold"].as<  int >();
  int maxImageThreshold =  vm["maxImageThreshold"].as<  int >();

  double re_convolution_kernel = vm["radius"].as< double >();

  //-----------------------------------------------------------------------------
  // Types.
  typedef Z3i::Space                                 Space;
  typedef Z3i::KSpace                                KSpace;
  typedef KSpace::SCell                              SCell;
  typedef KSpace::Cell                               Cell;
  typedef KSpace::Surfel                             Surfel;
  typedef Z3i::Domain                                Domain;
  typedef ImageSelector<Domain, unsigned char>::Type Image;
  typedef IntervalForegroundPredicate< Image >       Object;
  typedef KanungoNoise< Object, Domain >             KanungoPredicate;
  typedef BinaryPointPredicate<DomainPredicate<Domain>, KanungoPredicate, AndBoolFct2  > NoisyObject;

  //-----------------------------------------------------------------------------
  // Loading vol file.
  trace.beginBlock( "Loading vol file." );
  std::string inputFile = vm[ "input" ].as< std::string >();
  std::string extension = inputFile.substr(inputFile.find_last_of(".") + 1);
  if(extension!="vol" && extension != "p3d" && extension != "pgm3D" && extension != "pgm3d" && extension != "sdp" && extension != "pgm" )
    {
    trace.info() << "File extension not recognized: "<< extension << std::endl;
    return 0;
  }
  Image image = GenericReader<Image>::import (inputFile );
  trace.endBlock();

  //-----------------------------------------------------------------------------
  // Extracting object with possible noise.
  trace.beginBlock( "Extracting object with possible noise." );
  Object object( image,  minImageThreshold, maxImageThreshold );
  KanungoPredicate kanungo_pred( object, image.domain(), noiseLevel );
  DomainPredicate<Domain> domain_pred( image.domain() );
  AndBoolFct2 andF;
  NoisyObject noisy_object(domain_pred, kanungo_pred, andF  );
  Domain domain = image.domain();
  KSpace K;
  bool space_ok = K.init( domain.lowerBound()-Z3i::Domain::Point::diagonal(),
                          domain.upperBound()+Z3i::Domain::Point::diagonal(), true );
  if (!space_ok)
    {
      trace.error() << "Error in the Khalimsky space construction."<<std::endl;
      return 2;
    }
  CanonicSCellEmbedder< KSpace >       embedder( K );
  SurfelAdjacency< KSpace::dimension > surfAdj( true );
  trace.endBlock();


  //! [3dVolBoundaryViewer-ExtractingSurface]
  trace.beginBlock( "Extracting boundary by scanning the space. " );
  typedef KSpace::SurfelSet SurfelSet;
  typedef SetOfSurfels< KSpace, SurfelSet > MySetOfSurfels;
  typedef DigitalSurface< MySetOfSurfels > MyDigitalSurface;
  MySetOfSurfels theSetOfSurfels( K, surfAdj );
  Surfaces<KSpace>::sMakeBoundary( theSetOfSurfels.surfelSet(),
                                   K, image,
                                   domain.lowerBound(),
                                   domain.upperBound() );
  MyDigitalSurface digSurf( theSetOfSurfels );
  trace.info() << "Digital surface has " << digSurf.size() << " surfels."
               << std::endl;
  trace.endBlock();
  //! [3dVolBoundaryViewer-ExtractingSurface]
  
  //! [3dVolBoundaryViewer-ViewingSurface]
  trace.beginBlock( "Displaying everything. " );
  Viewer3D<Space,KSpace> viewer( K );
  viewer.setWindowTitle("Simple boundary of volume Viewer");
  viewer.show();
  typedef MyDigitalSurface::ConstIterator ConstIterator;
  viewer << SetMode3D(K.unsigns( *(digSurf.begin()) ).className(), "Basic");
  for ( ConstIterator it = digSurf.begin(), itE = digSurf.end(); it != itE; ++it )
    viewer << K.unsigns( *it );
  viewer << Viewer3D<>::updateDisplay;
  trace.endBlock();

  return application.exec();

  return 0;
}

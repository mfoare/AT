#include <sstream>
#include <string>
#include <boost/format.hpp>
//#include <boost/regex.hpp>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

// always include EigenSupport.h before any other Eigen headers
#include "DGtal/math/linalg/EigenDecomposition.h"

#include "DGtal/dec/DiscreteExteriorCalculus.h"
#include "DGtal/dec/DiscreteExteriorCalculusSolver.h"

#include "DGtal/base/Common.h"
#include "DGtal/helpers/StdDefs.h"
#include "DGtal/io/readers/GenericReader.h"
#include "DGtal/io/writers/GenericWriter.h"
#include "DGtal/io/boards/Board2D.h"
#include "DGtal/io/Display2DFactory.h"
#include "DGtal/io/colormaps/GrayscaleColorMap.h"
#include "DGtal/math/linalg/EigenSupport.h"
#include "DGtal/dec/DiscreteExteriorCalculus.h"
#include "DGtal/dec/DiscreteExteriorCalculusSolver.h"

// RealFFT
#include <DGtal/kernel/domains/HyperRectDomain.h>
#include <DGtal/kernel/SpaceND.h>
#include <DGtal/images/ImageContainerBySTLVector.h>
#include "VTKWriter.h"

// StructureTensor
//#include "structureTensor.h"

using namespace std;
using namespace DGtal;
using namespace Eigen;

double standard_deviation(const VectorXd& xx)
{
  const double mean = xx.mean();
  double accum = 0;
  for (int kk=0, kk_max=xx.rows(); kk<kk_max; kk++)
    accum += (xx(kk)-mean)*(xx(kk)-mean);
  return sqrt(accum)/(xx.size()-1);
}


template <typename Calculus, typename Image>
void PrimalForm0ToImage( const Calculus& calculus, const typename Calculus::PrimalForm0& u, Image& image )
{
  double min_u = u.myContainer[ 0 ];
  double max_u = u.myContainer[ 0 ];
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++)
    {
      min_u = min( min_u, u.myContainer[ index ] );
      max_u = max( max_u, u.myContainer[ index ] );
    }
  trace.info() << "min_u=" << min_u << " max_u=" << max_u << std::endl;
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++)
    {
      const typename Calculus::SCell& cell = u.getSCell( index );
      //int g = (int) round( ( u.myContainer[ index ] - min_u ) * 255.0 /( max_u - min_u ) );
      int g = (int) round( u.myContainer[ index ] * 255.0 );
      g = std::max( 0 , std::min( 255, g ) );
      image.setValue( calculus.myKSpace.sCoords( cell ), g );
    }
}

template <typename Calculus, typename Image>
void PrimalForm1ToImage( const Calculus& calculus, const typename Calculus::PrimalForm1& v, Image& image )
{
  double min_v = v.myContainer[ 0 ];
  double max_v = v.myContainer[ 0 ];
  for ( typename Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    {
      min_v = min( min_v, v.myContainer[ index ] );
      max_v = max( max_v, v.myContainer[ index ] );
    }
  trace.info() << "min_v=" << min_v << " max_v=" << max_v << std::endl;
  for ( typename Image::Iterator it = image.begin(), itE = image.end(); it != itE; ++it )
    *it = 255;
  for ( typename Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    {
      const typename Calculus::SCell& cell = v.getSCell( index );
      //int g = (int) round( ( v.myContainer[ index ] - min_v ) * 255.0 / ( max_v - min_v ) );
      int g = (int) round( v.myContainer[ index ] * 255.0 );
      g = std::max( 0 , std::min( 255, g ) );
      image.setValue( calculus.myKSpace.sKCoords( cell ), g );
    }
}

template <typename Calculus, typename Image>
void PrimalForm2ToImage( const Calculus& calculus, const typename Calculus::PrimalForm2& u, Image& image )
{
  double min_u = u.myContainer[ 0 ];
  double max_u = u.myContainer[ 0 ];
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++)
    {
      min_u = min( min_u, u.myContainer[ index ] );
      max_u = max( max_u, u.myContainer[ index ] );
    }
  trace.info() << "min_u=" << min_u << " max_u=" << max_u << std::endl;
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++)
    {
      const typename Calculus::SCell& cell = u.getSCell( index );
      //int g = (int) round( ( u.myContainer[ index ] - min_u ) * 255.0 /( max_u - min_u ) );
      int g = (int) round( u.myContainer[ index ] * 255.0 );
      g = std::max( 0 , std::min( 255, g ) );
      image.setValue( calculus.myKSpace.sCoords( cell ), g );
    }
}

template <typename Calculus, typename Image>
void PrimalForms2ToColorImage( const Calculus& calculus,
                               const typename Calculus::PrimalForm2& ur,
                               const typename Calculus::PrimalForm2& ug,
                               const typename Calculus::PrimalForm2& ub,
                               Image& image )
{
  for ( typename Calculus::Index index = 0; index < ur.myContainer.rows(); index++)
    {
      const typename Calculus::SCell& cell = ur.getSCell( index );
      int red   = (int) round( ur.myContainer[ index ] * 255.0 );
      red       = std::max( 0 , std::min( 255, red ) );
      int green = (int) round( ug.myContainer[ index ] * 255.0 );
      green     = std::max( 0 , std::min( 255, green ) );
      int blue  = (int) round( ub.myContainer[ index ] * 255.0 );
      blue      = std::max( 0 , std::min( 255, blue ) );
      image.setValue( calculus.myKSpace.sCoords( cell ), Color( red, green, blue ) );
    }
}

template <typename Calculus, typename Image>
void savePrimalForm0ToImage( const Calculus& calculus, const Image& image, const typename Calculus::PrimalForm0& u, const string& filename )
{
    Image end_image = image;
    PrimalForm0ToImage( calculus, u, end_image );
    ostringstream ossU;
    ossU << filename;
    string str_image_u = ossU.str();
    end_image >> str_image_u.c_str();
}

template <typename Calculus, typename Image>
void savePrimalForm1ToImage( const Calculus& calculus, const Image& image, const typename Calculus::PrimalForm1& v, const string& filename )
{
    Image end_image = image;
    PrimalForm1ToImage( calculus,v, end_image );
    ostringstream ossV;
    ossV << filename;
    string str_image_v = ossV.str();
    end_image >> str_image_v.c_str();
}

template <typename Calculus, typename Image>
void savePrimalForm2ToImage( const Calculus& calculus, const Image& image, const typename Calculus::PrimalForm2& u, const string& filename )
{
    Image end_image = image;
    PrimalForm2ToImage( calculus, u, end_image );
    ostringstream ossU;
    ossU << filename;
    string str_image_u = ossU.str();
    end_image >> str_image_u.c_str();
}

namespace DGtal {
namespace functors {
template <typename T>
struct IdentityFunctor : public std::unary_function<T,T>
{
    T operator()( T t ) const { return t; }
};
}
}

template <typename Calculus, typename Image>
void savePrimalForms2ToPPMImage( const Calculus& calculus, const Image& image,
                                 const typename Calculus::PrimalForm2& ur,
                                 const typename Calculus::PrimalForm2& ug,
                                 const typename Calculus::PrimalForm2& ub,
                                 const string& filename )
{
    Image end_image = image;
    PrimalForms2ToColorImage( calculus, ur, ug, ub, end_image );
    ostringstream ossU;
    ossU << filename;
    string str_image_u = ossU.str();

    typedef functors::IdentityFunctor<Color> IdColorFct;
    PPMWriter<Image,IdColorFct>::exportPPM( str_image_u.c_str(), end_image, IdColorFct(), true );
}

template < typename Board, typename Calculus >
void displayForms( Board& aBoard, const Calculus& calculus,
                   const typename Calculus::PrimalForm2& ur,
                   const typename Calculus::PrimalForm2& ug,
                   const typename Calculus::PrimalForm2& ub,
                   const typename Calculus::PrimalForm0& v )
{

  typedef typename Calculus::KSpace KSpace;
  typedef typename Calculus::SCell  SCell;
  typedef typename Calculus::Cell   Cell;
  typedef typename Calculus::Index  Index;
  typedef typename KSpace::Space    Space;
  typedef typename KSpace::Point    Point;
  typedef HyperRectDomain<Space>    Domain;
  typedef ImageContainerBySTLVector<Domain, unsigned char> Image;
  const KSpace& K = calculus.myKSpace;
  Domain domain( K.lowerBound(), K.upperBound() );
  Image image_u( domain );
  //PrimalForm2ToImage( calculus, u, image_u );
  //PrimalForms2ToColorImage( calculus, ur, ug, ub, image_u );
  typename Image::Value min = 0;
  typename Image::Value max = 255;
  //DGtal::GrayscaleColorMap<float> colormap( 0.0, 1.0 );
  // DGtal::Display2DFactory::drawImage< DGtal::GrayscaleColorMap<float>, Image>( aBoard, image_u, min, max );
  // aBoard << image_u;
  aBoard.setLineWidth( 0.0 );
  for ( Index idx = 0; idx < ur.myContainer.rows(); ++idx )
    {
      Cell cell = K.unsigns( ur.getSCell( idx ) );
      Point x   = K.uCoords( cell );
      //float val = u.myContainer( idx );
      int red   = (int) round( ur.myContainer[ idx ] * 255.0 );
      red       = std::max( 0 , std::min( 255, red ) );
      int green = (int) round( ug.myContainer[ idx ] * 255.0 );
      green     = std::max( 0 , std::min( 255, green ) );
      int blue  = (int) round( ub.myContainer[ idx ] * 255.0 );
      blue      = std::max( 0 , std::min( 255, blue ) );

      //Color c   = colormap( std::max( 0.0f, std::min( 1.0f, val ) ) );
      Color c( red , green , blue );
      aBoard.setPenColor( c );
      aBoard.setFillColor( c );
      aBoard.fillRectangle( NumberTraits<typename Image::Domain::Space::Integer>::
                           castToDouble(x[0]) - 0.5,
                           NumberTraits<typename Image::Domain::Space::Integer>::
                           castToDouble(x[1]) + 0.5, 1, 1);
    }

  Cell hor    = K.uIncident( K.uPointel( K.lowerBound() ), 0, true );
  Cell hfirst = K.uCell( K.lowerBound(), hor );
  Cell hlast  = K.uCell( K.upperBound()+Point(0,1), hor );
  aBoard << CustomStyle( hor.className(), new CustomColors( Color( 220, 0, 0 ), Color( 255, 0, 0 ) ) );
  do
    {
      Cell p0   = K.uIncident( hor, 0, false );
      Cell p1   = K.uIncident( hor, 0, true );
      double v0 = v.myContainer( calculus.getCellIndex( p0 ) );
      double v1 = v.myContainer( calculus.getCellIndex( p1 ) );
      if ( v0 <= 0.5 ) aBoard << p0;
      if ( v1 <= 0.5 ) aBoard << p1;
      if ( std::max(v0,v1) <= 0.5 ) aBoard << hor;
    }
  while ( K.uNext( hor, hfirst, hlast ) );
  Cell ver    = K.uIncident( K.uPointel( K.lowerBound() ), 1, true );
  Cell vfirst = K.uCell( K.lowerBound(), ver );
  Cell vlast  = K.uCell( K.upperBound()+Point(1,0), ver );
  aBoard << CustomStyle( hor.className(), new CustomColors( Color( 220, 0, 0 ), Color( 255, 0, 0 ) ) );
  do
    {
      Cell p0   = K.uIncident( ver, 1, false );
      Cell p1   = K.uIncident( ver, 1, true );
      double v0 = v.myContainer( calculus.getCellIndex( p0 ) );
      double v1 = v.myContainer( calculus.getCellIndex( p1 ) );
      if ( v0 <= 0.5 ) aBoard << p0;
      if ( v1 <= 0.5 ) aBoard << p1;
      if ( std::max(v0,v1) <= 0.5 ) aBoard << ver;
    }
  while ( K.uNext( ver, vfirst, vlast ) );
}

template <typename Calculus>
void saveFormsToEps( const Calculus& calculus,
                     const typename Calculus::PrimalForm2& ur,
                     const typename Calculus::PrimalForm2& ug,
                     const typename Calculus::PrimalForm2& ub,
                     const typename Calculus::PrimalForm0& v,
                     const string& filename )
{
    Board2D aBoard;
    displayForms( aBoard, calculus, ur, ug, ub, v );
    aBoard.saveEPS( filename.c_str() );
}

double tronc( const double& nb, const int& p )
{
  int i = pow(10,p) * nb;
  return i/pow(10,p);
}

template <typename Calculus>
typename Calculus::PrimalIdentity0 diag( const Calculus& calculus,
                                         const typename Calculus::PrimalForm0& v )
{
  typename Calculus::PrimalIdentity0 diag_v = calculus.template identity<0, PRIMAL>();
  for ( typename Calculus::Index index = 0; index < v.myContainer.rows(); index++ )
    diag_v.myContainer.coeffRef( index, index ) = v.myContainer( index );
  return diag_v;
}

template <typename Calculus>
typename Calculus::PrimalIdentity1 diag( const Calculus& calculus,
                                         const typename Calculus::PrimalForm1& v )
{
  typename Calculus::PrimalIdentity1 diag_v = calculus.template identity<1, PRIMAL>();
  for ( typename Calculus::Index index = 0; index < v.myContainer.rows(); index++ )
    diag_v.myContainer.coeffRef( index, index ) = v.myContainer( index );
  return diag_v;
}

template <typename Calculus>
typename Calculus::PrimalIdentity0 square( const Calculus& calculus,
                                           const typename Calculus::PrimalIdentity0& B )
{
  typename Calculus::PrimalIdentity0 tB_B = calculus.template identity<0, PRIMAL>();
  tB_B.myContainer = B.myContainer.transpose() * B.myContainer;
  return tB_B;
}

template <typename Calculus>
typename Calculus::PrimalIdentity0 square( const Calculus& calculus,
                                           const typename Calculus::PrimalDerivative0& B )
{
  typename Calculus::PrimalIdentity0 tB_B = calculus.template identity<0, PRIMAL>();
  tB_B.myContainer = B.myContainer.transpose() * B.myContainer;
  return tB_B;
}

template <typename Calculus>
typename Calculus::PrimalIdentity1 square( const Calculus& calculus,
                                           const typename Calculus::PrimalIdentity1& B )
{
  typename Calculus::PrimalIdentity1 tB_B = calculus.template identity<1, PRIMAL>();
  tB_B.myContainer = B.myContainer.transpose() * B.myContainer;
  return tB_B;
}

template <typename Calculus>
double innerProduct( const Calculus& calculus,
                     const typename Calculus::PrimalForm0& u,
                     const typename Calculus::PrimalForm0& v )
{
  double val = 0.0;
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++ )
    val += u.myContainer( index ) * v.myContainer( index );
  return val;
}

template <typename Calculus>
double innerProduct( const Calculus& calculus,
                     const typename Calculus::PrimalForm1& u,
                     const typename Calculus::PrimalForm1& v )
{
  double val = 0.0;
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++ )
    val += u.myContainer( index ) * v.myContainer( index );
  return val;
}

template <typename Calculus>
double innerProduct( const Calculus& calculus,
                     const typename Calculus::PrimalForm2& u,
                     const typename Calculus::PrimalForm2& v )
{
  double val = 0.0;
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++ )
    val += u.myContainer( index ) * v.myContainer( index );
  return val;
}

namespace DGtal {
  template <typename TComponent, DGtal::Dimension TM, DGtal::Dimension TN>
  bool
  operator!=( const DGtal::SimpleMatrix< TComponent, TM, TN >& m1,
          const DGtal::SimpleMatrix< TComponent, TM, TN >& m2 )
  {
    return ! ( m1.operator==( m2 ) );
  }
}

int main( int argc, char* argv[] )
{
  using namespace Z2i;
    typedef ImageContainerBySTLVector<Domain, Color>            ColorImage;
  typedef ImageContainerBySTLVector<Domain, unsigned char>      Image;
  typedef ImageContainerBySTLVector<Domain, double>             ImageDouble;
  typedef ImageContainerBySTLVector<Domain,
                    SimpleMatrix<double,2,2> >                  ImageSimpleMatrix2d;
  typedef ImageContainerBySTLVector<Domain, RealVector>         ImageVector;

  typedef std::vector< unsigned char >::iterator 				ImageIterator;
  typedef std::vector< double >::iterator 						ImageDoubleIterator;
  typedef std::vector< SimpleMatrix<double,2,2> >::iterator     ImageSimpleMatrix2dIterator;
  typedef std::vector< RealVector >::iterator 					ImageVectorIterator;

    // parse command line ----------------------------------------------
    namespace po = boost::program_options;
    po::options_description general_opt("Allowed options are: ");
    general_opt.add_options()
      ("help,h", "display this message")
      ("input,i", po::value<string>(), "the input image filename." )
      ("original,d", po::value<string>(), "the original image filename." )
      ("output,o", po::value<string>()->default_value( "AT" ), "the output image basename." )
      ("lambda,l", po::value<double>(), "the parameter lambda." )
      ("lambda-1,1", po::value<double>()->default_value( 0.3125 ), "the initial parameter lambda (l1)." ) // 0.3125
      ("lambda-2,2", po::value<double>()->default_value( 0.00005 ), "the final parameter lambda (l2)." )
      ("lambda-ratio,q", po::value<double>()->default_value( sqrt(2.0) ), "the division ratio for lambda from l1 to l2." )
      ("alpha-1", po::value<double>()->default_value( 1.0 ), "the parameter alpha." )
      ("alpha-2", po::value<double>()->default_value( 0.001 ), "the parameter alpha." )
      ("alpha-ratio", po::value<double>()->default_value( sqrt(2.0) ), "the parameter alpha." )
      ("epsilon,e", po::value<double>()->default_value( 1.0 ), "the initial and final parameter epsilon of AT functional at the same time." )
      ("epsilon-1", po::value<double>(), "the initial parameter epsilon." )
      ("epsilon-2", po::value<double>(), "the final parameter epsilon." )
      ("epsilon-r", po::value<double>()->default_value( 2.0 ), "sets the ratio between two consecutive epsilon values of AT functional." )
      //("gridstep,g", po::value<double>()->default_value( 1.0 ), "the parameter h, i.e. the gridstep." )
      ("nbiter,n", po::value<int>()->default_value( 10 ), "the maximum number of iterations." )
      ("sigma,s", po::value<double>()->default_value( 2.0 ), "the parameter of the first convolution." )
      ("rho,r", po::value<double>()->default_value( 3.0 ), "the parameter of the second convolution." )
      ("image-size,t", po::value<double>()->default_value( 64.0 ), "the size of the image." )
      ;


  bool parseOK=true;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, general_opt), vm);
  } catch ( const exception& ex ) {
    parseOK = false;
    cerr << "Error checking program options: "<< ex.what()<< endl;
  }
  po::notify(vm);
  if ( ! parseOK || vm.count("help") || !vm.count("input") )
    {
      cerr << "Usage: " << argv[0] << " -i toto.pgm\n"
           << "Computes the Ambrosio-Tortorelli reconstruction/segmentation of an input image."
           << endl << endl
           << " / " << endl
           << " | a.(u-g)^2 + v^2 |grad u|^2 + le.|grad v|^2 + (l/4e).(1-v)^2 " << endl
           << " / " << endl
           << "Discretized as (u 2-form, v 0-form, A vertex-edge bdry, B edge-face bdy, M vertex-edge average)" << endl
           << "E(u,v) = a(u-g)^t (u-g) +  u^t B diag(M v)^2 B^t u + l e v^t A^t A v + l/(4e) (1-v)^t (1-v)" << endl
           << endl
           << general_opt << "\n"
           << "Example: ./at-u2-v0 -i ../Images/cerclesTriangle64b02.pgm -o tmp -a 0.05 -e 1 -1 0.1 -2 0.00001 -t 20"
           << endl;
      return 1;
    }
  string f1 = vm[ "input" ].as<string>();
  string fd = vm[ "original" ].as<string>();
  string f2 = vm[ "output" ].as<string>();
  double l1  = vm[ "lambda-1" ].as<double>();
  double l2  = vm[ "lambda-2" ].as<double>();
  double lr  = vm[ "lambda-ratio" ].as<double>();
  if ( vm.count( "lambda" ) ) l1 = l2 = vm[ "lambda" ].as<double>();
  if ( l2 > l1 ) l2 = l1;
  if ( lr <= 1.0 ) lr = sqrt(2);
  double a1 = vm[ "alpha-1" ].as<double>();
  double a2 = vm[ "alpha-2" ].as<double>();
  double ar = vm[ "alpha-ratio" ].as<double>();
  double e  = vm[ "epsilon" ].as<double>();
  double e1 = vm.count( "epsilon-1" ) ? vm[ "epsilon-1" ].as<double>() : e;
  double e2 = vm.count( "epsilon-2" ) ? vm[ "epsilon-2" ].as<double>() : e;
  double er = vm[ "epsilon-r" ].as<double>();
  double t  = vm[ "image-size" ].as<double>();
  //double h  = vm[ "gridstep" ].as<double>();
  double h  = 1.0 / t;

  int    n  = vm[ "nbiter" ].as<int>();
  double s  = vm[ "sigma" ].as<double>();
  double r  = vm[ "rho" ].as<double>();

  trace.beginBlock("Reading image");
  ColorImage image = PPMReader<ColorImage>::importPPM( f1 );
  ColorImage perfect_image = PPMReader<ColorImage>::importPPM( fd );
  ColorImage end_image = image;
  trace.endBlock();

  // opening file
  const string file = f2 + ".txt";
  ofstream f(file.c_str());

  trace.beginBlock("Creating calculus");
  typedef DiscreteExteriorCalculus<2,2, EigenLinearAlgebraBackend> Calculus;
  typedef Calculus::PrimalForm0       PrimalForm0;
  typedef Calculus::PrimalForm1       PrimalForm1;
  typedef Calculus::PrimalForm2       PrimalForm2;
  typedef Calculus::PrimalDerivative0 PrimalDerivative0;
  typedef Calculus::PrimalDerivative1 PrimalDerivative1;
  typedef Calculus::PrimalDerivative2 PrimalDerivative2;
  typedef Calculus::DualDerivative0   DualDerivative0;
  typedef Calculus::DualDerivative1   DualDerivative1;
  typedef Calculus::DualDerivative2   DualDerivative2;
  typedef Calculus::PrimalHodge0      PrimalHodge0;
  typedef Calculus::PrimalHodge1      PrimalHodge1;
  typedef Calculus::PrimalHodge2      PrimalHodge2;
  typedef Calculus::DualHodge0        DualHodge0;
  typedef Calculus::DualHodge1        DualHodge1;
  typedef Calculus::DualHodge2        DualHodge2;
  typedef Calculus::PrimalIdentity0   PrimalIdentity0;
  typedef Calculus::PrimalIdentity1   PrimalIdentity1;
  typedef Calculus::PrimalIdentity2   PrimalIdentity2;
  typedef Calculus::Index             Index;
  typedef Calculus::SCell             SCell;
  Domain domain = image.domain();
  Point  p0     = domain.lowerBound(); p0 *= 2;
  Point  p1     = domain.upperBound(); p1 *= 2;
         p1    += Point::diagonal(2);
  Domain kdomain( p0, p1 );
  Image dbl_image( kdomain );
  Calculus calculus;
  calculus.initKSpace( ConstAlias<Domain>( domain ) );
  const KSpace& K = calculus.myKSpace;
  // Les pixels sont des 0-cellules du primal.
  for ( Domain::ConstIterator it = kdomain.begin(), itE = kdomain.end(); it != itE; ++it )
    calculus.insertSCell( K.sCell( *it ) ); // ajoute toutes les cellules de Khalimsky.
  calculus.updateIndexes();
  trace.info() << calculus << endl;
//  Calculus::PrimalForm2 g( calculus );
//  for ( Calculus::Index index = 0; index < g.myContainer.rows(); index++)
//    {
//      const Calculus::SCell& cell = g.getSCell( index );
//      g.myContainer( index ) = ((double) image( K.sCoords( cell ) )) /
//        255.0;
//    }
  vector<PrimalForm2> g;
  g.push_back( PrimalForm2( calculus ) );
  g.push_back( PrimalForm2( calculus ) );
  g.push_back( PrimalForm2( calculus ) );
  for ( Index index = 0; index < g[ 0 ].myContainer.rows(); index++)
    {
      SCell cell = g[ 0 ].getSCell( index );
      Color  col = image( K.sCoords( cell ) );
      g[ 0 ].myContainer( index ) = ( (double) col.red()   ) / 255.0;
      g[ 1 ].myContainer( index ) = ( (double) col.green() ) / 255.0;
      g[ 2 ].myContainer( index ) = ( (double) col.blue()  ) / 255.0;
    }

  vector<PrimalForm2> perfect_g;
  perfect_g.push_back( PrimalForm2( calculus ) );
  perfect_g.push_back( PrimalForm2( calculus ) );
  perfect_g.push_back( PrimalForm2( calculus ) );
  for ( Index index = 0; index < perfect_g[ 0 ].myContainer.rows(); index++)
    {
      SCell cell = perfect_g[ 0 ].getSCell( index );
      Color  col = perfect_image( K.sCoords( cell ) );
      perfect_g[ 0 ].myContainer( index ) = ( (double) col.red()   ) / 255.0;
      perfect_g[ 1 ].myContainer( index ) = ( (double) col.green() ) / 255.0;
      perfect_g[ 2 ].myContainer( index ) = ( (double) col.blue()  ) / 255.0;
    }
  trace.endBlock();

  // u = g at the beginning
  trace.info() << "u" << endl;
  vector<PrimalForm2> u( g );
  // v = 1 at the beginning
  trace.info() << "v" << endl;
  Calculus::PrimalForm0 v( calculus );
  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    v.myContainer( index ) = 1.0;
  Calculus::PrimalForm0 former_v = v;
  trace.info() << "v1 = 1" << endl;
  Calculus::PrimalForm1 v1( calculus );
  for ( Index index = 0; index < v1.myContainer.rows(); index++) 
    v1.myContainer( index ) = 1.0;
  Index nb0   = u[ 0 ].myContainer.rows();
  Index nb1   = v.myContainer.rows();

  trace.beginBlock("building AT functionnals");
  trace.info() << "primal_D0" << endl;
  const PrimalDerivative0 primal_D0 = calculus.derivative<0,PRIMAL>();
  trace.info() << "primal_D1" << endl;
  const PrimalDerivative1 primal_D1 = calculus.derivative<1,PRIMAL>();
  trace.info() << "dual_D0" << endl;
  const DualDerivative0   dual_D0   = calculus.derivative<0,DUAL>();
  trace.info() << "dual_D1" << endl;
  const DualDerivative1   dual_D1   = calculus.derivative<1,DUAL>();
  trace.info() << "primal_h0" << endl;
  const PrimalHodge0      primal_h0 = calculus.hodge<0,PRIMAL>();
  trace.info() << "primal_h1" << endl;
  const PrimalHodge1      primal_h1 = calculus.hodge<1,PRIMAL>();
  trace.info() << "primal_h2" << endl;
  const PrimalHodge2      primal_h2 = calculus.hodge<2,PRIMAL>();
  trace.info() << "dual_h1" << endl;
  const DualHodge1        dual_h1   = calculus.hodge<1,DUAL>();
  trace.info() << "dual_h2" << endl;
  const DualHodge2        dual_h2   = calculus.hodge<2,DUAL>();
//  trace.endBlock();

  // point_to_edge average operator
  Calculus::PrimalDerivative0   M01 = calculus.derivative<0, PRIMAL>();
  M01.myContainer = .5 * M01.myContainer.cwiseAbs();
  // edge_to_face average operator
  Calculus::PrimalDerivative1   M12 = calculus.derivative<1, PRIMAL>();
  M12.myContainer = .25 * M12.myContainer.cwiseAbs(); 
  const Calculus::PrimalAntiderivative2 primal_AD2 = dual_h1 * dual_D0 * primal_h2;

  trace.endBlock();

  const Calculus::PrimalIdentity0 Id0 = calculus.identity<0, PRIMAL>();
  const Calculus::PrimalIdentity1 Id1 = calculus.identity<1, PRIMAL>();
  const Calculus::PrimalIdentity2 Id2 = calculus.identity<2, PRIMAL>();

  typedef Calculus::PrimalDerivative0::Container Matrix;


  const Calculus::PrimalIdentity0  lap_operator_v = -1.0 * dual_h2 * dual_D1 * primal_h1 * primal_D0;

  // SparseLU is so much faster than SparseQR
  // SimplicialLLT is much faster than SparseLU
  // typedef EigenLinearAlgebraBackend::SolverSparseQR LinearAlgebraSolver;
  // typedef EigenLinearAlgebraBackend::SolverSparseLU LinearAlgebraSolver;
  typedef EigenLinearAlgebraBackend::SolverSimplicialLLT LinearAlgebraSolver;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 2, PRIMAL, 2, PRIMAL> SolverU;
  SolverU solver_u;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 0, PRIMAL, 0, PRIMAL> SolverV;
  SolverV solver_v;

  for ( double a = a1; a >= a2; a /= ar )
    {
    //f << tronc(a,5);

    // Building alpha_G0_1
    //  const Calculus::PrimalIdentity2       alpha_Id2 = a * Id2;
    //  const Calculus::PrimalForm2           alpha_g   = a * g;

    const PrimalIdentity2 alpha_Id2   = a * Id2;
    vector<PrimalForm2> alpha_g;
    alpha_g.push_back( alpha_Id2 * g[ 0 ] );
    alpha_g.push_back( alpha_Id2 * g[ 1 ] );
    alpha_g.push_back( alpha_Id2 * g[ 2 ] );

    for ( double l = l1; l >= l2; l /= lr )
    //while ( l1 >= l2 )
    {
      trace.info() << "************ lambda = " << l1 << " **************" << endl;
      //double l = l1;
      trace.info() << "B'B'" << endl;
      const PrimalIdentity0 l_LAPV = l * lap_operator_v;
      PrimalForm0 l_1_over_4( calculus );
      for ( Index index = 0; index < l_1_over_4.myContainer.rows(); index++)
        l_1_over_4.myContainer( index ) = l / 4.0;

      double last_eps = e1;
      for ( double eps = e1; eps >= e2; eps /= er )
        {
          trace.info() << "---------------------------------------------------------------" << endl;
          trace.info() << "--------------- eps = " << eps << " --------------------" << endl;
          last_eps = eps;
          PrimalIdentity0 Per_Op      =  eps * l_LAPV + ( l/(4.0*eps) ) * Id0;
          PrimalForm0     l_1_over_4e = (1.0/eps) * l_1_over_4;

          int i = 0;
          for ( ; i < n; ++i )
            {
              trace.info() << "------ Iteration " << i << "/" << n << " ------" << endl;
              trace.beginBlock("Solving for u as a 2-form");
              trace.info() << "E(u,v) = a(u-g)^t (u-g) +  u^t B diag(M v)^2 B^t u + l e v^t A^t A v + l/(4e) (1-v)^t (1-v)" << endl;
              trace.info() << "dE/du  = 2( a Id (u-g) + B diag(M v)^2 B^t u )" << endl;
              trace.info() << "Building matrix U =  [ a Id + B diag(M v)^2 B^t ]" << endl;
              PrimalIdentity1 diag_v1 = diag( calculus, v1 );
              PrimalIdentity2 M_Id2   = -1.0 * primal_D1 * diag_v1 * diag_v1 * primal_AD2
                                        + alpha_Id2;
              trace.info() << "Prefactoring matrix U" << endl;
              solver_u.compute( M_Id2 );

              for ( Dimension i = 0; i < 3; ++i )
                {
                    u[i] = solver_u.solve( alpha_g[i] );
                    trace.info() << "  => " << ( solver_u.isValid() ? "OK" : "ERROR" )
                                 << " " << solver_u.myLinearAlgebraSolver.info() << endl;
                }
              trace.endBlock();


              // E(u,v) = a(u-g)^t (u-g) +  u^t B diag(M v)^2 B^t u + l e v^t A^t A v + l/(4e) (1-v)^t (1-v)
              // dE/dv  = 2( M^t diag( B^t u )^2 M v  + l e A^t A v  - l/4e Id (1-v) )
              //  dE/dv = 0 <=> [ M^t diag( B^t u )^2 M + l e A^t A  + l/4e Id ] v = l/4e 1
              trace.beginBlock("Solving for v");
              former_v = v;
              trace.info() << "E(u,v) = a(u-g)^t (u-g) +  u^t B diag(M v)^2 B^t u + l e v^t A^t A v + l/(4e) (1-v)^t (1-v)" << endl;
              trace.info() << " 2( M^t diag( B^t u )^2 M v  + l e A^t A v  - l/4e Id (1-v) )" << endl;
              trace.info() << "Building matrix V = [ M^t diag( B^t u )^2 M + l e A^t A  + l/4e Id ]" << endl;
              PrimalIdentity0 N_Id0 = Per_Op;
              for ( Dimension i = 0; i < 3; ++i )
                {
                    const PrimalIdentity1 diag_Bt_u = diag( calculus, primal_AD2 * u[i] );
                    N_Id0.myContainer += ( M01.transpose() * diag_Bt_u * diag_Bt_u * M01).myContainer;
                }
              trace.info() << "Prefactoring matrix V" << endl;
              solver_v.compute( N_Id0 );
              trace.info() << "Solving V v = l/4e * 1" << endl;
              v = solver_v.solve( l_1_over_4e );
              trace.info() << "  => " << ( solver_v.isValid() ? "OK" : "ERROR" )
                           << " " << solver_v.myLinearAlgebraSolver.info() << endl;
              trace.info() << "Projecting v0 onto v1" << endl;
              v1 = M01 * v;
              trace.endBlock();

              trace.beginBlock("Checking v, computing norms");
              double m1 = 1.0;
              double m2 = 0.0;
              double ma = 0.0;
              for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
                {
                  double val = v.myContainer( index );
                  m1 = std::min( m1, val );
                  m2 = std::max( m2, val );
                  ma += val;
                }
              trace.info() << "1-form v: min=" << m1 << " avg=" << ( ma/ v.myContainer.rows() )
                           << " max=" << m2 << std::endl;
              for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
                v.myContainer( index ) = std::min( std::max(v.myContainer( index ), 0.0) , 1.0 );

              double n_infty = 0.0;
              double n_2 = 0.0;
              double n_1 = 0.0;
                  
              for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
                {
                  n_infty = max( n_infty, abs( v.myContainer( index ) - former_v.myContainer( index ) ) );
                  n_2    += ( v.myContainer( index ) - former_v.myContainer( index ) )
                    * ( v.myContainer( index ) - former_v.myContainer( index ) );
                  n_1    += abs( v.myContainer( index ) - former_v.myContainer( index ) );
                }
              n_1 /= v.myContainer.rows();
              n_2 = sqrt( n_2 / v.myContainer.rows() );

              trace.info() << "Variation |v^k+1 - v^k|_oo = " << n_infty << endl;
              trace.info() << "Variation |v^k+1 - v^k|_2 = " << n_2 << endl;
              trace.info() << "Variation |v^k+1 - v^k|_1 = " << n_1 << endl;
              trace.endBlock();
              if ( n_infty < 1e-4 ) break;
            }
        }
  
      // affichage des energies ********************************************************************

      trace.beginBlock("Computing energies");

//       // a(u-g)^2
//       Calculus::PrimalIdentity2 diag_alpha = a * Id2;
//       double alpha_square_u_minus_g = 0.0;
//       for ( Dimension i = 0; i < 3; ++i )
//         {
//           const PrimalForm2 u_minus_g = u[ i ] - g[ i ];
//           alpha_square_u_minus_g += innerProduct( calculus, diag_alpha * u_minus_g, u_minus_g );
//         }
//       trace.info() << "- a(u-g)^2      = " << alpha_square_u_minus_g << std::endl;
//
//       // v^2|grad u|^2
//       const Calculus::PrimalIdentity1 diag_v1 = diag( calculus, v1 );
//       double square_v1_grad_u = 0.0;
//       for ( Dimension i = 0; i < 3; ++i )
//         {
//           const Calculus::PrimalForm1 v1_A_u = diag_v1 * primal_AD2 * u[ i ];
//           square_v1_grad_u += innerProduct( calculus, v1_A_u, v1_A_u );
//         }
//       trace.info() << "- v^2|grad u|^2 = " << square_v1_grad_u << std::endl;

//       // le|grad v|^2
//       Calculus::PrimalForm0 v_prime = lap_operator_v * v;
//       double le_square_grad_v = l * last_eps * innerProduct( calculus, v, v_prime );
//       trace.info() << "- le|grad v|^2  = " << le_square_grad_v << std::endl;

//       // l(1-v)^2/4e
//       Calculus::PrimalForm0 one_minus_v = v;
//       for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
//         one_minus_v.myContainer( index_i ) = 1.0 - one_minus_v.myContainer( index_i );
//       double l_over_4e_square_1_minus_v
//         = l / (4.0*last_eps) * innerProduct( calculus, one_minus_v, one_minus_v );
//       trace.info() << "- l(1-v)^2/4e   = " << l_over_4e_square_1_minus_v << std::endl;

//       // l.per
//       double Lper = le_square_grad_v + l_over_4e_square_1_minus_v;
//       trace.info() << "- l.per         = " << Lper << std::endl;

//       // AT tot
//       double ATtot = alpha_square_u_minus_g + square_v1_grad_u + Lper;

       // (u-perfect_g)^2
      double u_minus_perfect_g_square = 0.0;
      for ( Dimension i = 0; i < 3; ++i )
        {
          const PrimalForm2 u_minus_perfect_g = u[ i ] - perfect_g[ i ];
          u_minus_perfect_g_square += innerProduct( calculus, u_minus_perfect_g, u_minus_perfect_g );
        }

        f << a
          << "\t" << l
          << "\t" << tronc(u_minus_perfect_g_square,5) //tronc(alpha_square_u_minus_g,5);
          << endl;


      trace.endBlock();

      // ***********************************************************************************************************************

      int int_l = (int) floor(l);
      int dec_l = (int) (floor((l-floor(l))*10000000));

      ostringstream ossU;
      ossU << boost::format("%s-l%.7f-a%.7f-u.ppm") %f2 %l %a;
      string str_image_u = ossU.str();
      savePrimalForms2ToPPMImage( calculus, end_image, u[ 0 ], u[ 1 ], u[ 2 ], str_image_u);

//      // ostringstream ossV;
//      // ossV << boost::format("%s-l%.7f-v.pgm") %f2 %l;
//      // string str_image_v = ossV.str();
//      // savePrimalForm0ToImage( calculus, end_image, v, str_image_v);

//      ostringstream ossV1;
//      ossV1 << boost::format("%s-l%.7f-v1.pgm") %f2 %l;
//      string str_image_v1 = ossV1.str();
//      savePrimalForm1ToImage( calculus, dbl_image, v1, str_image_v1 );

      ostringstream ossU2V0;
      ossU2V0 << boost::format("%s-l%.7f-a%.7f-u2-v0.eps") %f2 %l %a;
      string str_image_u2_v0 = ossU2V0.str();
      saveFormsToEps( calculus, u[ 0 ], u[ 1 ], u[ 2 ], v, str_image_u2_v0 );

//      ostringstream ossGV0;
//      ossGV0 << boost::format("%s-l%.7f-g-v0.eps") %f2 %l;
//      string str_image_g_v0 = ossGV0.str();
//      saveFormsToEps( calculus, g[ 0 ], g[ 1 ], g[ 2 ], v, str_image_g_v0 );

    }
  }


  f.close();

  return 0;
}



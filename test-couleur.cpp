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
#include "DGtal/math/linalg/EigenSupport.h"
#include "DGtal/dec/DiscreteExteriorCalculus.h"
#include "DGtal/dec/DiscreteExteriorCalculusSolver.h"

// RealFFT
#include <DGtal/kernel/domains/HyperRectDomain.h>
#include <DGtal/kernel/SpaceND.h>
#include <DGtal/images/ImageContainerBySTLVector.h>
#include "VTKWriter.h"

// StructureTensor
#include "structureTensor.h"


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
void savePrimalForm0ToImage( const Calculus& calculus, const Image& image, const typename Calculus::PrimalForm0& u, const string& filename )
{
    Image end_image = image;
    PrimalForm0ToImage( calculus, u, end_image );
    ostringstream ossU;
    ossU << filename;
    string str_image_u = ossU.str();
    //end_image >> str_image_u.c_str();


//    PPMWriter<I,C>::exportPPM(const std::string & filename, const I & aImage,
//                  const Functor & aFunctor, bool topbotomOrder)
    //PPMWriter<Image,Calculus>::exportPPM( str_image_u.c_str(), end_image, DGtal::functors::ColorRGBEncoder<unsigned char>, false );
typedef functors::IdentityFunctor<Color> IdColorFct;
    //DGtal::functors::ColorRGBEncoder<unsigned char> rgb();
    PPMWriter<Image,IdColorFct>::exportPPM( str_image_u.c_str(), end_image, IdColorFct(), false );

//    DGtal::functors::Identity idV;
//    DGtal::HueShadeColorMap<unsigned char> hscm;
//    DGtal::GenericWriter<Image, 2, unsigned char, DGtal::functors::Identity>
//            ::exportFile( str_image_u.c_str(), end_image, idV );
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

template <typename Calculus, typename ImageDouble>
double interpolation(const Calculus& calculus, const ImageDouble& I, int const& i, int const& j, double const& x, double const& y)
//double interpolation(const ImageDouble& I, typename Calculus::SCell const& cell, RealVector const& n)
{

    typedef typename Calculus::KSpace KSpace;
    typedef typename KSpace::Point Point;

    const KSpace& K = calculus.myKSpace;
    const int n  = K.size(0); // = w-1
    const int m  = K.size(1); // = h-1

    // construction of the square {P0, P1, P2, P3} which contains (i+x,j+y) s.t.
    // P0 = (i,j), P1 = (i+/-1,j), P2 = (i,j+/-1), P3 = (i+/-1,j+/-1)
    int i1 = 0, i2 = 0, i3 = 0;
    int j1 = 0, j2 = 0, j3 = 0;


    if( x >= 0 && y >= 0)
    {
        i1 = std::min(n, std::max(0,i+1));  j1 = j;                             //const Point P1 (i+1,j  );
        i2 = i;                             j2 = std::min(m, std::max(0,j-1));  //const Point P2 (i  ,j-1);
        i3 = std::min(n, std::max(0,i+1));  j3 = std::min(m, std::max(0,j-1));  //const Point P3 (i+1,j-1);
    }
    else if( (x >= 0 && y < 0) || (i==0 && j < n) )
    {
        i1 = std::min(n, std::max(0,i+1));  j1 = j;                             //const Point P1 (i+1,j  );
        i2 = i;                             j2 = std::min(m, std::max(0,j+1));  //const Point P2 (i  ,j+1);
        i3 = std::min(n, std::max(0,i+1));  j3 = std::min(m, std::max(0,j+1));  //const Point P3 (i+1,j+1);
    }
    else if( x < 0 && y >= 0)
    {
        i1 = std::min(n, std::max(0,i-1));  j1 = j;                             //const Point P1 (i-1,j  );
        i2 = i;                             j2 = std::min(m, std::max(0,j-1));  //const Point P2 (i  ,j-1);
        i3 = std::min(n, std::max(0,i-1));  j3 = std::min(m, std::max(0,j-1));  //const Point P3 (i-1,j-1);
    }
    else if( x < 0 && y < 0)
    {
        i1 = std::min(n, std::max(0,i-1));  j1 = j;                             //const Point P1 (i-1,j  );
        i2 = i;                             j2 = std::min(m, std::max(0,j+1));  //const Point P2 (i  ,j+1);
        i3 = std::min(n, std::max(0,i-1));  j3 = std::min(m, std::max(0,j+1));  //const Point P3 (i-1,j+1);
    }

    const Point P0 (i ,j );
    const Point P1 (i1,j1);
    const Point P2 (i2,j2);
    const Point P3 (i3,j3);


//    std::cout << " (i,j)= ("      << i << "," << j << ")"
//              << " (x,y)= ("      << x << "," << y << ")"
//              << " P(i,j)= "      << P0
//              << " P(i+1,j)= "    << P1
//              << " P(i,j+1)= "    << P2
//              << " P(i+1,j+1)= "  << P3
//              << std::endl;


    const double a = std::fabs(x);
    const double b = std::fabs(y);

    double val =   (1.0-a)*(1.0-b) * I(P0)
                 +    a   *(1.0-b) * I(P1)
                 + (1.0-a)*   b    * I(P2)
                 +    a   *   b    * I(P3);

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
  typedef ImageContainerBySTLVector<Domain, unsigned char>      Image;
  typedef ImageContainerBySTLVector<Domain, double>             ImageDouble;
  typedef ImageContainerBySTLVector<Domain,
                    SimpleMatrix<double,2,2> >  ImageSimpleMatrix2d;
  typedef ImageContainerBySTLVector<Domain, RealVector>         ImageVector;

  typedef std::vector< unsigned char >::iterator 			ImageIterator;
  typedef std::vector< double >::iterator 					ImageDoubleIterator;
  typedef std::vector< SimpleMatrix<double,2,2> >::iterator ImageSimpleMatrix2dIterator;
  typedef std::vector< RealVector >::iterator 				ImageVectorIterator;

  // parse command line ----------------------------------------------
  namespace po = boost::program_options;
  po::options_description general_opt("Allowed options are: ");
  general_opt.add_options()
    ("help,h", "display this message")
    ("input,i", po::value<string>(), "the input image filename." )
    ("output,o", po::value<string>()->default_value( "AT" ), "the output image basename." )
    ("lambda,l", po::value<double>(), "the parameter lambda." )
    ("lambda-1,1", po::value<double>()->default_value( 0.3125 ), "the initial parameter lambda (l1)." ) // 0.3125
    ("lambda-2,2", po::value<double>()->default_value( 0.00005 ), "the final parameter lambda (l2)." )
    ("lambda-ratio,q", po::value<double>()->default_value( sqrt(2) ), "the division ratio for lambda from l1 to l2." )
    ("alpha,a", po::value<double>()->default_value( 1.0 ), "the parameter alpha." )
    ("epsilon,e", po::value<double>()->default_value( 4.0/64.0 ), "the initial parameter epsilon." )
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
       << " / "
       << endl
       << " | a.(u-g)^2 + v^2 |grad u|^2 + le.|grad v|^2 + (l/4e).(1-v)^2 "
       << endl
       << " / "
       << endl << endl
       << general_opt << "\n";
      return 1;
    }
  string f1 = vm[ "input" ].as<string>();
  string f2 = vm[ "output" ].as<string>();
  double l1  = vm[ "lambda-1" ].as<double>();
  double l2  = vm[ "lambda-2" ].as<double>();
  double lr  = vm[ "lambda-ratio" ].as<double>();
  if ( vm.count( "lambda" ) ) l1 = l2 = vm[ "lambda" ].as<double>();
  if ( l2 > l1 ) l2 = l1;
  if ( lr <= 1.0 ) lr = sqrt(2);
  double a  = vm[ "alpha" ].as<double>();
  double e  = vm[ "epsilon" ].as<double>();
  double t  = vm[ "image-size" ].as<double>();
  //double h  = vm[ "gridstep" ].as<double>();
  double h  = 1.0 / t;

  int    n  = vm[ "nbiter" ].as<int>();
  double s  = vm[ "sigma" ].as<double>();
  double r  = vm[ "rho" ].as<double>();

  trace.beginBlock("Reading image");
  Image image = GenericReader<Image>::import( f1 ); // Image image = GenericReader<Image>::importWithColorFunctor<TFunctor>( f1 ) ?
  Image end_image = image;
  trace.endBlock();

  trace.beginBlock("Creating calculus");
  typedef DiscreteExteriorCalculus<2,2, EigenLinearAlgebraBackend> Calculus;
  Domain domain = image.domain();
  Point p0 = domain.lowerBound(); p0 *= 2;
  Point p1 = domain.upperBound(); p1 *= 2;
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
  Calculus::PrimalForm0 g( calculus );
  double min_g = 300.0, max_g = 0.0;
  for ( Calculus::Index index = 0; index < g.myContainer.rows(); index++)
    {
      const Calculus::SCell& cell = g.getSCell( index );
      g.myContainer( index ) = ((double) image( K.sCoords( cell ) )) /
    255.0;
      min_g = std::min( min_g , g.myContainer( index ) );
      max_g = std::max( max_g , g.myContainer( index ) );
    }
  trace.info() << "min_g= " << min_g << " max_g= " << max_g << std::endl;
  trace.endBlock();

  // u = g at the beginning
  trace.info() << "u" << endl;
  Calculus::PrimalForm0 u = g;


  ostringstream ossU;
  ossU << boost::format("test-u.ppm");
  string str_image_u = ossU.str();
  savePrimalForm0ToImage( calculus, end_image, u, str_image_u);

  return 0;
}



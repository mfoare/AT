

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

double tronc( const double& nb, const int& p )
{
  int i = pow(10,p) * nb;
  return i/pow(10,p);
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

  typedef std::vector< unsigned char >::iterator 						ImageIterator;
  typedef std::vector< double >::iterator 									ImageDoubleIterator;
  typedef std::vector< SimpleMatrix<double,2,2> >::iterator ImageSimpleMatrix2dIterator;
  typedef std::vector< RealVector >::iterator 							ImageVectorIterator;

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
  Image image = GenericReader<Image>::import( f1 );
  trace.endBlock();

  // opening file
  const string file = f2 + ".txt";
  ofstream f(file.c_str());
  f << "#  l \t" << " a \t" << " e \t" << "a(u-g)^2 \t" << "v^2|grad u|^2 \t" << "  le|grad v|^2 \t" << "  l(1-v)^2/4e \t" << " l.per \t" << "AT tot"<< endl;

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
  for ( Calculus::Index index = 0; index < g.myContainer.rows(); index++)
    {
      const Calculus::SCell& cell = g.getSCell( index );
      g.myContainer( index ) = ((double) image( K.sCoords( cell ) )) /
    255.0;
    }
  trace.endBlock();

  // u = g at the beginning
  trace.info() << "u" << endl;
  Calculus::PrimalForm0 u = g;
  // v = 1 at the beginning
  trace.info() << "v" << endl;
  Calculus::PrimalForm1 v( calculus );
  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    v.myContainer( index ) = 1.0;

  trace.beginBlock("building AT functionnals");
  trace.info() << "primal_D0" << endl;
  const Calculus::PrimalDerivative0 	primal_D0 = calculus.derivative<0,PRIMAL>();
  trace.info() << "primal_D1" << endl;
  const Calculus::PrimalDerivative1 	primal_D1 = calculus.derivative<1,PRIMAL>();
  trace.info() << "dual_D0" << endl;
  const Calculus::DualDerivative0       dual_D0   = calculus.derivative<0,DUAL>();
  trace.info() << "dual_D1" << endl;
  const Calculus::DualDerivative1 	dual_D1   = calculus.derivative<1,DUAL>();
  trace.info() << "primal_h0" << endl;
  const Calculus::PrimalHodge0  	primal_h0 = calculus.hodge<0,PRIMAL>();
  trace.info() << "primal_h1" << endl;
  const Calculus::PrimalHodge1  	primal_h1 = calculus.hodge<1,PRIMAL>();
  trace.info() << "primal_h2" << endl;
  const Calculus::PrimalHodge2     	primal_h2 = calculus.hodge<2,PRIMAL>();
  trace.info() << "dual_h1" << endl;
  const Calculus::DualHodge1         	dual_h1   = calculus.hodge<1,DUAL>();
  trace.info() << "dual_h2" << endl;
  const Calculus::DualHodge2      	dual_h2   = calculus.hodge<2,DUAL>();

  const DGtal::Dimension dimX = 0, dimY = 1;
  
  // BEG JOL
  // Calculus::PrimalIdentity1 tSS = calculus.identity<1, PRIMAL>();
  // for ( Calculus::Index index_i = 0; index_i < tSS.myContainer.rows(); index_i++ )
  //   for ( Calculus::Index index_j = 0; index_j < tSS.myContainer.cols(); index_j++ )
  //     {
  //       tSS.myContainer.coeffRef( index_i, index_j ) = 0.0;
  //       for ( Calculus::Index index_k = 0; index_k < tSS.myContainer.rows(); index_k++ )
  //           tSS.myContainer.coeffRef( index_i, index_j ) +=  sharp_x.myContainer.coeffRef( index_k, index_i ) * sharp_x.myContainer.coeffRef( index_k, index_j )
  //                                                          + sharp_y.myContainer.coeffRef( index_k, index_i ) * sharp_y.myContainer.coeffRef( index_k, index_j ) ;
  //     }
  // END JOL
  trace.endBlock();


  // Weight matrices
  //  Calculus::DualIdentity2   G0 		= ( 1.0/(h*h) ) * calculus.identity<2, DUAL>();
  Calculus::PrimalIdentity0 G0 		= ( 1.0/(h*h) ) * calculus.identity<0, PRIMAL>();
  Calculus::PrimalIdentity0 invG0   = 		(h*h) 	* calculus.identity<0, PRIMAL>();

  //  Calculus::DualIdentity1   G1 		= calculus.identity<1, DUAL>();
  Calculus::PrimalIdentity1 G1 		= calculus.identity<1, PRIMAL>();
  Calculus::PrimalIdentity1 invG1   = calculus.identity<1, PRIMAL>();
  
  //  Calculus::DualIdentity0   G2 		= 		(h*h) 	* calculus.identity<0, DUAL>();
  Calculus::PrimalIdentity2 G2 		= 		(h*h) 	* calculus.identity<2, PRIMAL>();
  Calculus::PrimalIdentity2 invG2   = ( 1.0/(h*h) ) * calculus.identity<2, PRIMAL>();

  Calculus::PrimalForm1 vG1( calculus );

  typedef Calculus::PrimalDerivative0::Container Matrix;
  
  // Building alpha_G0_1
  const Calculus::PrimalIdentity0 alpha_iG0 = a * invG0;
  const Calculus::PrimalForm0 alpha_iG0_g   = alpha_iG0 * g;

  // Building tA_A
  const Matrix& A = primal_D0.myContainer;
  const Matrix tA = A.transpose();
  Calculus::PrimalIdentity1 tA_A = G1;
  G1.myContainer = tA * A;

  // Building tS_S
  Calculus::PrimalAntiderivative1   sharp_x   = calculus.sharpDirectional<PRIMAL>(dimX);
  Calculus::PrimalAntiderivative1   sharp_y   = calculus.sharpDirectional<PRIMAL>(dimY);
  const Calculus::PrimalDerivative0 flat_x    = calculus.flatDirectional<PRIMAL>(dimX);
  const Calculus::PrimalDerivative0 flat_y    = calculus.flatDirectional<PRIMAL>(dimY);
  const Matrix& Sx = sharp_x.myContainer; 
  const Matrix tSx = Sx.transpose();
  const Matrix& Sy = sharp_y.myContainer; 
  const Matrix tSy = Sy.transpose();
  Calculus::PrimalIdentity1 tS_S = G1;
  tS_S.myContainer = (tSx * Sx + tSy * Sy); // (1.0/(h*h))*(tSx * Sx + tSy * Sy);
  
  // const Matrix& M = tS_S.myContainer;
  // for (int k = 0; k < M.outerSize(); ++k)
  //   for ( Matrix::InnerIterator it( M, k ); it; ++it )
  //     trace.info() << "[" << it.row() << "," << it.col() << "] = " << it.value() << endl;

  // Building tA_tS_S_A
  Calculus::PrimalIdentity0 tA_tS_S_A = G0;
  tA_tS_S_A.myContainer = tA * tS_S.myContainer * A;

  // Building iG1_A_G0_tA_iG1 + tB_iG2_B
  const Calculus::PrimalIdentity1 lap_operator_v = -1.0 * ( invG1 * primal_D0 * G0 * dual_h2 * dual_D1 * primal_h1 * invG1
                                                            + dual_h1 * dual_D0 * primal_h2 * invG2 * primal_D1 );
  // const Calculus::PrimalIdentity1 lap_operator_v = -1.0 * ( invG1 * primal_D0 * G0 * dual_h2 * dual_D1 * primal_h1 * invG1
  //                                                           + dual_h1 * dual_D0 * primal_h2 * invG2 * primal_D1 );

  // SparseLU is so much faster than SparseQR
  // SimplicialLLT is much faster than SparseLU
  // typedef EigenLinearAlgebraBackend::SolverSparseQR LinearAlgebraSolver;
  // typedef EigenLinearAlgebraBackend::SolverSparseLU LinearAlgebraSolver;
  typedef EigenLinearAlgebraBackend::SolverSimplicialLLT LinearAlgebraSolver;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 0, PRIMAL, 0, PRIMAL> SolverU;
  SolverU solver_u;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 1, PRIMAL, 1, PRIMAL> SolverV;
  SolverV solver_v;

  while ( l1 >= l2 )
    {
      trace.info() << "************ lambda = " << l1 << " **************" << endl;
      double l = l1;
      trace.info() << "B'B'" << endl;
      const Calculus::PrimalIdentity1 lBB = l * lap_operator_v;
      Calculus::PrimalForm1 l_sur_4( calculus );
      for ( Calculus::Index index = 0; index < l_sur_4.myContainer.rows(); index++)
        l_sur_4.myContainer( index ) = l/4.0;

      double coef_eps = 2.0;
      double eps = coef_eps*e;
      
      for( int k = 0 ; k < 5 ; ++k )
        {
          if (eps/coef_eps < h*h)
            break;
          else
            {
              eps /= coef_eps;
              Calculus::PrimalIdentity1 BB = eps * lBB + ( l/(4.0*eps) ) * tS_S;
              int i = 0;
              for ( ; i < n; ++i )
                {
                  trace.info() << "------ Iteration " << k << ":" << 	i << "/" << n << " ------" << endl;
                  trace.beginBlock("Solving for u");
                  trace.info() << "Building matrix Av2A" << endl;
                  
                  double tvtSSv = 0.0;
                  Calculus::PrimalForm1 tS_S_v = tS_S * v;
                  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
                    tvtSSv += v.myContainer( index ) * tS_S_v.myContainer( index );
                  const Calculus::PrimalIdentity0 Av2A = ( ( 1.0 * tvtSSv ) * tA_tS_S_A ) + alpha_iG0;
                  trace.info() << "tvtSSv = " << tvtSSv << endl;
                  trace.info() << "Prefactoring matrix Av2A := tv_tS_S_v.tA_tS_S_A + alpha_iG0" << endl;
                  trace.info() << "-------------------------------------------------------------------------------" << endl;
                  // const Matrix& M = Av2A.myContainer;
                  // for (int k = 0; k < M.outerSize(); ++k)
                  //   for ( Matrix::InnerIterator it( M, k ); it; ++it )
                  //     trace.info() << "[" << it.row() << "," << it.col() << "] = " << it.value() << endl;
                  solver_u.compute( Av2A );
                  trace.info() << "Solving Av2A u = ag" << endl;
                  u = solver_u.solve( alpha_iG0_g );
                  trace.info() << ( solver_u.isValid() ? "OK" : "ERROR" ) << " " << solver_u.myLinearAlgebraSolver.info() << endl;
                  trace.info() << "-------------------------------------------------------------------------------" << endl;
                  trace.endBlock();

                  trace.beginBlock("Solving for v");
                  trace.info() << "Building matrix BB+Mw2" << endl;
                  const Calculus::PrimalForm1 former_v = v;
                  double tutAtSSAu = 0.0;
                  Calculus::PrimalForm0 tA_tS_S_A_u = tA_tS_S_A * u;
                  for ( Calculus::Index index = 0; index < u.myContainer.rows(); index++)
                    tutAtSSAu += u.myContainer( index ) * tA_tS_S_A_u.myContainer( index );

                  trace.info() << "Prefactoring matrix BB+Mw2" << endl;
                  trace.info() << "tutAtSSAu = " << tutAtSSAu << endl;
                  solver_v.compute( BB + tutAtSSAu * tS_S );
                  trace.info() << 	"Solving (BB+Mw2)v = l_4e" << endl;
                  v = solver_v.solve( (1.0/eps) * tS_S * l_sur_4 );
                  trace.info() << ( solver_v.isValid() ? "OK" : "ERROR" ) << " " << solver_v.myLinearAlgebraSolver.info() << endl;
                  trace.endBlock();

                  double m1 = 0.0;
                  double m2 = h;
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
                  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
                    n_infty = max( n_infty, abs( v.myContainer( index ) - former_v.myContainer( index ) ) );
                  trace.info() << "Variation |v^k+1 - v^k|_oo = " << n_infty << endl;
                  if ( n_infty < 1e-4 ) break;
                }
            }
        }

/** A REPRENDRE */
//      // affichage des energies ***********************************************************************************

//      typedef Calculus::SparseMatrix SparseMatrix;
//      typedef Eigen::Matrix<double,Dynamic,Dynamic> Matrix;

//      trace.beginBlock("Computing energies");

//      // a(u-g)^2
//      trace.info() << "- a(u-g)^2 " << std::endl;
//      double UmG2 = 0.0;
//      for ( Calculus::Index index = 0; index < u.myContainer.rows(); index++)
//        UmG2 += (a*h*h) * (u.myContainer( index ) - g.myContainer( index )) * (u.myContainer( index ) - g.myContainer( index ));

//      // v^2|grad u|^2
//      trace.info() << "- v^2|grad u|^2" << std::endl;
//      double V2gradU2 = 0.0;
//      SolverU solver_Av2A;
//      trace.info() << "  - Id" << std::endl;
//      Calculus::PrimalIdentity1 Mv2 = calculus.identity<1, PRIMAL>();
//      trace.info() << "  - M := Diag(v^2)" << std::endl;
//      for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
//        Mv2.myContainer.coeffRef( index, index ) = v.myContainer[ index ] * v.myContainer[ index ];
//      trace.info() << "  - * D_1 * M * D_0" << std::endl;
//      const Calculus::PrimalIdentity0 Av2A = - (1.0/h) * dual_h2 * dual_D1 * primal_h1 * Mv2 * invG1 * primal_D0;
//      trace.info() << "  - N := compute (* D_1 * M * D_0)" << std::endl;
//      solver_Av2A.compute( Av2A );
//      // JOL: 1000 * plus rapide !
//      trace.info() << "  - u^t N u" << std::endl;
//      Calculus::PrimalForm0 u_prime = Av2A * u;
//      for ( Calculus::Index index = 0; index < u.myContainer.rows(); index++)
//        V2gradU2 += u.myContainer( index ) * u_prime.myContainer( index );
//      // for ( Calculus::Index index_i = 0; index_i < u.myContainer.rows(); index_i++)
//      // 	for ( Calculus::Index index_j = 0; index_j < u.myContainer.rows(); index_j++)
//      //     V2gradU2 += u.myContainer( index_i ) * Av2A.myContainer.coeff( index_i,index_j ) * u.myContainer( index_j ) ;

//      // le|grad v|^2
//      trace.info() << "- le|grad v|^2" << std::endl;
//      Calculus::PrimalForm1 v_prime = lap_operator_v * v;
//      double gradV2 = 0.0;
//      for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
//        gradV2 += (l * eps / h) * v.myContainer( index ) * v_prime.myContainer( index );
//      // for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
//      //   for ( Calculus::Index index_j = 0; index_j < v.myContainer.rows(); index_j++)
//      //     gradV2 += l * eps * v.myContainer( index_i ) * lap_operator_v.myContainer.coeff( index_i,index_j ) * v.myContainer( index_j );

//      // l(1-v)^2/4e
//      trace.info() << "- l(1-v)^2/4e" << std::endl;
//      double Vm12 = 0.0;
//      for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
//        Vm12 += (l*h/(4*eps)) * (1 - 2*v.myContainer( index_i ) + v.myContainer( index_i )*v.myContainer( index_i ));

//      // l.per
//      trace.info() << "- l.per" << std::endl;
//      double Lper = gradV2 + Vm12;
//      //      double per = 0.0;
//      //      for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
//      //      {
//      //        per += (1/(4*e)) * (1 - 2*v.myContainer( index_i ) + v.myContainer( index_i )*v.myContainer( index_i ));
//      //        for ( Calculus::Index index_j = 0; index_j < v.myContainer.rows(); index_j++)
//      //            per += e * v.myContainer( index_i ) * tBB.myContainer( index_i,index_j ) * v.myContainer( index_j );
//      //      }

//      // AT tot
//      double ATtot = UmG2 + V2gradU2 + gradV2 + Vm12;

//      //f << "l  " << "  a  " << "  e  " << "  a(u-g)^2  " << "  v^2|grad u|^2  " << "  le|grad v|^2  " << "  l(1-v)^2/4e  " << "  l.per  " << "  AT tot"<< endl;
//      f << tronc(l,8) << "\t" << a << "\t"  << tronc(eps,4) << "\t"  << tronc(UmG2,5) << "\t"  << tronc(V2gradU2,5) << "\t"  << tronc(gradV2,5) << "\t" << tronc(Vm12,5) << "\t" << tronc(Lper,5)  << "\t" << tronc(ATtot,5) << endl;

//      trace.endBlock();

      // ***********************************************************************************************************************

      int int_l = (int) floor(l);
      int dec_l = (int) (floor((l-floor(l))*10000000));

      {
        // Board2D board;
        // board << calculus;
        // board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) ) << u;
        // ostringstream oss;
        // oss << f2 << "-u-" << i << ".eps";
        // string str_u = oss.str(); //f2 + "-u-" + .eps";
        // board.saveEPS( str_u.c_str() );
        Image end_image = image;
        PrimalForm0ToImage( calculus, u, end_image );
        ostringstream ossU;
        //ossU << f2 << "-l" << int_l << "_" << dec_l << "-u.pgm";
        ossU << boost::format("%s-l%.7f-u.pgm") %f2 %l;
        string str_image_u = ossU.str();
        end_image >> str_image_u.c_str();
      }
      {
        // Board2D board;
        // board << calculus;
        // board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) )
        //       << v;
        // ostringstream oss;
        // oss << f2 << "-v-" << i << ".eps";
        // string str_v = oss.str();
        // board.saveEPS( str_v.c_str() );
        PrimalForm1ToImage( calculus, v, dbl_image );
        ostringstream ossV;
        //ossV << f2 << "-l" << int_l << "_" << dec_l << "-v.pgm";
        ossV << boost::format("%s-l%.7f-v.pgm") %f2 %l;
        string str_image_v = ossV.str();
        dbl_image >> str_image_v.c_str();
      }
      l1 /= lr;
    }
  // typedef SelfAdjointEigenSolver<MatrixXd> EigenSolverMatrix;
  // const EigenSolverMatrix eigen_solver(laplace.myContainer);

  // const VectorXd eigen_values = eigen_solver.eigenvalues();
  // const MatrixXd eigen_vectors = eigen_solver.eigenvectors();

  // for (int kk=0; kk<laplace.myContainer.rows(); kk++)
  // {
  //     Calculus::Scalar eigen_value = eigen_values(kk, 0);

  //     const VectorXd eigen_vector = eigen_vectors.col(kk);
  //     const Calculus::DualForm0 eigen_form = Calculus::DualForm0(calculus, eigen_vector);
  //     std::stringstream ss;
  //     ss << "chladni_eigen_" << kk << ".svg";
  //     const std::string filename = ss.str();
  //     ss << "chladni_eigen_vector_" << kk << ".svg";
  //     trace.info() << kk << " " << eigen_value << " " << sqrt(eigen_value) << " " << eigen_vector.minCoeff() << " " << eigen_vector.maxCoeff() << " " << standard_deviation(eigen_vector) << endl;

  //     Board2D board;
  //     board << domain;
  //     board << calculus;
  //     board << CustomStyle("KForm", new KFormStyle2D(eigen_vectors.minCoeff(),eigen_vectors.maxCoeff()));
  //     board << eigen_form;
  //     board.saveSVG(filename.c_str());
  // }

  f.close();

  return 0;
}



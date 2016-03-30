#include <sstream>
#include <fstream>
#include <string>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

// always include EigenSupport.h before any other Eigen headers
#include "DGtal/math/linalg/EigenSupport.h"
#include <Eigen/Eigenvalues>

#include "DGtal/dec/DiscreteExteriorCalculus.h"
#include "DGtal/dec/DiscreteExteriorCalculusSolver.h"

#include "DGtal/base/Common.h"
#include "DGtal/helpers/StdDefs.h"
#include "DGtal/images/ImageSelector.h"
#include "DGtal/io/readers/GenericReader.h"
#include "DGtal/io/writers/GenericWriter.h"
#include "DGtal/io/boards/Board2D.h"
#include "DGtal/math/linalg/EigenSupport.h"
#include "DGtal/dec/DiscreteExteriorCalculus.h"
#include "DGtal/dec/DiscreteExteriorCalculusSolver.h"

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
void PrimalForm0ToImage( const Calculus& calculus, const typename Calculus::PrimalForm0& u, Image& image, const bool writeDetails, ofstream& f )
{
  double min_u = u.myContainer[ 0 ];
  double max_u = u.myContainer[ 0 ];
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++)
    {
      min_u = min( min_u, u.myContainer[ index ] );
      max_u = max( max_u, u.myContainer[ index ] );
    }
  if ( writeDetails )
    {
      trace.info() << "min_u=" << min_u << " max_u=" << max_u << std::endl;
      f << "min_u=" << min_u << " max_u=" << max_u << std::endl;
    }
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++)
    {
      const typename Calculus::SCell& cell = u.getSCell( index );
      // *** MODIF : pas de re normalisation ****************************************************
      //int g = (int) round( ( u.myContainer[ index ] - min_u ) * 255.0 / ( max_u -min_u ) );
      int g = (int) round( u.myContainer[ index ] * 255.0 );
      // ****************************************************************************************
      g = std::min( 255, g );
      image.setValue( calculus.myKSpace.sCoords( cell ), g );
    }
}

template <typename Calculus, typename Image>
void PrimalForm1ToImage( const Calculus& calculus, const typename Calculus::PrimalForm1& v, Image& image, const bool writeDetails, ofstream& f )
{
  double min_v = v.myContainer[ 0 ];
  double max_v = v.myContainer[ 0 ];
  for ( typename Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    {
      min_v = min( min_v, v.myContainer[ index ] );
      max_v = max( max_v, v.myContainer[ index ] );
    }
  if ( writeDetails )
    {
      trace.info() << "min_v=" << min_v << " max_v=" << max_v << std::endl;
      f << "min_v=" << min_v << " max_v=" << max_v << std::endl << std::endl;
    }
  for ( typename Image::Iterator it = image.begin(), itE = image.end(); it != itE; ++it )
    *it = 255;
  for ( typename Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    {
      const typename Calculus::SCell& cell = v.getSCell( index );
      // *** MODIF : pas de re normalisation ****************************************************
      //int g = (int) round( ( v.myContainer[ index ] - min_v ) * 255.0 / ( max_v -min_v ) );
      double vi = fabs( v.myContainer[ index ] );
      int g = (int) round( vi * 255.0 );
      // ****************************************************************************************
      g = std::min( 255, g );
      image.setValue( calculus.myKSpace.sKCoords( cell ), g );
    }
}

int main( int argc, char* argv[] )
{
  using namespace Z2i;
  typedef ImageSelector<Domain, unsigned char>::Type Image;

  // parse command line ----------------------------------------------
  namespace po = boost::program_options;
  po::options_description general_opt("Allowed options are: ");
  general_opt.add_options()
    ("help,h", "display this message")
    ("input,i", po::value<string>(), "the input image filename." )
    ("output,o", po::value<string>()->default_value( "AT" ), "the output image basename." )
    ("lambda,l", po::value<double>(), "the parameter lambda." )
    ("lambda-1,l1", po::value<double>()->default_value( 10.0 ), "the initial parameter lambda (l1)." )
    ("lambda-2,l2", po::value<double>()->default_value( 0.001 ), "the final parameter lambda (l2)." )
    ("lambda-ratio,r", po::value<double>()->default_value( sqrt(2) ), "the division ratio for lambda from l1 to l2." )
    ("alpha,a", po::value<double>()->default_value( 1.0 ), "the parameter alpha." )
    ("epsilon,e", po::value<double>()->default_value( 1.0 ), "the parameter epsilon." )
    ("gridstep,g", po::value<double>()->default_value( 1.0 ), "the parameter h, i.e. the gridstep." )
    ("nbiter,n", po::value<int>()->default_value( 10 ), "the maximum number of iterations." )
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
	   << endl
	   << general_opt << "\n";
      return 1;
    }
  string f1  = vm[ "input" ].as<string>();
  string f2  = vm[ "output" ].as<string>();
  double l1  = vm[ "lambda-1" ].as<double>();
  double l2  = vm[ "lambda-2" ].as<double>();
  double lr  = vm[ "lambda-ratio" ].as<double>();
  if ( vm.count( "lambda" ) ) l1 = l2 = vm[ "lambda" ].as<double>();
  if ( l2 > l1 ) l2 = l1;
  if ( lr <= 1.0 ) lr = sqrt(2);
  double a  = vm[ "alpha" ].as<double>();
  double e  = vm[ "epsilon" ].as<double>();
  double h  = vm[ "gridstep" ].as<double>();
  int    n  = vm[ "nbiter" ].as<int>();

  trace.beginBlock("Reading image");
  Image image = GenericReader<Image>::import( f1 );
  trace.endBlock();

  trace.beginBlock("Creating calculus");
  typedef DiscreteExteriorCalculus<2, EigenLinearAlgebraBackend> Calculus;
  Domain domain = image.domain();
  Point p0 = domain.lowerBound(); p0 *= 2;
  Point p1 = domain.upperBound(); p1 *= 2;
  Domain kdomain( p0, p1 );
  Image dbl_image( kdomain );
  Calculus calculus;
  const KSpace& K = calculus.myKSpace;
  // Les pixels sont des 0-cellules du primal.
  for ( Domain::ConstIterator it = kdomain.begin(), itE = kdomain.end(); it != itE; ++it )
    calculus.insertSCell( K.sCell( *it ) ); // ajoute toutes les cellules de Khalimsky.
  trace.info() << calculus << endl;
  Calculus::PrimalForm0 g( calculus );
  for ( Calculus::Index index = 0; index < g.myContainer.rows(); index++)
    {
      const Calculus::SCell& cell = g.getSCell( index );
      g.myContainer( index ) = ((double) image( K.sCoords( cell ) )) / 255.0;
    }
  {
    Board2D board;
    board << calculus;
    board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) )
          << g;
    string str_calculus = f2 + "-calculus.eps";
    board.saveEPS( str_calculus.c_str() );
  }
  trace.endBlock();

  trace.beginBlock("building AT functionnals");
  trace.info() << "primal_D0" << endl;
  const Calculus::PrimalDerivative0 primal_D0 = calculus.derivative<0, PRIMAL>();
  trace.info() << "primal_D1" << endl;
  const Calculus::PrimalDerivative1 primal_D1 = calculus.derivative<1, PRIMAL>();
  trace.info() << "primal_h0" << endl;
  const Calculus::PrimalHodge0  primal_h0 = calculus.primalHodge<0>();
  trace.info() << "primal_h1" << endl;
  const Calculus::PrimalHodge1  primal_h1 = calculus.primalHodge<1>();
  trace.info() << "primal_h2" << endl;
  const Calculus::PrimalHodge2      primal_h2 = calculus.primalHodge<2>();

  trace.info() << "dual_D0" << endl;
  const Calculus::DualDerivative0     dual_D0 = calculus.derivative<0, DUAL>();
  trace.info() << "dual_D1" << endl;
  const Calculus::DualDerivative1 dual_D1 = calculus.derivative<1, DUAL>();
  trace.info() << "dual_h1" << endl;
  const Calculus::DualHodge1          dual_h1 = calculus.dualHodge<1>();
  trace.info() << "dual_h2" << endl;
  const Calculus::DualHodge2      dual_h2 = calculus.dualHodge<2>();

  trace.info() << "ag" << endl;
  const Calculus::PrimalForm0 ag = a * g;
  trace.info() << "u" << endl;
  Calculus::PrimalForm0 u = ag;
  // trace.info() << "A^t*diag(v)^2*A = " << Av2A << endl;
  trace.info() << "v" << endl;
  Calculus::PrimalForm1 v( calculus );
  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    v.myContainer( index ) = 1.0;
  trace.endBlock();
  // SparseLU is so much faster than SparseQR
  // SimplicialLLT is much faster than SparsLU
  // typedef EigenLinearAlgebraBackend::SolverSparseQR LinearAlgebraSolver;
  // typedef EigenLinearAlgebraBackend::SolverSparseLU LinearAlgebraSolver;
  typedef EigenLinearAlgebraBackend::SolverSimplicialLLT LinearAlgebraSolver;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 0, PRIMAL, 0, PRIMAL> Solver;
  Solver solver;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 1, PRIMAL, 1, PRIMAL> SolverV;
  SolverV solver_v;
	
  // MODIF : calcul de B^t.B ***********************************************************************************
  SolverV solver_B2;
  const Calculus::PrimalIdentity1 B2 = -1.0 * dual_h1 * dual_D0 * primal_h2 * primal_D1;
  solver_B2.compute( B2 ); 
  // ***********************************************************************************************************************

  // MODIF : ouverture du fichier pour les resultats ***********************************************************************************
  const string file_name = f2 + ".txt";
  ofstream file(file_name.c_str());
  // ***********************************************************************************************************************



  while ( l1 >= l2 )
    {
      trace.info() << "************ lambda = " << l1 << " **************" << endl;
      double l = l1;
      trace.info() << "BB" << endl;
      const Calculus::PrimalIdentity1          BB = -l * e * dual_h1 * dual_D0 * primal_h2 * primal_D1;
      // + (l/(4*e)) *  calculus.identity<1, PRIMAL>();
      trace.info() << "le*B^t*B" << BB << endl;
      double l_4e_cst = l/(4.0*e);
      Calculus::PrimalForm1 l_4e( calculus );
      for ( Calculus::Index index = 0; index < l_4e.myContainer.rows(); index++)
        l_4e.myContainer( index ) = l_4e_cst;
      trace.info() << "l_4e" << l_4e << endl;
      int i = 0;
      for ( ; i < n; ++i )
        {
          trace.info() << "------ Iteration " << i << "/" << n << " ------" << endl;
          trace.beginBlock("Solving for u");
          trace.info() << "Building matrix Av2A" << endl;
          Calculus::PrimalIdentity1 Mv2 = calculus.identity<1, PRIMAL>();
          for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
            Mv2.myContainer( index, index ) = v.myContainer[ index ] * v.myContainer[ index ];
          const Calculus::PrimalIdentity0 Av2A = -1.0 * dual_h2 * dual_D1 * primal_h1 * Mv2 * primal_D0
            + a * calculus.identity<0, PRIMAL>();
          trace.info() << "Prefactoring matrix Av2A" << endl;
          solver.compute( Av2A);
          trace.info() << "Solving Av2A u = ag" << endl;
          u = solver.solve( ag );
          trace.info() << ( solver.isValid() ? "OK" : "ERROR" ) << " " << solver.myLinearAlgebraSolver.info() << endl;
          trace.endBlock();

          trace.beginBlock("Solving for v");
          trace.info() << "Building matrix BB+Mw2" << endl;
          const Calculus::PrimalForm1 former_v = v;
          const Calculus::PrimalForm1 w = primal_D0 * u;
          Calculus::PrimalIdentity1 Mw2 = calculus.identity<1, PRIMAL>();
          for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
            Mw2.myContainer( index, index ) = w.myContainer[ index ] * w.myContainer[ index ];
          // Building l/(4e)*v/|v|
          Calculus::PrimalIdentity1 l_4e_sign = calculus.identity<1, PRIMAL>();
          for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
            { 
              double vi = v.myContainer( index );
              l_4e_sign.myContainer( index, index ) = ( vi > 0.0 ) 
                ? l_4e_cst : ( ( vi < 0.0 ) ? -l_4e_cst : 0.0 );
            }
	  //          Calculus::PrimalIdentity1          BB_sign = BB;
	  //          for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
	  //            {
	  //              bool si = v.myContainer( index_i ) >= 0.0;
	  //             for ( Calculus::Index index_j = 0; index_j < v.myContainer.rows(); index_j++)
	  //                {
	  //                  bool sj = v.myContainer( index_j ) >= 0.0;
	  //                  if ( si != sj ) BB_sign.myContainer(index_i, index_j ) *= -1.0; 
	  //                }
	  //            }      
	  typedef Calculus::PrimalIdentity1::Container Matrix1;

          Calculus::PrimalIdentity1          BB_sign = BB;
          Matrix1& mat = BB_sign.myContainer;
          for ( int k = 0; k < mat.outerSize(); ++k )
            for ( Matrix1::InnerIterator it(mat,k); it; ++it )
              {
                Calculus::Index index_i = it.row();
                Calculus::Index index_j = it.col();
                bool si = v.myContainer( index_i ) >= 0.0;
                bool sj = v.myContainer( index_j ) >= 0.0;
                if ( si != sj ) it.valueRef() = -it.value();
              }


     
          trace.info() << "Prefactoring matrix BB+Mw2" << endl;
          solver_v.compute( BB_sign + l_4e_sign + Mw2 );
          trace.info() << "Solving (BB_sign + l_4e_sign + Mw2 )v = l_4e" << endl;
          v = solver_v.solve( l_4e );
          // *** MODIF : troncature de v -> v in [0,1] *****************************************************************************
          // for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
          // {
          //     if ( v.myContainer( index ) < 0.0 )
          //       v.myContainer( index ) = 0.0;
          //     else if ( v.myContainer( index ) > 1.0 )
          //       v.myContainer( index ) = 1.0;
          // }
          // ***********************************************************************************************************************
          trace.info() << ( solver_v.isValid() ? "OK" : "ERROR" ) << " " << solver_v.myLinearAlgebraSolver.info() << endl;
          trace.endBlock();

          double n_infty = 0.0;
          for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
            n_infty = max( n_infty, abs( v.myContainer( index ) - former_v.myContainer( index ) ) );
          trace.info() << "Variation |v^k+1 - v^k|_oo = " << n_infty << endl;
          if ( n_infty < 1e-4 ) break;
        }

	
      // *** MODIF : affichage du perimetre ***********************************************************************************
      double per = 0.0;
      for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
	{
	  double vi = v.myContainer( index_i );
	  per += (1/(4*e)) * (1 - 2*fabs(vi) + vi*vi );
	  for ( Calculus::Index index_j = 0; index_j < v.myContainer.rows(); index_j++)
            // B2 = B^t.B
            per += e * vi * B2.myContainer( index_i,index_j ) * v.myContainer( index_j );
	}
      trace.info() << "Per = " << per << endl;
      
      file << "************ lambda = " << l1 << " **************" << endl;
      file << "Per = " << per << endl << endl;
      // ***********************************************************************************************************************


      // *** MODIF : projection de v sur une 0-forme ***************************************************************************
      Calculus::PrimalForm0 vfin( calculus );
      vfin = dual_h2 * dual_D1 * primal_h1 * v;
      // ***********************************************************************************************************************


      int int_l = (int) floor(l);
      int dec_l = (int) (floor((l-floor(l))*10000));
      {
        // Board2D board;
        // board << calculus;
        // board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) ) << u;
        // ostringstream oss;
        // oss << f2 << "-u-" << i << ".eps";
        // string str_u = oss.str(); //f2 + "-u-" + .eps";
        // board.saveEPS( str_u.c_str() );
        Image image2 = image;
        PrimalForm0ToImage( calculus, u, image2, true, file );
        ostringstream oss2;
        oss2 << f2 << "-l" << int_l << "_" << dec_l << "-u.pgm";
        string str_image_u = oss2.str();
        image2 >> str_image_u.c_str();
      }
      {
        PrimalForm1ToImage( calculus, v, dbl_image, true, file );
        ostringstream oss2;
        oss2 << f2 << "-l" << int_l << "_" << dec_l << "-v.pgm";
        string str_image_v = oss2.str();
        dbl_image >> str_image_v.c_str();
      }
      {
	Image image_v0 = image;
	PrimalForm0ToImage( calculus, vfin, image_v0, true, file );
	ostringstream oss3;
	oss3 << f2 << "-l" << int_l << "_" << dec_l << "-v0.pgm";
	string str_image_vfin = oss3.str();
	cout << "image_v0 = " << image_v0 << endl;
	image_v0 >> str_image_vfin.c_str();
      }
      //    {
      // 	Calculus::PrimalForm1 v_inf0 = v;
      // 	Image dbl_image_v_inf0( kdomain );
      // 	for ( Calculus::Index index = 0; index < v_inf0.myContainer.rows(); index++)
      // 		if ( v.myContainer( index ) > 0.0 ) 
      // 			v_inf0.myContainer( index ) = 1.0;
      // 		else
      // 			v_inf0.myContainer( index ) = fabs(v.myContainer( index ));
	
      //     PrimalForm1ToImage( calculus, v_inf0, dbl_image_v_inf0, false, file );
      //     ostringstream oss2;
      //     oss2 << f2 << "-l" << int_l << "_" << dec_l << "-v_inf0.pgm";
      //     string str_image_v_inf0 = oss2.str();
      //     dbl_image_v_inf0 >> str_image_v_inf0.c_str();
      //   }
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
  
  
  file.close();
  
  return 0;
}



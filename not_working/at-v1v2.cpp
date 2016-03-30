#include <sstream>
#include <fstream>
#include <string>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

// always include EigenSupport.h before any other Eigen headers
#include "DGtal/math/linalg/EigenSupport.h"
#include <Eigen/Eigenvalues>

#include "DGtal/dec/DiscreteExteriorCalculus.h" /*pour faire alternate_derivative*/
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
      // *** MODIF : pas de re normalisation ****************************************************
      //int g = (int) round( ( u.myContainer[ index ] - min_u ) * 255.0 / ( max_u -min_u ) );
      int g = (int) round( u.myContainer[ index ] * 255.0 );
      // ****************************************************************************************
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
      // *** MODIF : pas de re normalisation ****************************************************
      //int g = (int) round( ( v.myContainer[ index ] - min_v ) * 255.0 / ( max_v -min_v ) );
      int g = (int) round( v.myContainer[ index ] * 255.0 );
      // ****************************************************************************************
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
    ("lambda-1,1", po::value<double>()->default_value( 10.0 ), "the initial parameter lambda (l1)." )
    ("lambda-2,2", po::value<double>()->default_value( 0.001 ), "the final parameter lambda (l2)." )
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
  Image image = GenericReader<Image>::import( f1 ); /* pour les 0-formes*/
  trace.endBlock();

  trace.beginBlock("Creating calculus");
  typedef DiscreteExteriorCalculus<2, EigenLinearAlgebraBackend> Calculus;
  Domain domain = image.domain();
  Point p0 = domain.lowerBound(); p0 *= 2;
  Point p1 = domain.upperBound(); p1 *= 2;
  Domain kdomain( p0, p1 );
  Image dbl_image( kdomain ); /* pour les 1-formes */
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
//  {
//    Board2D board;
//    board << calculus;
//    board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) )
//          << g;
//    string str_calculus = f2 + "-calculus.eps";
//    board.saveEPS( str_calculus.c_str() );
//  }
  trace.endBlock();

  trace.beginBlock("building AT functionnals");
  trace.info() << "primal_D0" << endl;
  const Calculus::PrimalDerivative0 primal_D0     = calculus.derivative<0, PRIMAL>();
  const Calculus::PrimalDerivative0 primal_D0_alt = calculus.alternate_derivative<0, PRIMAL>();
  trace.info() << "primal_h0" << endl;
  const Calculus::PrimalHodge0  primal_h0         = calculus.primalHodge<0>();
  trace.info() << "primal_h1" << endl;
  const Calculus::PrimalHodge1  primal_h1         = calculus.primalHodge<1>();
  trace.info() << "dual_D1" << endl;
  const Calculus::DualDerivative1 dual_D1         = calculus.derivative<1, DUAL>();
  const Calculus::DualDerivative1 dual_D1_alt     = calculus.alternate_derivative<1, DUAL>();
  trace.info() << "dual_h2" << endl;
  const Calculus::DualHodge2      dual_h2         = calculus.dualHodge<2>();
  trace.info() << "primal_D1" << endl;
  const Calculus::PrimalDerivative1 primal_D1     = calculus.derivative<1, PRIMAL>();
  const Calculus::PrimalDerivative1 primal_D1_alt = calculus.alternate_derivative<1, PRIMAL>();
  trace.info() << "primal_h2" << endl;
  const Calculus::PrimalHodge2      primal_h2     = calculus.primalHodge<2>();
  trace.info() << "dual_D0" << endl;
  const Calculus::DualDerivative0     dual_D0     = calculus.derivative<0, DUAL>();
  const Calculus::DualDerivative0     dual_D0_alt = calculus.alternate_derivative<0, DUAL>();
  trace.info() << "dual_h1" << endl;
  const Calculus::DualHodge1          dual_h1     = calculus.dualHodge<1>();
  trace.info() << "ag" << endl;
  const Calculus::PrimalForm0 ag = a * g;
  trace.info() << "u" << endl;
  Calculus::PrimalForm0 u = ag;
  // trace.info() << "A^t*diag(v)^2*A = " << Av2A << endl;
  trace.info() << "v1 & v2" << endl;
  Calculus::PrimalForm1 v ( calculus );
  Calculus::PrimalForm1 v1( calculus );
  Calculus::PrimalForm1 v2( calculus );
  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    {
      v1.myContainer( index ) = 1.0;
      v2.myContainer( index ) = 1.0;
       v.myContainer( index )  = v1.myContainer( index ) + v2.myContainer( index ) - 1.0;
    }
  trace.endBlock();
  // SparseLU is so much faster than SparseQR
  // SimplicialLLT is much faster than SparsLU
  // typedef EigenLinearAlgebraBackend::SolverSparseQR LinearAlgebraSolver;
  // typedef EigenLinearAlgebraBackend::SolverSparseLU LinearAlgebraSolver;
  typedef EigenLinearAlgebraBackend::SolverSimplicialLLT LinearAlgebraSolver;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 0, PRIMAL, 0, PRIMAL> SolverU;
  SolverU solver_u;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 1, PRIMAL, 1, PRIMAL> SolverV;
  SolverV solver_v1, solver_v2;
	
	// MODIF : calcul de B^t.B ***********************************************************************************
	SolverV solver_B2;
	const Calculus::PrimalIdentity1 B2     = -1.0 * dual_h1 * dual_D0     * primal_h2 * primal_D1;
 	solver_B2.compute( B2 );
	SolverV solver_B2_alt;
	const Calculus::PrimalIdentity1 B2_alt =        dual_h1 * dual_D0_alt * primal_h2 * primal_D1_alt;
 	solver_B2.compute( B2_alt ); 
	SolverV solver_tBB_alt;
	const Calculus::PrimalIdentity1 tBB_alt = -1.0 * (dual_h1 * dual_D0     * primal_h2) * primal_D1_alt;
 	solver_tBB_alt.compute( tBB_alt );
	SolverV solver_tB_altB;
	const Calculus::PrimalIdentity1 tB_altB =        (dual_h1 * dual_D0_alt * primal_h2) * primal_D1;
 	solver_tB_altB.compute( tB_altB ); 
	// ***********************************************************************************************************************

	// MODIF : ouverture du fichier pour les resultats ***********************************************************************************
	const string file = f2 + ".txt";
	ofstream f(file.c_str());
	// ***********************************************************************************************************************



  while ( l1 >= l2 )
    {
      trace.info() << "************ lambda = " << l1 << " **************" << endl;
      double l = l1;
      trace.info() << "BB" << endl;
      const Calculus::PrimalIdentity1				BB     = -  l * e    * dual_h1 * dual_D0     * primal_h2 * primal_D1
        																					   + (l/(4*e)) * calculus.identity<1, PRIMAL>();
      const Calculus::PrimalIdentity1				BB_alt =    l * e    * dual_h1 * dual_D0_alt * primal_h2 * primal_D1_alt
        																					   + (l/(4*e)) * calculus.identity<1, PRIMAL>();
      trace.info() << "le*B^t*B - l/(4e)Id" << BB << endl;
      trace.info() << "l_4e" << endl;
      Calculus::PrimalForm1 l_4e( calculus );
      for ( Calculus::Index index = 0; index < l_4e.myContainer.rows(); index++)
        l_4e.myContainer( index ) = l/(4*e);
      
      int i = 0;
      for ( ; i < n; ++i )
        {
          trace.info() << "------ Iteration " << i << "/" << n << " ------" << endl;
          trace.beginBlock("Solving for u");
          trace.info() << "Building matrix Av2A" << endl;
		  	  Calculus::PrimalIdentity1 Mv12 = calculus.identity<1, PRIMAL>(), Mv22 = calculus.identity<1, PRIMAL>();
		  		for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
		  		{
            Mv12.myContainer( index, index ) = v1.myContainer[ index ] * v1.myContainer[ index ];
            Mv22.myContainer( index, index ) = v2.myContainer[ index ] * v2.myContainer[ index ];
      		}
          const Calculus::PrimalIdentity0 Av2A     =  - 1.0 * dual_h2 * dual_D1     * primal_h1 * Mv12 * primal_D0
          											  +       dual_h2 * dual_D1_alt * primal_h1 * Mv22 * primal_D0_alt
            								  										    + a * calculus.identity<0, PRIMAL>();
//          const Calculus::PrimalIdentity0 Av2A_alt =          dual_h2 * dual_D1_alt * primal_h1 * Mv2 * primal_D0_alt
//            								  										    + a * calculus.identity<0, PRIMAL>();
          trace.info() << "Prefactoring matrix Av2A" << endl;
          solver_u.compute( Av2A );
          trace.info() << "Solving Av2A u = ag" << endl;
          u = solver_u.solve( ag );
          trace.info() << ( solver_u.isValid() ? "OK" : "ERROR" ) << " " << solver_u.myLinearAlgebraSolver.info() << endl;
          trace.endBlock();

          trace.beginBlock("Solving for v");
          trace.info() << "Building matrix BB+Mw2" << endl;
          const Calculus::PrimalForm1 former_v  = v;
          const Calculus::PrimalForm1 former_v1 = v1;
          const Calculus::PrimalForm1 former_v2 = v2;
          const Calculus::PrimalForm1 Au     = primal_D0     * u;
          const Calculus::PrimalForm1 Au_alt = primal_D0_alt * u;
          Calculus::PrimalIdentity1 MAu2 = calculus.identity<1, PRIMAL>(), MAu2_alt = calculus.identity<1, PRIMAL>();
          for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
          {
            MAu2    .myContainer( index, index ) = Au    .myContainer[ index ] * Au    .myContainer[ index ];
            MAu2_alt.myContainer( index, index ) = Au_alt.myContainer[ index ] * Au_alt.myContainer[ index ];
          }
          trace.info() << "Prefactoring matrix BB+Mw2" << endl;
          solver_v1.compute( BB     + MAu2     );
          trace.info() << "Solving (BB+Mw2)v1 = leBBv2 + l_4e" << endl;
          // membre gauche = tB_alt * B_alt * v2
          v1 = solver_v1.solve( l * e * B2_alt * v2 + l_4e );
          // membre gauche = 0.5 * [ tB * B_alt + tB_alt * B ] * v2
					// v1 = solver_v1.solve( 0.5*l*e*( tBB_alt*v2 + tB_altB*v2 ) + l_4e );
          solver_v2.compute( BB_alt + MAu2_alt );
          trace.info() << "Solving (BB_alt+Mw2_alt)v2 = leBBv1 + l_4e" << endl;
          // membre gauche = tB * B * v1
          v2 = solver_v2.solve( l * e * B2 * v1 + l_4e  );
          // membre gauche = 0.5 * [ tB * B_alt + tB_alt * B ] * v1
          // v2 = solver_v2.solve( 0.5*l*e*( tBB_alt*v1 + tB_altB*v1 ) + l_4e  );
          trace.info() << ( solver_v1.isValid() ? "OK" : "ERROR" ) << " " << solver_v1.myLinearAlgebraSolver.info() << endl;
          trace.info() << ( solver_v2.isValid() ? "OK" : "ERROR" ) << " " << solver_v2.myLinearAlgebraSolver.info() << endl;
          trace.endBlock();


	  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
	      v.myContainer( index )  = v1.myContainer( index ) + v2.myContainer( index ) - 1.0;


          double n_infty = 0.0;
          for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
            n_infty = max( n_infty, abs( v.myContainer( index ) - former_v.myContainer( index ) ) );
          trace.info() << "Variation |v^k+1 - v^k|_oo = " << n_infty << endl;
          if ( n_infty < 1e-4 ) break;
        }

	
      // *** MODIF : affichage du perimetre ***********************************************************************************
//      double per = 0.0;
//      for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
//      {
//        per += (1/(4*e)) * (1 - 2*v.myContainer( index_i ) + v.myContainer( index_i )*v.myContainer( index_i ));
//        for ( Calculus::Index index_j = 0; index_j < v.myContainer.rows(); index_j++)
//            // B2 = B^t.B
//            per += e * v.myContainer( index_i ) * B2.myContainer( index_i,index_j ) * v.myContainer( index_j );
//      }
//      trace.info() << "Per = " << per << endl;

//			f << "************ lambda = " << l1 << " **************" << endl;
//			f << "Per = " << per << endl << endl;
      // ***********************************************************************************************************************

//      // *** MODIF : projection de v sur une 0-forme ***************************************************************************
//      Calculus::PrimalForm0 vfin( calculus );
//      vfin = dual_h2 * dual_D1 * primal_h1 * v;
//      // ***********************************************************************************************************************

//      int int_l = (int) floor(l);
//      int dec_l = (int) (floor((l-floor(l))*10000)); 
			std::ostringstream ss;
			ss << (double) floor(l*10000) / 10000;     // troncature<double>(nb,4);
			string s = ss.str();
			const int pos = s.find(".");
			string tronc_l = s.substr(0,pos) + "_" + s.substr(pos+1);
 
     
      {
        // Board2D board;
        // board << calculus;
        // board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) ) << u;
        // ostringstream oss;
        // oss << f2 << "-u-" << i << ".eps";
        // string str_u = oss.str(); //f2 + "-u-" + .eps";
        // board.saveEPS( str_u.c_str() );
        Image image2 = image;
        PrimalForm0ToImage( calculus, u, image2 );
        ostringstream oss2;
        oss2 << f2 << "-l" << tronc_l << "-u.pgm";
        string str_image_u = oss2.str();
        image2 >> str_image_u.c_str();
      }
      {
        PrimalForm1ToImage( calculus, v, dbl_image );
        ostringstream oss2;
        oss2 << f2 << "-l" << tronc_l << "-v.pgm";
        string str_image_v = oss2.str();
        dbl_image >> str_image_v.c_str();
      }
//      {
//				Image image_v0 = image;
//				PrimalForm0ToImage( calculus, vfin, image_v0 );
//				ostringstream oss3;
//				oss3 << f2 << "-l" << tronc_l << "-v0.pgm";
//				string str_image_vfin = oss3.str();
//				cout << "image_v0 = " << image_v0 << endl;
//				image_v0 >> str_image_vfin.c_str();
//      }
      {
				Image image_v1 = dbl_image;
				PrimalForm1ToImage( calculus, v1, image_v1 );
				ostringstream oss3;
				oss3 << f2 << "-l" << tronc_l << "-v1.pgm";
				string str_image_v1 = oss3.str();
				cout << "image_v1 = " << image_v1 << endl;
				image_v1 >> str_image_v1.c_str();
      }
      {
				Image image_v2 = dbl_image;
				PrimalForm1ToImage( calculus, v2, image_v2 );
				ostringstream oss3;
				oss3 << f2 << "-l" << tronc_l << "-v2.pgm";
				string str_image_v2 = oss3.str();
				cout << "image_v2 = " << image_v2 << endl;
				image_v2 >> str_image_v2.c_str();
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



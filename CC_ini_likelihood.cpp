/**
 * C++ version of Bayesian sparse reconstruction fitting likelihood
 * ----------------------------------------------------------------
 *
 * C++ code to evaluate the likelihoods used in the paper with PolyChord.
 *
 * Results should be identical to the ones using Python, but the C++ version is faster.
**/
# include "CC_ini_likelihood.hpp"
# include <iostream>  // For printing to terminal
# include <fstream>  // For loading config file
# include <assert.h>
# include <vector>  // For reading in data arrays in str_to_valarray
# include <iterator>  // For reading in data arrays in str_to_valarray
# include <sstream>  // For reading in data arrays in str_to_valarray
# include <Eigen/Dense>

// Define global config variables
// ------------------------------
std::string INI_STR;  // for finding config file path
Eigen::ArrayXd Y;
Eigen::ArrayXd X1;
Eigen::ArrayXd X2;
std::string FIT_FUNC;
double X_ERROR_SIGMA;
double Y_ERROR_SIGMA;
int NFUNC;
bool ADAPTIVE;
const double PI = 3.14159265358979323846;

// Define basis functions
// ----------------------

Eigen::ArrayXd gg_1d (Eigen::ArrayXd x1, double a, double mu, double sigma, double beta)
{
    /**
    1-dimensional generalised Gaussian basis function evaluated on x1 valarray.
 
    @param x1: coordinates on which to evaluate basis function
    @param a: amplitude
    @param mu: mean
    @param sigma: standard deviation
    @param beta: shape parameter
    @return y: basis function evaluated at each x
    **/
    return a * ( - ((x1 - mu).abs() / sigma).pow(beta)).exp();
}

Eigen::ArrayXd ta_1d (Eigen::ArrayXd x1, double a, double b, double w)
{
    /**
    1-dimensional tanh basis function evaluated on x1 valarray.
 
    @param x1: coordinates on which to evaluate basis function
    @param a: amplitude
    @param w: weight parameter
    @param b: bias parameter
    @return y: basis function evaluated at each x1
    **/
    return a * ((x1 * w) + b).tanh();
}

Eigen::ArrayXd gg_2d (Eigen::ArrayXd x1, Eigen::ArrayXd x2, double a, double mu1, double mu2,
                             double sigma1, double sigma2, double beta1, double beta2, double omega)
{
    /**
    2-dimensional generalised Gaussian basis function evaluated on x1, x2 valarrays.
 
    @param x1: x1 coordinates on which to evaluate basis function
    @param x2: x2 coordinates on which to evaluate basis function (must be same size as x2)
    @param a: amplitude
    @param mu1: x1 mean
    @param mu2: x2 mean
    @param sigma1: x1 standard deviation
    @param sigma2: x2 standard deviation
    @param beta1: x1shape parameter
    @param beta2: x2shape parameter
    @param omega: rotation angle
    @return y: basis function evaluated at each (x1, x2) coordinate
    **/
    Eigen::ArrayXd x1_new = (std::cos(omega) * (x1 - mu1)) - (std::sin(omega) * (x2 - mu2));
    Eigen::ArrayXd x2_new = (std::sin(omega) * (x1 - mu1)) + (std::cos(omega) * (x2 - mu2));
    return a * gg_1d(x1_new, 1.0, 0.0, sigma1, beta1) * gg_1d(x2_new, 1.0, 0.0, sigma2, beta2);
}

// Overload basis functions with summation functions
// -------------------------------------------------

Eigen::ArrayXd gg_1d (Eigen::ArrayXd x1, Eigen::ArrayXd theta, int nf, int nf_ad)
{
    /**
    Sum of 1-dimensional generalised Gaussian basis function evaluated on valarray.
    Adaptive method uses nf_ad basis functions; vanilla method has nf_ad=nf.
 
    @param x1: coordinates on which to evaluate basis function sum
    @param theta: parameters
    @param nf: total number of basis functions for which theta provides parameters
    @param nf_ad: basis functions to use (1<=nf_ad<=nf)
    @return y: basis function sum evaluated at each x
    **/
    Eigen::ArrayXd sum = gg_1d(x1, theta[0], theta[nf], theta[nf * 2], theta[nf * 3]);
    for (int i=1; i<nf_ad; i++)
    {
        sum += gg_1d(x1, theta[i], theta[i + nf], theta[i + (nf * 2)], theta[i + (nf * 3)]);
    }
    return sum;
}

Eigen::ArrayXd ta_1d (Eigen::ArrayXd x1, Eigen::ArrayXd theta, int nf, int nf_ad)
{
    /**
    Sum of 1-dimensional tanh basis function evaluated on x1 valarray.
    Adaptive method uses nf_ad basis functions; vanilla method has nf_ad=nf.
 
    @param x1: coordinates on which to evaluate basis function sum
    @param theta: parameters
    @param nf: total number of basis functions for which theta provides parameters
    @param nf_ad: basis functions to use (1<=nf_ad<=nf)
    @return y: basis function sum evaluated at each x
    **/
    Eigen::ArrayXd sum = ta_1d(x1, theta[0], theta[nf], theta[nf * 2]);
    for (int i=1; i<nf_ad; i++)
    {
        sum += ta_1d(x1, theta[i], theta[i + nf], theta[i + (nf * 2)]);
    }
    return sum;
}

Eigen::ArrayXd gg_2d (Eigen::ArrayXd x1, Eigen::ArrayXd x2, Eigen::ArrayXd theta, int nf, int nf_ad)
{
    /**
    Sum of 2-dimensional generalised Gaussian basis function evaluated on valarray.
    Adaptive method uses nf_ad basis functions; vanilla method has nf_ad=nf.
 
    @param x1: x1 coordinates on which to evaluate basis function sum
    @param x2: x2 coordinates on which to evaluate basis function sum
    @param theta: parameters
    @param nf: total number of basis functions for which theta provides parameters
    @param nf_ad: basis functions to use (1<=nf_ad<=nf)
    @return y: basis function sum evaluated at each x1,x2
    **/
    Eigen::ArrayXd sum = gg_2d(x1, x2, theta[0], theta[nf], theta[nf * 2], theta[nf * 3],
                                      theta[nf * 4], theta[nf * 5], theta[nf * 6], theta[nf * 7]);
    for (int i=1; i<nf_ad; i++)
    {
        sum += gg_2d(x1, x2, theta[i], theta[i + nf], theta[i + (nf * 2)], theta[i + (nf * 3)],
                     theta[i + (nf * 4)], theta[i + (nf * 5)], theta[i + (nf * 6)], theta[i + (nf * 7)]);
    }
    return sum;
}

// Neural network functions
// ------------------------

Eigen::ArrayXXd prop_layer(Eigen::ArrayXXd input, Eigen::ArrayXd bias, Eigen::MatrixXd w)
{
    /**
    Propogate signal from input nodes through a hidden layer.

    @param input: input signal. Shape=(Nin,M), where Nin is number of input nodes and M is number of training examples being evaluated.
    @param bias: bias params. Shape=(Nout,1).
    @param wL weights. Shape=(Nin, nout).
    @return out: output. Shape=(Nout,M).
    **/
    assert(w.cols() == input.rows());
    assert(w.rows() == bias.size());
    Eigen::ArrayXXd out = w * input.matrix();
    out.colwise() += bias;
    assert(out.cols() == input.cols());
    assert(out.rows() == w.rows());
    return out;
}

int nn_num_params (int nodes_per_layer, int nlayer)
{
    /**
    Get the number of parameters in a feed-forward neural network.

    @param nodes_per_layer
    @param nlayer
    @return nparam: number of parameters of network
    **/
    int nparam = 0;
    // input layer (with 2 inputs - x1 and x2) to first hidden layer
    nparam += 3 * nodes_per_layer;
    // connecting hidden layers if more than one
    nparam += nodes_per_layer * (nodes_per_layer + 1) * (nlayer - 1);
    // last hidden layer to output layer
    nparam += nodes_per_layer + 1;
    return nparam;
}

Eigen::ArrayXd nn_fit (Eigen::ArrayXd x1, Eigen::ArrayXd x2, Eigen::ArrayXd theta, int nlayer, int nf, int nf_ad)
{
    /**
    Fit a feed-formard neural network to input data.
    This has nlayer hidden layers, each with nf_ad nodes and tanh activation functions.
    The output y is a scalar and uses a sigmoid activation function.
 
    @param x1: x1 coordinates on which to evaluate
    @param x2: x2 coordinates on which to evaluate
    @param theta: parameters
    @param nlayer: number of hidden layers
    @param nf: total number of nodes per hidden layer for which theta provides parameters
    @param nf_ad: nodes per hidden layer to use (1<=nf_ad<=nf)
    @return y: basis function sum evaluated at each x
    **/

    // Check the number of params
    assert(theta.size() == nn_num_params(nf, nlayer) + 1);
    // Get data into the array of shape (2,m), where m is the number of data points
    Eigen::ArrayXXd input(x1.size(), 2);
    input << x1, x2;
    input = input.transpose().eval();
    // Get array without the final hyperparameter (which controls weight sizes)
    Eigen::ArrayXd params = theta.head(theta.size() - 1);
    params *= theta[theta.size() - 1];  // Multiply by parameter controlling weight sizes
    // Iterate over the layers
    int prev_nparam;
    Eigen::ArrayXd bias;
    // Note that the weights (but not the bias) of the output layer are stored at the start
    // of the params params (in order to use the sorted adaptive prior).
    int ind_start = nf;  // exclude weights of final layer
    // In general start_ind counter uses nf and size of w and bias use nf_ad
    for (int i=0; i<nlayer; i++)
    {
        // select bias params
        bias = params.segment(ind_start, nf_ad);
        ind_start += nf;
        // select weights params matrix
        if (i == 0) {
            prev_nparam = input.rows();
        } else {
            prev_nparam = nf;
        }
        Eigen::Map<Eigen::MatrixXd> w(params.segment(ind_start, prev_nparam * nf).data(), nf, prev_nparam);
        ind_start += (nf * prev_nparam);
        // pass only parameters to be used to prop_layer
        input = prop_layer(input, bias, w.topLeftCorner(nf_ad, input.rows()));
        input = input.tanh();
    }
    assert(input.rows() == nf_ad);
    assert(ind_start == params.size() - 1);
    // Map final hidden layer to output node
    // NB weights are stored at the start of params
    Eigen::ArrayXd out_bias = params.tail(1);
    Eigen::Map<Eigen::MatrixXd> out_w(params.head(nf_ad).data(), 1, nf_ad);
    Eigen::ArrayXd output = prop_layer(input, out_bias, out_w).row(0);
    // Sigmoid activation function on output
    output = ((-output).exp() + 1).inverse();
    assert(output.size() == x1.size());
    return output;
}

// Loglikelihood functions
// -----------------------

double logl_given_fit(Eigen::ArrayXd y_pred, Eigen::ArrayXd y_data, double y_error_sigma)
{
   double logl = -((y_pred - y_data).square()).sum() / (2. * std::pow(y_error_sigma, 2));
   logl -= std::log(PI * 2. * std::pow(y_error_sigma, 2)) * y_pred.size() / 2.;
   return logl;
}


Eigen::ArrayXd fit(Eigen::ArrayXd theta, Eigen::ArrayXd x1, Eigen::ArrayXd x2, std::string fit_func, int nfunc_max, int nfunc_ad, int nlayer_ad)
{
    Eigen::ArrayXd y_predicted;
    if (fit_func == "gg_1d") {
        y_predicted = gg_1d(x1, theta, nfunc_max, nfunc_ad);
    } else if (fit_func == "ta_1d") {
        y_predicted = ta_1d(x1, theta, nfunc_max, nfunc_ad);
    } else if (fit_func == "gg_2d") {
        y_predicted = gg_2d(x1, x2, theta, nfunc_max, nfunc_ad);
    } else if (fit_func == "nn_1l") {
        y_predicted = nn_fit(x1, x2, theta, 1, nfunc_max, nfunc_ad);
    } else if (fit_func == "nn_2l") {
        y_predicted = nn_fit(x1, x2, theta, 2, nfunc_max, nfunc_ad);
    } else if (fit_func == "nn_adl") {
        int nparam = nn_num_params(nfunc_max, nlayer_ad);
        Eigen::ArrayXd theta_to_use = theta.head(nparam + 1);
        // Add hyperparameter for scaling weights (stored in final element of theta)
        theta_to_use[nparam] = theta.tail(1)[0];
        y_predicted = nn_fit(x1, x2, theta_to_use, nlayer_ad, nfunc_max, nfunc_ad);
    } else {
        std::cout << "Unexpected fit_func value: fit_func=" << fit_func << '\n';
    }
    assert(y_predicted.size() > 0);
    return y_predicted;
}


Eigen::ArrayXd simpson_weights(std::size_t npoints, double dx)
{
   /**
   Get array of point weights for 1d Simpson integration of odd number of equally spaced samples.

   @param npoints: number of samples - must be odd
   @param dx: separation of samples
   @returns weights: point weights
   **/
   Eigen::ArrayXd weights;
   weights = weights.Ones(npoints) * 2;
   weights[npoints - 1] = 1;
   weights[0] = 1;
   Eigen::Map<Eigen::ArrayXd, 0,Eigen::InnerStride<2> > odd_weights(weights.tail(npoints - 1).data(), (npoints - 1) / 2);
   odd_weights += 2;
   weights *= dx / 3.0;
   return weights;
}


double loglikelihood (double theta_in[], int nDims, double phi[], int nDerived)
{
    /**
    Fitting likelihood for Bayesian sparse reconstruction.

    @param theta: parameter values
    @param nDims: number of dimensions
    @param: phi: derived parameters
    @param: nDerived: number of derived parameters
    @return logL: loglikelihood
    **/
    Eigen::ArrayXd theta;
    int nfunc_ad;
    int nlayer_ad;
    // Read theta_in to Eigen array
    Eigen::Map<Eigen::ArrayXd> theta_in_array(theta_in, nDims);
    if (ADAPTIVE)
    {
        if (FIT_FUNC == "nn_adl") {
            nlayer_ad = std::round(theta_in[0]);
            nfunc_ad = std::round(theta_in[1]);
            theta = theta_in_array.tail(nDims - 2);
            assert(nlayer_ad >= 1);
            assert(nlayer_ad <= 2);
        } else {
            nlayer_ad = 1;
            nfunc_ad = std::round(theta_in[0]);
            theta = theta_in_array.tail(nDims - 1);
        }
        assert(nfunc_ad >= 1);
        assert(nfunc_ad <= NFUNC);
    } else {
        nlayer_ad = 1;
        nfunc_ad = NFUNC;
        theta = theta_in_array;
    }
    if (X_ERROR_SIGMA < 0.0000000001) // Errors only on y values
    {
        Eigen::ArrayXd y_predicted = fit(theta, X1, X2, FIT_FUNC, NFUNC, nfunc_ad, nlayer_ad);
        return logl_given_fit(y_predicted, Y, Y_ERROR_SIGMA);
    } else {  // Errors only both x and y values
        // First calculate contributions of constants
        double logl = -std::log(2 * PI * Y_ERROR_SIGMA * X_ERROR_SIGMA) * Y.size();
        // Now do integrals
        std::size_t npoints = 501;  // Must be odd
        double dx = 1.0 / (npoints - 1);
        Eigen::ArrayXd x_int = Eigen::ArrayXd::LinSpaced(npoints, 0.0, 1.0);
        Eigen::ArrayXd simps_w = simpson_weights(npoints, dx);
        Eigen::ArrayXd y_int = fit(theta, x_int, X2, FIT_FUNC, NFUNC, nfunc_ad, nlayer_ad);
        Eigen::ArrayXd integrand;
        for (int i=0; i<Y.size(); i++)
        {
            integrand = ((X1[i] - x_int) / X_ERROR_SIGMA).square();
            integrand += ((Y[i] - y_int) / Y_ERROR_SIGMA).square();
            integrand = (- 0.5 * integrand).exp();
            // Vectorised integration
            double contrib = (integrand * simps_w).sum();
            if (contrib == 0)
            {
                return -std::pow(10.0, 300);
            }
            logl += std::log(contrib);
        }
        return logl;
    }
        
}


// Prior function
//
// Either write your prior code directly into this function, or call an
// external library from it. This should transform a coordinate in the unit hypercube
// stored in cube (of size nDims) to a coordinate in the physical system stored in theta
//
// This function is called from likelihoods/fortran_cpp_wrapper.f90
// If you would like to adjust the signature of this call, then you should adjust it there,
// as well as in likelihoods/my_cpp_likelihood.hpp
// 
// void prior (double cube[], double theta[], int nDims)
// {
//     //============================================================
//     // insert prior code here
//     //
//     //
//     //============================================================
//     for(int i=0;i<nDims;i++)
//         theta[i] = cube[i];
// 
// }

// Dumper function
//
// This function gives you runtime access to variables, every time the live
// points are compressed by a factor settings.compression_factor.
//
// To use the arrays, subscript by following this example:
//
//    for (auto i_dead=0;i_dead<ndead;i_dead++)
//    {
//        for (auto j_par=0;j_par<npars;j_par++)
//            std::cout << dead[npars*i_dead+j_par] << " ";
//        std::cout << std::endl;
//    }
//
// in the live and dead arrays, the rows contain the physical and derived
// parameters for each point, followed by the birth contour, then the
// loglikelihood contour
//
// logweights are posterior weights
// 
// void dumper(int ndead,int nlive,int npars,double* live,double* dead,double* logweights,double logZ, double logZerr)
// {
// }


// Ini path reading function
void set_ini(std::string ini_str_in)
{
    /**
    Set value for constant holding ini file path. This is used to find the config file's file path.

    @param ini_str_in: ini file path
    **/
    INI_STR = ini_str_in;
}


Eigen::ArrayXd str_to_array (std::string string_in)
{
    /**
    Converts string of doubles separated by commas to a valarray. Used for reading data from config file.

    @param string_in: string
    @returns output: valarray
    **/
    std::stringstream ss(string_in);
    std::vector<double> vec;
    while(ss.good())
    {
        std::string substr;
        getline(ss, substr, ',');
        vec.push_back(stod(substr));
    };
    return Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(vec.data(), vec.size());
}


// Setup of the loglikelihood
// 
// This is called before nested sampling, but after the priors and settings
// have been set up.
// 
// This is the time at which you should load any files that the likelihoods
// need, and do any initial calculations.
// 
// This module can be used to save variables in between calls
// (at the top of the file).
// 
// All MPI threads will call this function simultaneously, but you may need
// to use mpi utilities to synchronise them. This should be done through the
// integer mpi_communicator (which is normally MPI_COMM_WORLD).
//
// This function is called from likelihoods/fortran_cpp_wrapper.f90
// If you would like to adjust the signature of this call, then you should adjust it there,
// as well as in likelihoods/my_cpp_likelihood.hpp
void setup_loglikelihood()
{
    /**
    Load data and settings from config file.
    **/
    // Load config file. Filename is same as ini file but ending in .cfg instead of .ini.
    std::string cfg_file_name = INI_STR.substr(0, INI_STR.size()-3) + "cfg";
    std::ifstream cFile(cfg_file_name);
    if (cFile.is_open())
    {
        std::string line;
        while(getline(cFile, line)){
            line.erase(remove_if(line.begin(), line.end(), ::isspace),
                                 line.end());
            if(line[0] == '#' || line.empty())
                continue;
            auto delimiterPos = line.find("=");
            std::string name = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);
            if (name == "nfunc") {
                NFUNC = stoi(value);
            } else if (name == "fit_func") {
                FIT_FUNC = value;
            } else if (name == "x_error_sigma") {
                X_ERROR_SIGMA = stod(value);
            } else if (name == "y_error_sigma") {
                Y_ERROR_SIGMA = stod(value);
            } else if (name == "adaptive") {
                ADAPTIVE = (value == "True");
            } else if (name == "y") {
                Y = str_to_array(value);
            } else if (name == "x1") {
                X1 = str_to_array(value);
            } else if (name == "x2") {
                X2 = str_to_array(value);
            } else {
                std::cout << "unexpected config argument: " << name << "=" << value << '\n';
            }
        }
        
    }
    else {
        std::cerr << "Couldn't open config file for reading: " << cfg_file_name << "\n";
    }
    // Print some info
    // ---------------
    // std::cout << "fit_func=" << FIT_FUNC;
    // std::cout << " nfunc=" << NFUNC;
    // std::cout << " adaptive=" << ADAPTIVE;
    // std::cout << " y_error_sigma=" << Y_ERROR_SIGMA;
    // std::cout << " x_error_sigma=" << X_ERROR_SIGMA << "\n";
    // std::cout << "sizes: y=" << Y.size() << " x1=" << X1.size() << " x2=" << X2.size() << "\n";
    // std::cout.precision(15);
    // std::cout << "y.sum()  = " << Y.sum() << "\n";
    // std::cout << "x1.sum() = " << X1.sum() << "\n";
    // std::cout << "x2.sum() = " << X2.sum() << "\n";
    // Check the loaded inputs
    assert(X1.size() == Y.size());
    if (FIT_FUNC == "nn_adl")
    {
        assert(ADAPTIVE);
    }
    if (X2.size())  // if x2 is not empty, it should be the same size as y
    {
        assert(X2.size() == Y.size());
        // use fact x1 and x2 should both cover a grid with the same density in [0,1] to check data loading
        assert((std::abs(X2.sum() - X1.sum()) / std::abs(X2.sum() + X1.sum()))
                < std::pow(10, -12));
        assert(X_ERROR_SIGMA == 0.0);  // x errors not set up for x2 input
    }
}

#include <Rcpp.h>
#include <numeric>   	// for std::partial_sum
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector cumsum_4(NumericVector x){
  // initialize an accumulator variable
  double acc = 0;
  
  // initialize the result vector
  NumericVector res(x.size());
  
  for(int i = 0; i < x.size(); i++){
    acc += x[i];
    res[i] = acc;
  }
  return res;
}

// [[Rcpp::export]]
NumericVector cumsum_5(NumericVector x){
  // initialize the result vector
  NumericVector res(x.size());
  std::partial_sum(x.begin(), x.end(), res.begin());
  return res;
}

// [[Rcpp::export]]
NumericVector cumsum_6(NumericVector x){
  return cumsum(x);    // compute the result vector and return it
}

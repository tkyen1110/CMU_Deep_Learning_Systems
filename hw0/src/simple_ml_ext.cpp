#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t epochs = m / batch;
    float* Z_numerator = new float[batch * k];
    float* Z_denominator = new float[batch];
    float* Z = Z_numerator;  // Reuse Z_numerator for Z after normalization
    float* gradient = new float[n * k];

    for (size_t i = 0; i < epochs; i++) {
        const float *X_batch = X+(i*batch*n);
        const unsigned char *y_batch = y+(i*batch);

        // Compute logits and softmax probabilities
        for (size_t j = 0; j < batch; j++) {
            Z_denominator[j] = 0.0f;
            for (size_t c = 0; c < k; c++) {        
                Z_numerator[j * k + c] = 0.0f;
                for (size_t d = 0; d < n; d++) {
                    Z_numerator[j * k + c] += X_batch[j * n + d] * theta[d * k + c];
                }
                Z_numerator[j * k + c] = std::exp(Z_numerator[j * k + c]);
                Z_denominator[j] += Z_numerator[j * k + c];
            }
            
            for (size_t c = 0; c < k; c++) {
                Z_numerator[j * k + c] = Z_numerator[j * k + c] / Z_denominator[j];
                if (y_batch[j] == c) {
                    Z_numerator[j * k + c] -= 1.0f;  // Subtract 1 for the correct class
                }
            }
        }

        for (size_t d = 0; d < n; d++) {
            for (size_t c = 0; c < k; c++) {
                gradient[d * k + c] = 0.0f;
                for (size_t j = 0; j < batch; j++) {
                     gradient[d * k + c] += X_batch[j * n + d] * Z[j * k + c];
                }
                gradient[d * k + c] /= batch;
                theta[d * k + c] -= lr * gradient[d * k + c];
            }
        }
    }

    delete[] Z_numerator;
    delete[] Z_denominator;
    delete[] gradient;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}

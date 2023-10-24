use ndarray;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

// NOTE
// * numpy defaults to np.float64, if you use other type than f64 in Rust
//   you will have to change type in Python before calling the Rust function.

// The name of the module must be the same as the rust package name
#[pymodule]
fn rust_numpy_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // This is a pure function (no mutations of incoming data).
    // You can see this as the python array in the function arguments is readonly.
    // The object we return will need ot have the same lifetime as the Python.
    // Python will handle the objects deallocation.
    // We are having the Python as input with a lifetime parameter.
    // Basically, none of the data that comes from Python can survive
    // longer than Python itself. Therefore, if Python is dropped, so must our Rust Python-dependent variables.
    #[pyfn(m)]
    fn max_min<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<f64>) -> &'py PyArray1<f64> {
        // Here we have a numpy array of dynamic size. But we could restrict the
        // function to only take arrays of certain size
        // e.g. We could say PyReadonlyArray3 and only take 3 dim arrays.
        // These functions will also do type checking so a
        // numpy array of type np.float32 will not be accepted and will
        // yield an Exception in Python as expected
        let array = x.as_array();
        let result_array = rust_fn::max_min(&array);
        result_array.into_pyarray(py)
    }
    #[pyfn(m)]
    fn double_and_random_perturbation(
        _py: Python<'_>,
        x: &PyArrayDyn<f64>,
        perturbation_scaling: f64,
    ) {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let mut array = unsafe { x.as_array_mut() }; // Convert to ndarray type

        // Mutate the data
        // No need to return any value as the input data is mutated
        rust_fn::double_and_random_perturbation(&mut array, perturbation_scaling);
    }

    #[pyfn(m)]
    fn eye<'py>(py: Python<'py>, size: usize) -> &PyArray2<f64> {
        // Simple demonstration of creating an ndarray inside Rust and return
        let array = ndarray::Array::eye(size);
        array.into_pyarray(py)
    }

    Ok(())
}

// The rust side functions
// Put it in mod to separate it from the python bindings
// These are just some random operations
// you probably want to do something more meaningful.
mod rust_fn {
    use ndarray::{arr1, Array1};
    use numpy::ndarray::{ArrayViewD, ArrayViewMutD};
    use ordered_float::OrderedFloat;
    use rand::Rng;

    // If we wanted to do something like this in python
    // we probably would want to generate matrices and add them
    // together. This can be problematic in terms of memory if working with large
    // matrices. And looping is usually painfully slow.
    // Rayon could be used here to run the mutation in parallel
    // this may be good for huge matrices
    pub fn double_and_random_perturbation(x: &mut ArrayViewMutD<'_, f64>, scaling: f64) {
        let mut rng = rand::thread_rng();
        x.iter_mut()
            .for_each(|x| *x = *x * 2. + (rng.gen::<f64>() - 0.5) * scaling);
    }

    pub fn max_min(x: &ArrayViewD<'_, f64>) -> Array1<f64> {
        if x.len() == 0 {
            return arr1(&[]); // If the array has no elements, return empty array
        }
        let max_val = x
            .iter()
            .map(|a| OrderedFloat(*a))
            .max()
            .expect("Error calculating max value.")
            .0;
        let min_val = x
            .iter()
            .map(|a| OrderedFloat(*a))
            .min()
            .expect("Error calculating min value.")
            .0;
        let result_array = arr1(&[max_val, min_val]);
        result_array
    }
}

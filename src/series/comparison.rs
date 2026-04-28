//! Comparison operations on `Series`.
//!
//! Generates element-wise boolean `Series` by comparing against a scalar value.
//! Results can be used directly as filter masks with [`DataFrame::filter_by_mask`].
//!
//! The comparison value must be an Arrow scalar type
//! (e.g. `Scalar::<Int32Type>::new(42)`).

use crate::error::CrossbowError;
use crate::series::Series;

impl Series {
    impl_comparison_op!(
        gt,
        arrow::compute::kernels::cmp::gt,
        "Returns `true` where values are **greater than** `value`.\n\n\
         Produces a boolean `Series` suitable as a filter mask."
    );
    impl_comparison_op!(
        lt,
        arrow::compute::kernels::cmp::lt,
        "Returns `true` where values are **less than** `value`."
    );
    impl_comparison_op!(
        eq,
        arrow::compute::kernels::cmp::eq,
        "Returns `true` where values are **equal to** `value`."
    );
    impl_comparison_op!(
        neq,
        arrow::compute::kernels::cmp::neq,
        "Returns `true` where values are **not equal to** `value`."
    );
    impl_comparison_op!(
        gt_eq,
        arrow::compute::kernels::cmp::gt_eq,
        "Returns `true` where values are **greater than or equal to** `value`."
    );
    impl_comparison_op!(
        lt_eq,
        arrow::compute::kernels::cmp::lt_eq,
        "Returns `true` where values are **less than or equal to** `value`."
    );
}

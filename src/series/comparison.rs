use crate::error::CrossbowError;
use crate::series::Series;

impl Series {
    impl_comparison_op!(
        gt,
        arrow::compute::kernels::cmp::gt,
        "Compares the Series with a scalar value (greater than)."
    );
    impl_comparison_op!(
        lt,
        arrow::compute::kernels::cmp::lt,
        "Compares the Series with a scalar value (less than)."
    );
    impl_comparison_op!(
        eq,
        arrow::compute::kernels::cmp::eq,
        "Compares the Series with a scalar value (equal to)."
    );
    impl_comparison_op!(
        neq,
        arrow::compute::kernels::cmp::neq,
        "Compares the Series with a scalar value (not equal to)."
    );
    impl_comparison_op!(
        gt_eq,
        arrow::compute::kernels::cmp::gt_eq,
        "Compares the Series with a scalar value (greater than or equal to)."
    );
    impl_comparison_op!(
        lt_eq,
        arrow::compute::kernels::cmp::lt_eq,
        "Compares the Series with a scalar value (less than or equal to)."
    );
}

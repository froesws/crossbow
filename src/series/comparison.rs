//! Comparison operations on `Series`.
//!
//! Generates element-wise boolean `Series` by comparing against a scalar value.
//! Results can be used directly as filter masks with [`crate::DataFrame::filter_by_mask`].

use crate::error::CrossbowError;
use crate::series::Series;

impl Series {
    impl_comparison_op!(
        greater_than,
        arrow::compute::kernels::cmp::gt,
        "Returns `true` where values are **greater than** `value`.\n\n\
         Produces a boolean `Series` suitable as a filter mask.\n\n\
         # Examples\n\n\
         ```\n\
         use crossbow::{Series, Numeric};\n\
         \n\
         let s = Series::from(\"x\", vec![1i32, 5, 10]);\n\
         let mask = s.greater_than(Numeric::int32(4)).unwrap();\n\
         assert_eq!(mask.len(), 3);\n\
         ```"
    );
    impl_comparison_op!(
        less_than,
        arrow::compute::kernels::cmp::lt,
        "Returns `true` where values are **less than** `value`.\n\n\
         # Examples\n\n\
         ```\n\
         use crossbow::{Series, Numeric};\n\
         \n\
         let s = Series::from(\"x\", vec![1i32, 5, 10]);\n\
         let mask = s.less_than(Numeric::int32(6)).unwrap();\n\
         assert_eq!(mask.len(), 3);\n\
         ```"
    );
    impl_comparison_op!(
        equal,
        arrow::compute::kernels::cmp::eq,
        "Returns `true` where values are **equal to** `value`.\n\n\
         # Examples\n\n\
         ```\n\
         use crossbow::{Series, Numeric};\n\
         \n\
         let s = Series::from(\"x\", vec![5i32, 5, 3]);\n\
         let mask = s.equal(Numeric::int32(5)).unwrap();\n\
         assert_eq!(mask.len(), 3);\n\
         ```"
    );
    impl_comparison_op!(
        not_equal,
        arrow::compute::kernels::cmp::neq,
        "Returns `true` where values are **not equal to** `value`.\n\n\
         # Examples\n\n\
         ```\n\
         use crossbow::{Series, Numeric};\n\
         \n\
         let s = Series::from(\"x\", vec![1i32, 2, 3]);\n\
         let mask = s.not_equal(Numeric::int32(2)).unwrap();\n\
         assert_eq!(mask.len(), 3);\n\
         ```"
    );
    impl_comparison_op!(
        greater_than_or_equal,
        arrow::compute::kernels::cmp::gt_eq,
        "Returns `true` where values are **greater than or equal to** `value`.\n\n\
         # Examples\n\n\
         ```\n\
         use crossbow::{Series, Numeric};\n\
         \n\
         let s = Series::from(\"x\", vec![5i32, 10, 15]);\n\
         let mask = s.greater_than_or_equal(Numeric::int32(10)).unwrap();\n\
         assert_eq!(mask.len(), 3);\n\
         ```"
    );
    impl_comparison_op!(
        less_than_or_equal,
        arrow::compute::kernels::cmp::lt_eq,
        "Returns `true` where values are **less than or equal to** `value`.\n\n\
         # Examples\n\n\
         ```\n\
         use crossbow::{Series, Numeric};\n\
         \n\
         let s = Series::from(\"x\", vec![5i32, 10, 15]);\n\
         let mask = s.less_than_or_equal(Numeric::int32(10)).unwrap();\n\
         assert_eq!(mask.len(), 3);\n\
         ```"
    );
}

//! Comparison operations on `Series`.
//!
//! Generates element-wise boolean `Series` by comparing against a scalar value.
//! Results can be used directly as filter masks with [`DataFrame::filter_by_mask`].

use crate::error::CrossbowError;
use crate::series::Series;

impl Series {
    impl_comparison_op!(
        gt,
        arrow::compute::kernels::cmp::gt,
        "Returns `true` where values are **greater than** `value`.\n\n\
         Produces a boolean `Series` suitable as a filter mask.\n\n\
         # Examples\n\n\
         ```ignore\n\
         use crossbow::Series;\n\
         \n\
         let s = Series::from(\"x\", vec![1i32, 5, 10]);\n\
         let mask = s.gt(4i32).unwrap();\n\
         assert_eq!(mask.len(), 3);\n\
         ```"
    );
    impl_comparison_op!(
        lt,
        arrow::compute::kernels::cmp::lt,
        "Returns `true` where values are **less than** `value`.\n\n\
         # Examples\n\n\
         ```ignore\n\
         use crossbow::Series;\n\
         \n\
         let s = Series::from(\"x\", vec![1i32, 5, 10]);\n\
         let mask = s.lt(6i32).unwrap();\n\
         assert_eq!(mask.len(), 3);\n\
         ```"
    );
    impl_comparison_op!(
        eq,
        arrow::compute::kernels::cmp::eq,
        "Returns `true` where values are **equal to** `value`.\n\n\
         # Examples\n\n\
         ```ignore\n\
         use crossbow::Series;\n\
         \n\
         let s = Series::from(\"x\", vec![5i32, 5, 3]);\n\
         let mask = s.eq(5i32).unwrap();\n\
         assert_eq!(mask.len(), 3);\n\
         ```"
    );
    impl_comparison_op!(
        neq,
        arrow::compute::kernels::cmp::neq,
        "Returns `true` where values are **not equal to** `value`.\n\n\
         # Examples\n\n\
         ```ignore\n\
         use crossbow::Series;\n\
         \n\
         let s = Series::from(\"x\", vec![1i32, 2, 3]);\n\
         let mask = s.neq(2i32).unwrap();\n\
         assert_eq!(mask.len(), 3);\n\
         ```"
    );
    impl_comparison_op!(
        gt_eq,
        arrow::compute::kernels::cmp::gt_eq,
        "Returns `true` where values are **greater than or equal to** `value`.\n\n\
         # Examples\n\n\
         ```ignore\n\
         use crossbow::Series;\n\
         \n\
         let s = Series::from(\"x\", vec![5i32, 10, 15]);\n\
         let mask = s.gt_eq(10i32).unwrap();\n\
         assert_eq!(mask.len(), 3);\n\
         ```"
    );
    impl_comparison_op!(
        lt_eq,
        arrow::compute::kernels::cmp::lt_eq,
        "Returns `true` where values are **less than or equal to** `value`.\n\n\
         # Examples\n\n\
         ```ignore\n\
         use crossbow::Series;\n\
         \n\
         let s = Series::from(\"x\", vec![5i32, 10, 15]);\n\
         let mask = s.lt_eq(10i32).unwrap();\n\
         assert_eq!(mask.len(), 3);\n\
         ```"
    );
}

//! Element-wise arithmetic on `Series`.
//!
//! All operations require both operands to have the same length and data type.
//! Null values propagate: `null + x = null`, `null / x = null`, etc.
//! Integer overflow returns [`CrossbowError::ArithmeticOverflow`].
//! Division or modulo by zero returns [`CrossbowError::DivisionByZero`].

use std::sync::Arc;
use arrow::datatypes::DataType;
use arrow::array::Array;
use crate::error::CrossbowError;
use crate::series::Series;
use crate::expected_array;

impl Series {
    /// Element-wise addition: `self[i] + other[i]`.
    ///
    /// Both `Series` must have the same length and compatible types.
    /// Supports `Int32` and `Float64`.
    pub fn add(&self, other: &Series) -> Result<Series, CrossbowError> {
        binary_op(self, other, "+",
            |a, b| a.checked_add(b),
            std::ops::Add::add,
            false,
        )
    }

    /// Element-wise subtraction: `self[i] - other[i]`.
    pub fn subtract(&self, other: &Series) -> Result<Series, CrossbowError> {
        binary_op(self, other, "-",
            |a, b| a.checked_sub(b),
            std::ops::Sub::sub,
            false,
        )
    }

    /// Element-wise multiplication: `self[i] * other[i]`.
    pub fn multiply(&self, other: &Series) -> Result<Series, CrossbowError> {
        binary_op(self, other, "*",
            |a, b| a.checked_mul(b),
            std::ops::Mul::mul,
            false,
        )
    }

    /// Element-wise division: `self[i] / other[i]`.
    ///
    /// Returns [`CrossbowError::DivisionByZero`] if any element of `other` is zero.
    pub fn divide(&self, other: &Series) -> Result<Series, CrossbowError> {
        binary_op(self, other, "/",
            |a, b| Some(a.checked_div(b).unwrap_or(a / b)),
            std::ops::Div::div,
            true,
        )
    }

    /// Element-wise modulo: `self[i] % other[i]`. Only supports `Int32`.
    pub fn modulo(&self, other: &Series) -> Result<Series, CrossbowError> {
        binary_op_i32(self, other, "%", |a, b| Some(a % b), true)
    }
}

fn binary_op<FI, FF>(
    s1: &Series,
    s2: &Series,
    op_name: &str,
    op_i32: FI,
    op_f64: FF,
    check_division_by_zero: bool,
) -> Result<Series, CrossbowError>
where
    FI: Fn(i32, i32) -> Option<i32>,
    FF: Fn(f64, f64) -> f64,
{
    if s1.len() != s2.len() {
        return Err(CrossbowError::MismatchedColumnLengths);
    }

    match (s1.dtype(), s2.dtype()) {
        (DataType::Int32, DataType::Int32) => {
            let arr1 = s1.as_primitive::<arrow::array::Int32Array>()
                .ok_or_else(|| expected_array("Int32Array"))?;
            let arr2 = s2.as_primitive::<arrow::array::Int32Array>()
                .ok_or_else(|| expected_array("Int32Array"))?;

            if check_division_by_zero {
                for i in 0..arr2.len() {
                    if arr2.is_valid(i) && arr2.value(i) == 0 {
                        return Err(CrossbowError::DivisionByZero);
                    }
                }
            }

            let mut builder = arrow::array::Int32Builder::with_capacity(arr1.len());
            for i in 0..arr1.len() {
                if arr1.is_valid(i) && arr2.is_valid(i) {
                    match op_i32(arr1.value(i), arr2.value(i)) {
                        Some(v) => builder.append_value(v),
                        None => return Err(CrossbowError::ArithmeticOverflow(
                            format!("{} {} {} overflows Int32 at index {}",
                                arr1.value(i), op_name, arr2.value(i), i)
                        )),
                    }
                } else {
                    builder.append_null();
                }
            }
            Ok(Series::new(
                format!("{} {} {}", s1.name(), op_name, s2.name()),
                Arc::new(builder.finish()),
            ))
        }
        (DataType::Float64, DataType::Float64) => {
            let arr1 = s1.as_primitive::<arrow::array::Float64Array>()
                .ok_or_else(|| expected_array("Float64Array"))?;
            let arr2 = s2.as_primitive::<arrow::array::Float64Array>()
                .ok_or_else(|| expected_array("Float64Array"))?;

            if check_division_by_zero {
                for i in 0..arr2.len() {
                    if arr2.is_valid(i) && arr2.value(i) == 0.0 {
                        return Err(CrossbowError::DivisionByZero);
                    }
                }
            }

            let mut builder = arrow::array::Float64Builder::with_capacity(arr1.len());
            for i in 0..arr1.len() {
                if arr1.is_valid(i) && arr2.is_valid(i) {
                    builder.append_value(op_f64(arr1.value(i), arr2.value(i)));
                } else {
                    builder.append_null();
                }
            }
            Ok(Series::new(
                format!("{} {} {}", s1.name(), op_name, s2.name()),
                Arc::new(builder.finish()),
            ))
        }
        _ => Err(CrossbowError::TypeMismatch(
            format!("{} not supported between {:?} and {:?}", op_name, s1.dtype(), s2.dtype())
        )),
    }
}

fn binary_op_i32<F>(
    s1: &Series,
    s2: &Series,
    op_name: &str,
    op_i32: F,
    check_division_by_zero: bool,
) -> Result<Series, CrossbowError>
where
    F: Fn(i32, i32) -> Option<i32>,
{
    if s1.len() != s2.len() {
        return Err(CrossbowError::MismatchedColumnLengths);
    }

    match (s1.dtype(), s2.dtype()) {
        (DataType::Int32, DataType::Int32) => {
            let arr1 = s1.as_primitive::<arrow::array::Int32Array>()
                .ok_or_else(|| expected_array("Int32Array"))?;
            let arr2 = s2.as_primitive::<arrow::array::Int32Array>()
                .ok_or_else(|| expected_array("Int32Array"))?;

            if check_division_by_zero {
                for i in 0..arr2.len() {
                    if arr2.is_valid(i) && arr2.value(i) == 0 {
                        return Err(CrossbowError::DivisionByZero);
                    }
                }
            }

            let mut builder = arrow::array::Int32Builder::with_capacity(arr1.len());
            for i in 0..arr1.len() {
                if arr1.is_valid(i) && arr2.is_valid(i) {
                    match op_i32(arr1.value(i), arr2.value(i)) {
                        Some(v) => builder.append_value(v),
                        None => return Err(CrossbowError::ArithmeticOverflow(
                            format!("{} {} {} overflows Int32 at index {}",
                                arr1.value(i), op_name, arr2.value(i), i)
                        )),
                    }
                } else {
                    builder.append_null();
                }
            }
            Ok(Series::new(
                format!("{} {} {}", s1.name(), op_name, s2.name()),
                Arc::new(builder.finish()),
            ))
        }
        _ => Err(CrossbowError::TypeMismatch(
            format!("{} not supported between {:?} and {:?}", op_name, s1.dtype(), s2.dtype())
        )),
    }
}

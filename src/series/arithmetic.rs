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

impl Series {
    /// Element-wise addition: `self[i] + other[i]`.
    ///
    /// Both `Series` must have the same length and compatible types.
    /// Supports `Int32` and `Float64`.
    pub fn add(&self, other: &Series) -> Result<Series, CrossbowError> {
        if self.len() != other.len() {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        match (self.dtype(), other.dtype()) {
            (DataType::Int32, DataType::Int32) => {
                let arr1 = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let arr2 = other.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut builder = arrow::array::Int32Builder::with_capacity(arr1.len());
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) && arr2.is_valid(i) {
                        match arr1.value(i).checked_add(arr2.value(i)) {
                            Some(v) => builder.append_value(v),
                            None => return Err(CrossbowError::ArithmeticOverflow(
                                format!("{} + {} overflows Int32 at index {}", arr1.value(i), arr2.value(i), i)
                            )),
                        }
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{} + {}", self.name(), other.name()), Arc::new(builder.finish())))
            }
            (DataType::Float64, DataType::Float64) => {
                let arr1 = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let arr2 = other.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut builder = arrow::array::Float64Builder::with_capacity(arr1.len());
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) && arr2.is_valid(i) {
                        builder.append_value(arr1.value(i) + arr2.value(i));
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{} + {}", self.name(), other.name()), Arc::new(builder.finish())))
            }
            _ => Err(CrossbowError::TypeMismatch(
                format!("Addition not supported between {:?} and {:?}", self.dtype(), other.dtype())
            )),
        }
    }

    /// Element-wise subtraction: `self[i] - other[i]`.
    pub fn subtract(&self, other: &Series) -> Result<Series, CrossbowError> {
        if self.len() != other.len() {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        match (self.dtype(), other.dtype()) {
            (DataType::Int32, DataType::Int32) => {
                let arr1 = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let arr2 = other.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut builder = arrow::array::Int32Builder::with_capacity(arr1.len());
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) && arr2.is_valid(i) {
                        match arr1.value(i).checked_sub(arr2.value(i)) {
                            Some(v) => builder.append_value(v),
                            None => return Err(CrossbowError::ArithmeticOverflow(
                                format!("{} - {} overflows Int32 at index {}", arr1.value(i), arr2.value(i), i)
                            )),
                        }
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{} - {}", self.name(), other.name()), Arc::new(builder.finish())))
            }
            (DataType::Float64, DataType::Float64) => {
                let arr1 = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let arr2 = other.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut builder = arrow::array::Float64Builder::with_capacity(arr1.len());
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) && arr2.is_valid(i) {
                        builder.append_value(arr1.value(i) - arr2.value(i));
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{} - {}", self.name(), other.name()), Arc::new(builder.finish())))
            }
            _ => Err(CrossbowError::TypeMismatch(
                format!("Subtraction not supported between {:?} and {:?}", self.dtype(), other.dtype())
            )),
        }
    }

    /// Element-wise multiplication: `self[i] * other[i]`.
    pub fn multiply(&self, other: &Series) -> Result<Series, CrossbowError> {
        if self.len() != other.len() {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        match (self.dtype(), other.dtype()) {
            (DataType::Int32, DataType::Int32) => {
                let arr1 = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let arr2 = other.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut builder = arrow::array::Int32Builder::with_capacity(arr1.len());
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) && arr2.is_valid(i) {
                        match arr1.value(i).checked_mul(arr2.value(i)) {
                            Some(v) => builder.append_value(v),
                            None => return Err(CrossbowError::ArithmeticOverflow(
                                format!("{} * {} overflows Int32 at index {}", arr1.value(i), arr2.value(i), i)
                            )),
                        }
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{} * {}", self.name(), other.name()), Arc::new(builder.finish())))
            }
            (DataType::Float64, DataType::Float64) => {
                let arr1 = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let arr2 = other.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut builder = arrow::array::Float64Builder::with_capacity(arr1.len());
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) && arr2.is_valid(i) {
                        builder.append_value(arr1.value(i) * arr2.value(i));
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{} * {}", self.name(), other.name()), Arc::new(builder.finish())))
            }
            _ => Err(CrossbowError::TypeMismatch(
                format!("Multiplication not supported between {:?} and {:?}", self.dtype(), other.dtype())
            )),
        }
    }

    /// Element-wise division: `self[i] / other[i]`.
    ///
    /// Returns [`CrossbowError::DivisionByZero`] if any element of `other` is zero.
    pub fn divide(&self, other: &Series) -> Result<Series, CrossbowError> {
        if self.len() != other.len() {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        match (self.dtype(), other.dtype()) {
            (DataType::Int32, DataType::Int32) => {
                let arr1 = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let arr2 = other.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;

                for i in 0..arr2.len() {
                    if arr2.is_valid(i) && arr2.value(i) == 0 {
                        return Err(CrossbowError::DivisionByZero);
                    }
                }

                let mut builder = arrow::array::Int32Builder::with_capacity(arr1.len());
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) && arr2.is_valid(i) {
                        builder.append_value(arr1.value(i) / arr2.value(i));
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{} / {}", self.name(), other.name()), Arc::new(builder.finish())))
            }
            (DataType::Float64, DataType::Float64) => {
                let arr1 = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let arr2 = other.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;

                for i in 0..arr2.len() {
                    if arr2.is_valid(i) && arr2.value(i) == 0.0 {
                        return Err(CrossbowError::DivisionByZero);
                    }
                }

                let mut builder = arrow::array::Float64Builder::with_capacity(arr1.len());
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) && arr2.is_valid(i) {
                        builder.append_value(arr1.value(i) / arr2.value(i));
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{} / {}", self.name(), other.name()), Arc::new(builder.finish())))
            }
            _ => Err(CrossbowError::TypeMismatch(
                format!("Division not supported between {:?} and {:?}", self.dtype(), other.dtype())
            )),
        }
    }

    /// Element-wise modulo: `self[i] % other[i]`. Only supports `Int32`.
    pub fn modulo(&self, other: &Series) -> Result<Series, CrossbowError> {
        if self.len() != other.len() {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        match (self.dtype(), other.dtype()) {
            (DataType::Int32, DataType::Int32) => {
                let arr1 = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let arr2 = other.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;

                for i in 0..arr2.len() {
                    if arr2.is_valid(i) && arr2.value(i) == 0 {
                        return Err(CrossbowError::DivisionByZero);
                    }
                }

                let mut builder = arrow::array::Int32Builder::with_capacity(arr1.len());
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) && arr2.is_valid(i) {
                        builder.append_value(arr1.value(i) % arr2.value(i));
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{} % {}", self.name(), other.name()), Arc::new(builder.finish())))
            }
            _ => Err(CrossbowError::TypeMismatch(
                format!("Modulo not supported between {:?} and {:?}", self.dtype(), other.dtype())
            )),
        }
    }
}

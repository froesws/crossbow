//! Aggregation and statistical operations on `Series`.
//!
//! All functions ignore null values. Standard deviation and variance use
//! the **sample** formula (n-1 denominator).

use arrow::datatypes::DataType;
use arrow::array::Array;
use crate::error::CrossbowError;
use crate::series::Series;

impl Series {
    /// Sum of all non-null values. Returns `0.0` if all values are null.
    pub fn sum(&self) -> Result<f64, CrossbowError> {
        match self.dtype() {
            DataType::Int32 => {
                let arr = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut sum = 0i64;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        sum += arr.value(i) as i64;
                    }
                }
                Ok(sum as f64)
            }
            DataType::Float64 => {
                let arr = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut sum_val = 0.0;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        sum_val += arr.value(i);
                    }
                }
                Ok(sum_val)
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("sum() not supported for {:?}", self.dtype())
            )),
        }
    }

    /// Arithmetic mean of all non-null values. Returns `f64::NAN` if the
    /// `Series` is empty or all values are null (mean of empty set is undefined).
    pub fn mean(&self) -> Result<f64, CrossbowError> {
        match self.dtype() {
            DataType::Int32 | DataType::Float64 => {
                let sum = self.sum()?;
                let count = self.count_non_null()?;
                if count == 0 {
                    Ok(f64::NAN)
                } else {
                    Ok(sum / count as f64)
                }
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("mean() not supported for {:?}", self.dtype())
            )),
        }
    }

    /// Minimum non-null value. Returns `None` if all values are null or the `Series` is empty.
    pub fn min(&self) -> Result<Option<f64>, CrossbowError> {
        match self.dtype() {
            DataType::Int32 => {
                let arr = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut min_val: Option<i32> = None;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let val = arr.value(i);
                        min_val = Some(match min_val {
                            None => val,
                            Some(m) => if val < m { val } else { m },
                        });
                    }
                }
                Ok(min_val.map(|v| v as f64))
            }
            DataType::Float64 => {
                let arr = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut min_val: Option<f64> = None;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let val = arr.value(i);
                        min_val = Some(match min_val {
                            None => val,
                            Some(m) => if val < m { val } else { m },
                        });
                    }
                }
                Ok(min_val)
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("min() not supported for {:?}", self.dtype())
            )),
        }
    }

    /// Maximum non-null value. Returns `None` if all values are null or the `Series` is empty.
    pub fn max(&self) -> Result<Option<f64>, CrossbowError> {
        match self.dtype() {
            DataType::Int32 => {
                let arr = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut max_val: Option<i32> = None;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let val = arr.value(i);
                        max_val = Some(match max_val {
                            None => val,
                            Some(m) => if val > m { val } else { m },
                        });
                    }
                }
                Ok(max_val.map(|v| v as f64))
            }
            DataType::Float64 => {
                let arr = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut max_val: Option<f64> = None;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let val = arr.value(i);
                        max_val = Some(match max_val {
                            None => val,
                            Some(m) => if val > m { val } else { m },
                        });
                    }
                }
                Ok(max_val)
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("max() not supported for {:?}", self.dtype())
            )),
        }
    }

    /// Total number of elements (including nulls). Same as [`Series::len`].
    pub fn count(&self) -> usize {
        self.len()
    }

    /// Count of non-null elements.
    pub fn count_non_null(&self) -> Result<usize, CrossbowError> {
        let mut count = 0;
        for i in 0..self.len() {
            if self.data.is_valid(i) {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Sample standard deviation (n-1 divisor). Returns `f64::NAN` if fewer
    /// than 2 non-null values.
    pub fn std(&self) -> Result<f64, CrossbowError> {
        let (variance, count) = self.variance_sum()?;
        if count < 2 { return Ok(f64::NAN); }
        Ok((variance / (count - 1) as f64).sqrt())
    }

    /// Sample variance (n-1 divisor). Returns `f64::NAN` if fewer than
    /// 2 non-null values.
    pub fn var(&self) -> Result<f64, CrossbowError> {
        let (variance, count) = self.variance_sum()?;
        if count < 2 { return Ok(f64::NAN); }
        Ok(variance / (count - 1) as f64)
    }

    fn variance_sum(&self) -> Result<(f64, usize), CrossbowError> {
        if self.count_non_null()? < 2 {
            return Ok((0.0, 0));
        }
        let mean = self.mean()?;
        let mut variance = 0.0;
        let mut count = 0;
        match self.dtype() {
            DataType::Int32 => {
                let arr = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        variance += (arr.value(i) as f64 - mean).powi(2);
                        count += 1;
                    }
                }
            }
            DataType::Float64 => {
                let arr = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        variance += (arr.value(i) - mean).powi(2);
                        count += 1;
                    }
                }
            }
            _ => {
                return Err(CrossbowError::OperationNotSupported(
                    format!("std/var not supported for {:?}", self.dtype())
                ));
            }
        }
        Ok((variance, count))
    }
}

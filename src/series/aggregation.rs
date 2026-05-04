//! Aggregation and statistical operations on `Series`.
//!
//! All functions ignore null values. Standard deviation and variance use
//! the **sample** formula (n-1 denominator).

use crate::error::CrossbowError;
use crate::expected_array;
use crate::series::Series;
use arrow::array::Array;
use arrow::datatypes::DataType;

macro_rules! sum_int_body {
    ($slf:expr, $array_ty:ty) => {{
        let arr = $slf
            .as_primitive::<$array_ty>()
            .ok_or_else(|| expected_array(stringify!($array_ty)))?;
        let mut sum: i64 = 0;
        for i in 0..arr.len() {
            if arr.is_valid(i) {
                sum += arr.value(i) as i64;
            }
        }
        Ok(sum as f64)
    }};
}

macro_rules! sum_f64_body {
    ($slf:expr, $array_ty:ty) => {{
        let arr = $slf
            .as_primitive::<$array_ty>()
            .ok_or_else(|| expected_array(stringify!($array_ty)))?;
        let mut sum: f64 = 0.0;
        for i in 0..arr.len() {
            if arr.is_valid(i) {
                sum += arr.value(i) as f64;
            }
        }
        Ok(sum)
    }};
}

macro_rules! minmax_body {
    ($slf:expr, $array_ty:ty, $cmp:tt) => {{
        let arr = $slf
            .as_primitive::<$array_ty>()
            .ok_or_else(|| expected_array(stringify!($array_ty)))?;
        let mut opt: Option<f64> = None;
        for i in 0..arr.len() {
            if arr.is_valid(i) {
                let val = arr.value(i) as f64;
                opt = Some(match opt {
                    None => val,
                    Some(m) => {
                        if val $cmp m {
                            val
                        } else {
                            m
                        }
                    }
                });
            }
        }
        Ok(opt)
    }};
}

macro_rules! variance_body {
    ($slf:expr, $array_ty:ty, $mean:expr, $var:ident, $cnt:ident) => {{
        let arr = $slf
            .as_primitive::<$array_ty>()
            .ok_or_else(|| expected_array(stringify!($array_ty)))?;
        for i in 0..arr.len() {
            if arr.is_valid(i) {
                $var += (arr.value(i) as f64 - $mean).powi(2);
                $cnt += 1;
            }
        }
    }};
}

impl Series {
    /// Sum of all non-null values. Returns `0.0` if all values are null.
    ///
    /// Supports `Int32`, `Int64`, `Float32`, and `Float64` columns.
    /// Always returns `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![1i32, 2, 3, 4]);
    /// assert_eq!(s.sum().unwrap(), 10.0);
    ///
    /// let s2 = Series::from("y", vec![1i64, 2, 3]);
    /// assert_eq!(s2.sum().unwrap(), 6.0);
    /// ```
    pub fn sum(&self) -> Result<f64, CrossbowError> {
        match self.dtype() {
            DataType::Int32 => sum_int_body!(self, arrow::array::Int32Array),
            DataType::Int64 => sum_int_body!(self, arrow::array::Int64Array),
            DataType::Float32 => sum_f64_body!(self, arrow::array::Float32Array),
            DataType::Float64 => sum_f64_body!(self, arrow::array::Float64Array),
            _ => Err(CrossbowError::OperationNotSupported(format!(
                "sum() not supported for {:?}",
                self.dtype()
            ))),
        }
    }

    /// Arithmetic mean of all non-null values. Returns `f64::NAN` if the
    /// `Series` is empty or all values are null (mean of empty set is undefined).
    ///
    /// Supports `Int32`, `Int64`, `Float32`, and `Float64` columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![2.0, 4.0, 6.0]);
    /// assert_eq!(s.mean().unwrap(), 4.0);
    /// ```
    pub fn mean(&self) -> Result<f64, CrossbowError> {
        match self.dtype() {
            DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64 => {
                let sum = self.sum()?;
                let count = self.count_non_null()?;
                if count == 0 {
                    Ok(f64::NAN)
                } else {
                    Ok(sum / count as f64)
                }
            }
            _ => Err(CrossbowError::OperationNotSupported(format!(
                "mean() not supported for {:?}",
                self.dtype()
            ))),
        }
    }

    /// Minimum non-null value. Returns `None` if all values are null
    /// or the `Series` is empty.
    ///
    /// Supports `Int32`, `Int64`, `Float32`, `Float64`, `Date32`,
    /// `Date64`, and `Timestamp(Millisecond)` columns.
    /// Always returns `Option<f64>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![3i32, 1, 4, 2]);
    /// assert_eq!(s.min().unwrap(), Some(1.0));
    /// ```
    pub fn min(&self) -> Result<Option<f64>, CrossbowError> {
        match self.dtype() {
            DataType::Int32 => minmax_body!(self, arrow::array::Int32Array, <),
            DataType::Int64 => minmax_body!(self, arrow::array::Int64Array, <),
            DataType::Float32 => minmax_body!(self, arrow::array::Float32Array, <),
            DataType::Float64 => minmax_body!(self, arrow::array::Float64Array, <),
            DataType::Date32 => minmax_body!(self, arrow::array::Date32Array, <),
            DataType::Date64 => minmax_body!(self, arrow::array::Date64Array, <),
            DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None) => {
                minmax_body!(self, arrow::array::TimestampMillisecondArray, <)
            }
            _ => Err(CrossbowError::OperationNotSupported(format!(
                "min() not supported for {:?}",
                self.dtype()
            ))),
        }
    }

    /// Maximum non-null value. Returns `None` if all values are null
    /// or the `Series` is empty.
    ///
    /// Supports `Int32`, `Int64`, `Float32`, `Float64`, `Date32`,
    /// `Date64`, and `Timestamp(Millisecond)` columns.
    /// Always returns `Option<f64>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![3i32, 1, 4, 2]);
    /// assert_eq!(s.max().unwrap(), Some(4.0));
    /// ```
    pub fn max(&self) -> Result<Option<f64>, CrossbowError> {
        match self.dtype() {
            DataType::Int32 => minmax_body!(self, arrow::array::Int32Array, >),
            DataType::Int64 => minmax_body!(self, arrow::array::Int64Array, >),
            DataType::Float32 => minmax_body!(self, arrow::array::Float32Array, >),
            DataType::Float64 => minmax_body!(self, arrow::array::Float64Array, >),
            DataType::Date32 => minmax_body!(self, arrow::array::Date32Array, >),
            DataType::Date64 => minmax_body!(self, arrow::array::Date64Array, >),
            DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None) => {
                minmax_body!(self, arrow::array::TimestampMillisecondArray, >)
            }
            _ => Err(CrossbowError::OperationNotSupported(format!(
                "max() not supported for {:?}",
                self.dtype()
            ))),
        }
    }

    /// Total number of elements (including nulls). Same as [`Series::len`].
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![1i32, 2, 3]);
    /// assert_eq!(s.count(), 3);
    /// ```
    pub fn count(&self) -> usize {
        self.len()
    }

    /// Count of non-null elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![Some(1i32), None, Some(3)]);
    /// assert_eq!(s.count_non_null().unwrap(), 2);
    /// ```
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
    ///
    /// Supports `Int32`, `Int64`, `Float32`, and `Float64` columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    /// let stddev = s.std().unwrap();
    /// assert!(stddev > 0.0);
    /// ```
    pub fn std(&self) -> Result<f64, CrossbowError> {
        let (variance, count) = self.variance_sum()?;
        if count < 2 {
            return Ok(f64::NAN);
        }
        Ok((variance / (count - 1) as f64).sqrt())
    }

    /// Sample variance (n-1 divisor). Returns `f64::NAN` if fewer than
    /// 2 non-null values.
    ///
    /// Supports `Int32`, `Int64`, `Float32`, and `Float64` columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![2.0, 4.0, 6.0]);
    /// assert!(s.var().unwrap() > 0.0);
    /// ```
    pub fn var(&self) -> Result<f64, CrossbowError> {
        let (variance, count) = self.variance_sum()?;
        if count < 2 {
            return Ok(f64::NAN);
        }
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
                variance_body!(self, arrow::array::Int32Array, mean, variance, count)
            }
            DataType::Int64 => {
                variance_body!(self, arrow::array::Int64Array, mean, variance, count)
            }
            DataType::Float32 => {
                variance_body!(self, arrow::array::Float32Array, mean, variance, count)
            }
            DataType::Float64 => {
                variance_body!(self, arrow::array::Float64Array, mean, variance, count)
            }
            _ => {
                return Err(CrossbowError::OperationNotSupported(format!(
                    "std/var not supported for {:?}",
                    self.dtype()
                )));
            }
        }
        Ok((variance, count))
    }
}

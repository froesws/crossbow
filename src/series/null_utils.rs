//! Null handling, value access, and type-checking utilities for `Series`.

use crate::error::CrossbowError;
use crate::series::Series;
use crate::{build_column, build_string_column, expected_array};
use arrow::array::Array;
use arrow::datatypes::DataType;
use std::sync::Arc;

macro_rules! fill_null_body {
    ($slf:expr, $fill:expr, $array_ty:ty, $builder_ty:ty) => {{
        let arr1 = $slf
            .as_primitive::<$array_ty>()
            .ok_or_else(|| expected_array(stringify!($array_ty)))?;
        let arr2 = $fill
            .as_primitive::<$array_ty>()
            .ok_or_else(|| expected_array(stringify!($array_ty)))?;
        let mut builder = <$builder_ty>::with_capacity($slf.len());
        for i in 0..arr1.len() {
            if arr1.is_valid(i) {
                builder.append_value(arr1.value(i));
            } else if arr2.is_valid(i) {
                builder.append_value(arr2.value(i));
            } else {
                builder.append_null();
            }
        }
        Ok(Series::new(
            format!("{}_filled", $slf.name()),
            Arc::new(builder.finish()),
        ))
    }};
}

impl Series {
    /// Returns a boolean `Series` indicating which elements are null.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![1i32, 2, 3]);
    /// let nulls = s.is_null().unwrap();
    /// assert_eq!(nulls.len(), 3);
    /// ```
    pub fn is_null(&self) -> Result<Series, CrossbowError> {
        self.null_mask(false)
    }

    /// Returns a boolean `Series` indicating which elements are **not** null.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![Some(1i32), None, Some(3)]);
    /// let not_null = s.is_not_null().unwrap();
    /// assert_eq!(not_null.len(), 3);
    /// ```
    pub fn is_not_null(&self) -> Result<Series, CrossbowError> {
        self.null_mask(true)
    }

    /// Fills null values with the corresponding value from another `Series`.
    ///
    /// Both `Series` must have the same length and data type.
    /// Supports `Int32`, `Float64`, `Date32`, `Date64`, and `Timestamp(Millisecond)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let a = Series::from("a", vec![Some(1i32), None, Some(3)]);
    /// let b = Series::from("b", vec![10i32, 20, 30]);
    /// let filled = a.fill_null(&b).unwrap();
    /// assert_eq!(filled.len(), 3);
    /// ```
    pub fn fill_null(&self, fill_value: &Series) -> Result<Series, CrossbowError> {
        if self.len() != fill_value.len() {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        if self.dtype() != fill_value.dtype() {
            return Err(CrossbowError::TypeMismatch(format!(
                "Cannot fill {:?} with {:?}",
                self.dtype(),
                fill_value.dtype()
            )));
        }

        match self.dtype() {
            DataType::Int32 => fill_null_body!(
                self,
                fill_value,
                arrow::array::Int32Array,
                arrow::array::Int32Builder
            ),
            DataType::Float64 => fill_null_body!(
                self,
                fill_value,
                arrow::array::Float64Array,
                arrow::array::Float64Builder
            ),
            DataType::Date32 => fill_null_body!(
                self,
                fill_value,
                arrow::array::Date32Array,
                arrow::array::Date32Builder
            ),
            DataType::Date64 => fill_null_body!(
                self,
                fill_value,
                arrow::array::Date64Array,
                arrow::array::Date64Builder
            ),
            DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None) => {
                fill_null_body!(
                    self,
                    fill_value,
                    arrow::array::TimestampMillisecondArray,
                    arrow::array::TimestampMillisecondBuilder
                )
            }
            _ => Err(CrossbowError::OperationNotSupported(format!(
                "fill_null not supported for {:?}",
                self.dtype()
            ))),
        }
    }

    /// Removes all null values, returning a shorter `Series`.
    ///
    /// Supports `Int32`, `Float64`, `Utf8`, `Date32`, `Date64`,
    /// and `Timestamp(Millisecond)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![Some(1i32), None, Some(3)]);
    /// let cleaned = s.drop_null().unwrap();
    /// assert_eq!(cleaned.len(), 2);
    /// ```
    pub fn drop_null(&self) -> Result<Series, CrossbowError> {
        let valid: Vec<usize> = (0..self.len()).filter(|&i| self.data.is_valid(i)).collect();
        let n = valid.len();
        let d = self.data();
        let built = match self.dtype() {
            DataType::Int32 => build_column!(
                d.as_ref(),
                &valid,
                arrow::array::Int32Builder,
                arrow::array::Int32Array,
                n
            ),
            DataType::Float64 => build_column!(
                d.as_ref(),
                &valid,
                arrow::array::Float64Builder,
                arrow::array::Float64Array,
                n
            ),
            DataType::Utf8 => build_string_column!(d.as_ref(), &valid, n),
            DataType::Date32 => build_column!(
                d.as_ref(),
                &valid,
                arrow::array::Date32Builder,
                arrow::array::Date32Array,
                n
            ),
            DataType::Date64 => build_column!(
                d.as_ref(),
                &valid,
                arrow::array::Date64Builder,
                arrow::array::Date64Array,
                n
            ),
            DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None) => build_column!(
                d.as_ref(),
                &valid,
                arrow::array::TimestampMillisecondBuilder,
                arrow::array::TimestampMillisecondArray,
                n
            ),
            _ => {
                return Err(CrossbowError::OperationNotSupported(format!(
                    "drop_null not supported for {:?}",
                    self.dtype()
                )));
            }
        };
        Ok(Series::new(format!("{}_no_null", self.name()), built))
    }

    /// Returns the value at `index` as a `String`.
    ///
    /// UTF-8 values are returned with surrounding quotes.
    /// Returns [`CrossbowError::IndexOutOfBounds`] if `index` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![10i32]);
    /// assert_eq!(s.value_at(0).unwrap(), "10");
    /// ```
    pub fn value_at(&self, index: usize) -> Result<String, CrossbowError> {
        if index >= self.len() {
            return Err(CrossbowError::IndexOutOfBounds(index));
        }
        Ok(self.get_value_as_string(index))
    }

    /// Returns a zero-copy slice of the `Series` from `offset` with the given `length`.
    ///
    /// Returns [`CrossbowError::IndexOutOfBounds`] if the slice exceeds the `Series` length.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![1i32, 2, 3, 4, 5]);
    /// let slice = s.slice(1, 3).unwrap();
    /// assert_eq!(slice.len(), 3);
    /// ```
    pub fn slice(&self, offset: usize, length: usize) -> Result<Series, CrossbowError> {
        if offset + length > self.len() {
            return Err(CrossbowError::IndexOutOfBounds(offset + length));
        }
        let sliced = self.data.slice(offset, length);
        Ok(Series::new(format!("{}_slice", self.name()), sliced))
    }

    /// Returns `true` if the `Series` holds numeric data (integer or float).
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![1i32]);
    /// assert!(s.is_numeric());
    /// ```
    pub fn is_numeric(&self) -> bool {
        matches!(
            self.dtype(),
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64
        )
    }

    /// Returns `true` if the `Series` holds string data.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec!["hello"]);
    /// assert!(s.is_string());
    /// ```
    pub fn is_string(&self) -> bool {
        matches!(self.dtype(), DataType::Utf8 | DataType::LargeUtf8)
    }

    /// Returns `true` if the `Series` holds boolean data.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![true, false]);
    /// assert!(s.is_boolean());
    /// ```
    pub fn is_boolean(&self) -> bool {
        matches!(self.dtype(), DataType::Boolean)
    }

    // Builds a boolean Series marking null (or non-null when `invert` is true) positions.
    fn null_mask(&self, invert: bool) -> Result<Series, CrossbowError> {
        let mut builder = arrow::array::BooleanBuilder::with_capacity(self.len());
        let suffix = if invert { "_is_not_null" } else { "_is_null" };
        for i in 0..self.len() {
            let is_null = self.data.is_null(i);
            builder.append_value(if invert { !is_null } else { is_null });
        }
        Ok(Series::new(
            format!("{}{}", self.name(), suffix),
            Arc::new(builder.finish()),
        ))
    }
}

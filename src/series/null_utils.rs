//! Null handling, value access, and type-checking utilities for `Series`.

use std::sync::Arc;
use arrow::datatypes::DataType;
use arrow::array::Array;
use crate::error::CrossbowError;
use crate::series::Series;
use crate::{build_column, build_string_column, expected_array};

impl Series {
    /// Returns a boolean `Series` indicating which elements are null.
    pub fn is_null(&self) -> Result<Series, CrossbowError> {
        self.null_mask(false)
    }

    /// Returns a boolean `Series` indicating which elements are **not** null.
    pub fn is_not_null(&self) -> Result<Series, CrossbowError> {
        self.null_mask(true)
    }

    /// Fills null values with the corresponding value from another `Series`.
    ///
    /// Both `Series` must have the same length and data type. Supports `Int32` and `Float64`.
    pub fn fill_null(&self, fill_value: &Series) -> Result<Series, CrossbowError> {
        if self.len() != fill_value.len() {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        if self.dtype() != fill_value.dtype() {
            return Err(CrossbowError::TypeMismatch(
                format!("Cannot fill {:?} with {:?}", self.dtype(), fill_value.dtype())
            ));
        }

        match self.dtype() {
            DataType::Int32 => {
                let arr1 = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || expected_array("Int32Array"),
                )?;
                let arr2 = fill_value.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || expected_array("Int32Array"),
                )?;
                let mut builder = arrow::array::Int32Builder::with_capacity(self.len());
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) {
                        builder.append_value(arr1.value(i));
                    } else if arr2.is_valid(i) {
                        builder.append_value(arr2.value(i));
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{}_filled", self.name()), Arc::new(builder.finish())))
            }
            DataType::Float64 => {
                let arr1 = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || expected_array("Float64Array"),
                )?;
                let arr2 = fill_value.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || expected_array("Float64Array"),
                )?;
                let mut builder = arrow::array::Float64Builder::with_capacity(self.len());
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) {
                        builder.append_value(arr1.value(i));
                    } else if arr2.is_valid(i) {
                        builder.append_value(arr2.value(i));
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{}_filled", self.name()), Arc::new(builder.finish())))
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("fill_null not supported for {:?}", self.dtype())
            )),
        }
    }

    /// Removes all null values, returning a shorter `Series`.
    ///
    /// Supports `Int32`, `Float64`, and `Utf8`.
    pub fn drop_null(&self) -> Result<Series, CrossbowError> {
        let valid: Vec<usize> = (0..self.len()).filter(|&i| self.data.is_valid(i)).collect();
        let n = valid.len();
        let d = self.data();
        let built = match self.dtype() {
            DataType::Int32 => build_column!(d.as_ref(), &valid, arrow::array::Int32Builder, arrow::array::Int32Array, n),
            DataType::Float64 => build_column!(d.as_ref(), &valid, arrow::array::Float64Builder, arrow::array::Float64Array, n),
            DataType::Utf8 => build_string_column!(d.as_ref(), &valid, n),
            _ => return Err(CrossbowError::OperationNotSupported(
                format!("drop_null not supported for {:?}", self.dtype())
            )),
        };
        Ok(Series::new(format!("{}_no_null", self.name()), built))
    }

    /// Returns the value at `index` as a `String`.
    ///
    /// UTF-8 values are returned with surrounding quotes.
    /// Returns [`CrossbowError::IndexOutOfBounds`] if `index` is out of range.
    pub fn value_at(&self, index: usize) -> Result<String, CrossbowError> {
        if index >= self.len() {
            return Err(CrossbowError::IndexOutOfBounds(index));
        }
        Ok(self.get_value_as_string(index))
    }

    /// Returns a zero-copy slice of the `Series` from `offset` with the given `length`.
    ///
    /// Returns [`CrossbowError::IndexOutOfBounds`] if the slice exceeds the `Series` length.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Series, CrossbowError> {
        if offset + length > self.len() {
            return Err(CrossbowError::IndexOutOfBounds(offset + length));
        }
        let sliced = self.data.slice(offset, length);
        Ok(Series::new(format!("{}_slice", self.name()), sliced))
    }

    /// Returns `true` if the `Series` holds numeric data (integer or float).
    pub fn is_numeric(&self) -> bool {
        matches!(self.dtype(), 
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 |
            DataType::Float32 | DataType::Float64
        )
    }

    /// Returns `true` if the `Series` holds string data.
    pub fn is_string(&self) -> bool {
        matches!(self.dtype(), DataType::Utf8 | DataType::LargeUtf8)
    }

    /// Returns `true` if the `Series` holds boolean data.
    pub fn is_boolean(&self) -> bool {
        matches!(self.dtype(), DataType::Boolean)
    }

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

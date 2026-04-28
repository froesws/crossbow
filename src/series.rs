//! Series module
//!
//! A series is a named, strongly-typed column of data.
//!
//! A `Series` is the primary unit of data in this crate. It is a wrapper
//! around an [`ArrayRef`] from the `arrow-rs` crate, which allows it to
//! efficiently hold data of different types (like `i32`, `&str`, etc.).
//!
//! For a professional-grade library, consider using [polars](https://www.pola.rs/).
//! For more information on Apache Arrow, visit [arrow-rs](https://arrow.apache.org/).
//!
//! MIT License
//!
//! Copyright (c) [2025] [William Froes]
//! Project: [crossbow]
//! Developed at: [Uergs -- Universidade Estadual do Rio Grande do Sul]
use arrow::array::{Array, ArrayRef};
use arrow::datatypes::DataType;
use std::sync::Arc;

use crate::error::CrossbowError;

#[cfg(test)]
mod unit_test;

/// Represents a named, strongly-typed column of data.
///
/// A `Series` is the primary unit of data in this crate. It is a wrapper
/// around an [`ArrayRef`] from the `arrow-rs` crate, which allows it to
/// efficiently hold data of different types (like `i32`, `&str`, etc.).
#[derive(Debug, Clone)]
pub struct Series {
    name: String,
    data: ArrayRef,
}

impl Series {
    /// Creates a new Series with the given name and data.
    pub fn new(name: impl Into<String>, data: ArrayRef) -> Self {
        Series {
            name: name.into(),
            data,
        }
    }

    /// Creates a Series from a vector of values and a name.
    pub fn from<T>(name: &str, data: Vec<T>) -> Series
    where
        T: 'static,
        Vec<T>: IntoSeries,
    {
        data.into_series(name)
    }

    /// Returns the name of this [`Series`].
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the data of this [`Series`].
    #[allow(dead_code)]
    pub fn data(&self) -> &ArrayRef {
        &self.data
    }

    /// Returns the data type of the underlying Arrow array.
    pub fn dtype(&self) -> &DataType {
        self.data.data_type()
    }

    /// Returns the length of the Series.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the Series contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Attempts to downcast the internal data to a specific Arrow array type.
    pub fn as_primitive<T>(&self) -> Option<&T>
    where
        T: 'static,
    {
        self.data.as_any().downcast_ref::<T>()
    }

    /// Helper function to get a string representation of a value at a given index.
    pub(crate) fn get_value_as_string(&self, index: usize) -> String {
        if self.data.is_null(index) {
            return "null".to_string();
        }

        match self.dtype() {
            DataType::Int32 => {
                let array = self
                    .data
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .unwrap();
                array.value(index).to_string()
            }
            DataType::Float64 => {
                let array = self
                    .data
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();
                array.value(index).to_string()
            }
            DataType::Utf8 => {
                let array = self
                    .data
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .unwrap();
                format!("\"{}\"", array.value(index))
            }
            DataType::Boolean => {
                let array = self
                    .data
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .unwrap();
                array.value(index).to_string()
            }
            other_type => format!("Unsupported type: {:?}", other_type),
        }
    }
}

pub trait IntoSeries {
    fn into_series(self, name: &str) -> Series;
}

impl IntoSeries for Vec<&str> {
    fn into_series(self, name: &str) -> Series {
        let array = arrow::array::StringArray::from(self);
        Series::new(name, Arc::new(array))
    }
}

impl IntoSeries for Vec<Option<&str>> {
    fn into_series(self, name: &str) -> Series {
        let array = arrow::array::StringArray::from(self);
        Series::new(name, Arc::new(array))
    }
}

impl IntoSeries for Vec<String> {
    fn into_series(self, name: &str) -> Series {
        let array =
            arrow::array::StringArray::from(self.iter().map(|s| s.as_str()).collect::<Vec<&str>>());
        Series::new(name, Arc::new(array))
    }
}

macro_rules! impl_into_series_for_numerics {
    ($T:ty, $A:ty) => {
        impl IntoSeries for Vec<$T> {
            fn into_series(self, name: &str) -> Series {
                let array = <$A>::from(self);
                Series::new(name, Arc::new(array))
            }
        }
        impl IntoSeries for Vec<Option<$T>> {
            fn into_series(self, name: &str) -> Series {
                let array = <$A>::from(self);
                Series::new(name, Arc::new(array))
            }
        }
    };
}

impl_into_series_for_numerics!(bool, arrow::array::BooleanArray);
impl_into_series_for_numerics!(i8, arrow::array::Int8Array);
impl_into_series_for_numerics!(i16, arrow::array::Int16Array);
impl_into_series_for_numerics!(i32, arrow::array::Int32Array);
impl_into_series_for_numerics!(i64, arrow::array::Int64Array);
impl_into_series_for_numerics!(u8, arrow::array::UInt8Array);
impl_into_series_for_numerics!(u16, arrow::array::UInt16Array);
impl_into_series_for_numerics!(u32, arrow::array::UInt32Array);
impl_into_series_for_numerics!(u64, arrow::array::UInt64Array);
impl_into_series_for_numerics!(f32, arrow::array::Float32Array);
impl_into_series_for_numerics!(f64, arrow::array::Float64Array);

macro_rules! impl_comparison_op {
    ($func_name:ident, $kernel:path, $doc:expr) => {
        #[doc = $doc]
        pub fn $func_name<T>(&self, value: T) -> Result<Series, CrossbowError>
        where
            T: arrow::datatypes::ArrowNumericType,
            T: arrow::array::Datum,
            T::Native: arrow::datatypes::ArrowNativeType,
        {
            let array = self
                .data
                .as_any()
                .downcast_ref::<arrow::array::PrimitiveArray<T>>()
                .ok_or_else(|| {
                    CrossbowError::OperationNotSupported(format!(
                        "operation '{}' not supported for dtype {:?}",
                        stringify!($func_name),
                        self.dtype()
                    ))
                })?;

            let boolean_array = $kernel(array, &value)?;

            Ok(Series::new(self.name(), std::sync::Arc::new(boolean_array)))
        }
    };
}

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

impl Series {
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
                let mut builder = arrow::array::Int32Builder::new();
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) && arr2.is_valid(i) {
                        builder.append_value(arr1.value(i).wrapping_add(arr2.value(i)));
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
                let mut builder = arrow::array::Float64Builder::new();
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
                let mut builder = arrow::array::Int32Builder::new();
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) && arr2.is_valid(i) {
                        builder.append_value(arr1.value(i).wrapping_sub(arr2.value(i)));
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
                let mut builder = arrow::array::Float64Builder::new();
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
                let mut builder = arrow::array::Int32Builder::new();
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) && arr2.is_valid(i) {
                        builder.append_value(arr1.value(i).wrapping_mul(arr2.value(i)));
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
                let mut builder = arrow::array::Float64Builder::new();
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

                let mut builder = arrow::array::Int32Builder::new();
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

                let mut builder = arrow::array::Float64Builder::new();
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

                let mut builder = arrow::array::Int32Builder::new();
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

impl Series {
    pub fn is_null(&self) -> Result<Series, CrossbowError> {
        let mut builder = arrow::array::BooleanBuilder::new();
        for i in 0..self.len() {
            builder.append_value(self.data.is_null(i));
        }
        Ok(Series::new(format!("{}_is_null", self.name()), Arc::new(builder.finish())))
    }

    pub fn is_not_null(&self) -> Result<Series, CrossbowError> {
        let mut builder = arrow::array::BooleanBuilder::new();
        for i in 0..self.len() {
            builder.append_value(!self.data.is_null(i));
        }
        Ok(Series::new(format!("{}_is_not_null", self.name()), Arc::new(builder.finish())))
    }

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
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let arr2 = fill_value.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut builder = arrow::array::Int32Builder::new();
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
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let arr2 = fill_value.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut builder = arrow::array::Float64Builder::new();
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

    pub fn drop_null(&self) -> Result<Series, CrossbowError> {
        match self.dtype() {
            DataType::Int32 => {
                let arr = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut builder = arrow::array::Int32Builder::new();
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        builder.append_value(arr.value(i));
                    }
                }
                Ok(Series::new(format!("{}_no_null", self.name()), Arc::new(builder.finish())))
            }
            DataType::Float64 => {
                let arr = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut builder = arrow::array::Float64Builder::new();
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        builder.append_value(arr.value(i));
                    }
                }
                Ok(Series::new(format!("{}_no_null", self.name()), Arc::new(builder.finish())))
            }
            DataType::Utf8 => {
                let arr = self.data.as_any().downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| CrossbowError::TypeMismatch("Expected StringArray".to_string()))?;
                let mut builder = arrow::array::StringBuilder::new();
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        builder.append_value(arr.value(i));
                    }
                }
                Ok(Series::new(format!("{}_no_null", self.name()), Arc::new(builder.finish())))
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("drop_null not supported for {:?}", self.dtype())
            )),
        }
    }

    pub fn value_at(&self, index: usize) -> Result<String, CrossbowError> {
        if index >= self.len() {
            return Err(CrossbowError::IndexOutOfBounds(index));
        }
        Ok(self.get_value_as_string(index))
    }

    pub fn slice(&self, offset: usize, length: usize) -> Result<Series, CrossbowError> {
        if offset + length > self.len() {
            return Err(CrossbowError::IndexOutOfBounds(offset + length));
        }
        let sliced = self.data.slice(offset, length);
        Ok(Series::new(format!("{}_slice", self.name()), Arc::new(sliced)))
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self.dtype(), 
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 |
            DataType::Float32 | DataType::Float64
        )
    }

    pub fn is_string(&self) -> bool {
        matches!(self.dtype(), DataType::Utf8 | DataType::LargeUtf8)
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self.dtype(), DataType::Boolean)
    }
}

impl Series {
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

    pub fn mean(&self) -> Result<f64, CrossbowError> {
        match self.dtype() {
            DataType::Int32 | DataType::Float64 => {
                let sum = self.sum()?;
                let count = self.count_non_null()?;
                if count == 0 {
                    Ok(0.0)
                } else {
                    Ok(sum / count as f64)
                }
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("mean() not supported for {:?}", self.dtype())
            )),
        }
    }

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

    pub fn count(&self) -> usize {
        self.len()
    }

    pub fn count_non_null(&self) -> Result<usize, CrossbowError> {
        let mut count = 0;
        for i in 0..self.len() {
            if self.data.is_valid(i) {
                count += 1;
            }
        }
        Ok(count)
    }

    pub fn std(&self) -> Result<f64, CrossbowError> {
        if self.count_non_null()? < 2 {
            return Ok(0.0);
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
                        let val = arr.value(i) as f64;
                        variance += (val - mean).powi(2);
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
                        let val = arr.value(i);
                        variance += (val - mean).powi(2);
                        count += 1;
                    }
                }
            }
            _ => {
                return Err(CrossbowError::OperationNotSupported(
                    format!("std() not supported for {:?}", self.dtype())
                ));
            }
        }

        if count < 2 {
            return Ok(0.0);
        }

        Ok((variance / (count - 1) as f64).sqrt())
    }

    pub fn var(&self) -> Result<f64, CrossbowError> {
        if self.count_non_null()? < 2 {
            return Ok(0.0);
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
                        let val = arr.value(i) as f64;
                        variance += (val - mean).powi(2);
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
                        let val = arr.value(i);
                        variance += (val - mean).powi(2);
                        count += 1;
                    }
                }
            }
            _ => {
                return Err(CrossbowError::OperationNotSupported(
                    format!("var() not supported for {:?}", self.dtype())
                ));
            }
        }

        if count < 2 {
            return Ok(0.0);
        }

        Ok(variance / (count - 1) as f64)
    }
}

impl std::fmt::Display for Series {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const HEAD: usize = 5;
        const TAIL: usize = 5;
        let len = self.len();

        let mut values_to_display = Vec::new();

        if len > HEAD + TAIL {
            for i in 0..HEAD {
                values_to_display.push(self.get_value_as_string(i));
            }
            for i in (len - TAIL)..len {
                values_to_display.push(self.get_value_as_string(i));
            }
        } else {
            for i in 0..len {
                values_to_display.push(self.get_value_as_string(i));
            }
        }

        let index_width = len.saturating_sub(1).to_string().len();

        let data_width = values_to_display
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
            .max(self.name().len());

        writeln!(f, "┌─{:-<index_width$}─┬─{:-<data_width$}─┐", "", "")?;
        writeln!(f, "│ {:>index_width$} │ {:^data_width$} │", "", self.name())?;
        writeln!(f, "├─{:-<index_width$}─┼─{:-<data_width$}─┤", "", "")?;

        if len > HEAD + TAIL {
            for (i, item) in values_to_display.iter().enumerate().take(HEAD) {
                writeln!(f, "│ {:>index_width$} │ {:<data_width$} │", i, item)?;
            }
            writeln!(f, "│ {:^index_width$} │ {:^data_width$} │", "...", "...")?;
            for i in 0..TAIL {
                let original_index = len - TAIL + i;
                let value_index = HEAD + i;
                writeln!(
                    f,
                    "│ {:>index_width$} │ {:<data_width$} │",
                    original_index, values_to_display[value_index]
                )?;
            }
        } else {
            for (i, item) in values_to_display.iter().enumerate().take(HEAD) {
                writeln!(f, "│ {:>index_width$} │ {:<data_width$} │", i, item)?;
            }
        }

        writeln!(f, "└─{:-<index_width$}─┴─{:-<data_width$}─┘", "", "")?;
        write!(f, "DataType: {:?}", self.dtype())?;

        Ok(())
    }
}

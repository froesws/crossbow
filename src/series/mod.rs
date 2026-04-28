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

pub mod comparison;
pub mod arithmetic;
pub mod null_utils;
pub mod aggregation;

#[cfg(test)]
mod unit_test;

/// A named, strongly-typed column of data backed by an Apache Arrow array.
///
/// `Series` is the foundational data structure in crossbow. It wraps an
/// [`ArrayRef`] from the `arrow-rs` crate, providing type-safe access
/// and a rich set of transformation, comparison, arithmetic, null-handling,
/// and aggregation operations.
#[derive(Debug, Clone)]
pub struct Series {
    name: String,
    data: ArrayRef,
}

impl Series {
    /// Creates a new `Series` from an Arrow array reference.
    ///
    /// Prefer [`Series::from`] for constructing from Rust `Vec<T>` values.
    pub fn new(name: impl Into<String>, data: ArrayRef) -> Self {
        Series {
            name: name.into(),
            data,
        }
    }

    /// Creates a `Series` from a `Vec<T>` and a column name.
    ///
    /// Convenience constructor using the [`IntoSeries`] trait. Supports all
    /// integer, float, bool, `&str`, and `String` vectors (including `Option` variants).
    pub fn from<T>(name: &str, data: Vec<T>) -> Series
    where
        T: 'static,
        Vec<T>: IntoSeries,
    {
        data.into_series(name)
    }

    /// Returns the column name of this `Series`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the underlying Arrow array.
    #[allow(dead_code)]
    pub fn data(&self) -> &ArrayRef {
        &self.data
    }

    /// Returns the Arrow data type of this column.
    pub fn dtype(&self) -> &DataType {
        self.data.data_type()
    }

    /// Returns the number of elements (including nulls).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the `Series` contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Attempts to downcast the internal array to a concrete Arrow array type.
    ///
    /// Returns `Some(&T)` on success, `None` if the type does not match.
    pub fn as_primitive<T>(&self) -> Option<&T>
    where
        T: 'static,
    {
        self.data.as_any().downcast_ref::<T>()
    }

    pub(crate) fn get_value_as_string(&self, index: usize) -> String {
        if self.data.is_null(index) {
            return "null".to_string();
        }

        match self.dtype() {
            DataType::Int32 => {
                match self.data.as_any().downcast_ref::<arrow::array::Int32Array>() {
                    Some(array) => array.value(index).to_string(),
                    None => "Invalid Int32 array".to_string(),
                }
            }
            DataType::Int64 => {
                match self.data.as_any().downcast_ref::<arrow::array::Int64Array>() {
                    Some(array) => array.value(index).to_string(),
                    None => "Invalid Int64 array".to_string(),
                }
            }
            DataType::Float32 => {
                match self.data.as_any().downcast_ref::<arrow::array::Float32Array>() {
                    Some(array) => array.value(index).to_string(),
                    None => "Invalid Float32 array".to_string(),
                }
            }
            DataType::Float64 => {
                match self.data.as_any().downcast_ref::<arrow::array::Float64Array>() {
                    Some(array) => array.value(index).to_string(),
                    None => "Invalid Float64 array".to_string(),
                }
            }
            DataType::Utf8 => {
                match self.data.as_any().downcast_ref::<arrow::array::StringArray>() {
                    Some(array) => format!("\"{}\"", array.value(index)),
                    None => "Invalid String array".to_string(),
                }
            }
            DataType::Boolean => {
                match self.data.as_any().downcast_ref::<arrow::array::BooleanArray>() {
                    Some(array) => array.value(index).to_string(),
                    None => "Invalid Boolean array".to_string(),
                }
            }
            other_type => format!("Unsupported type: {:?}", other_type),
        }
    }
}

/// Trait for converting a `Vec<T>` into a named `Series`.
///
/// Implemented for all common Rust types (`i8`-`i64`, `u8`-`u64`, `f32`, `f64`,
/// `bool`, `&str`, `String`) and their `Option` variants. Uses the
/// [`impl_into_series_for_numerics!`] macro for numeric types.
pub trait IntoSeries {
    /// Consumes the vector, returning a `Series` with the given name.
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

#[allow(unused_macros)]
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

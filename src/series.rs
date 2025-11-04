//! Series module
//!
//! A series is a named, strongly-typed column of data.
//!
//! MIT License
//! ... (copyright)

use crate::error::CrossbowError;
use crate::types::DataType;
use crate::types::SeriesData;

#[cfg(test)]
mod unit_tests;

// This macro will handle the core filter logic.
// It takes the data vector and the boolean mask slice.
// Zips them up, and only keeps items where mask is Some(true).
macro_rules! apply_boolean_mask {
    ($data_vec:expr, $mask_slice:expr) => {{
        $data_vec
            .iter()
            .zip($mask_slice.iter())
            .filter_map(|(data_opt, mask_opt)| {
                // Only keep if mask is explicitly true.
                // false or null masks mean we drop the value.
                match mask_opt {
                    Some(true) => Some(data_opt.clone()),
                    _ => None,
                }
            })
            .collect()
    }};
}

/// Represents a named, strongly-typed column of data.
///
/// This is the primary unit of data. It wraps an internal
/// data container to provide a safe, unified API.
#[derive(Debug, Clone)]
pub struct Series {
    name: String,
    data: SeriesData,
}

impl Series {
    // Internal constructors.
    // These are called by the `IntoSeries` trait implementations.
    pub(crate) fn new_int32(name: impl Into<String>, data: Vec<Option<i32>>) -> Self {
        Self {
            name: name.into(),
            data: SeriesData::Int32(data),
        }
    }
    pub(crate) fn new_int64(name: impl Into<String>, data: Vec<Option<i64>>) -> Self {
        Self {
            name: name.into(),
            data: SeriesData::Int64(data),
        }
    }
    pub(crate) fn new_f64(name: impl Into<String>, data: Vec<Option<f64>>) -> Self {
        Self {
            name: name.into(),
            data: SeriesData::Float64(data),
        }
    }
    pub(crate) fn new_bool(name: impl Into<String>, data: Vec<Option<bool>>) -> Self {
        Self {
            name: name.into(),
            data: SeriesData::Boolean(data),
        }
    }
    pub(crate) fn new_string(name: impl Into<String>, data: Vec<Option<String>>) -> Self {
        Self {
            name: name.into(),
            data: SeriesData::String(data),
        }
    }

    /// Creates a new column from a name and a collection of values.
    /// This is the main public constructor.
    pub fn from<T>(name: &str, data: Vec<T>) -> Series
    where
        T: 'static,
        Vec<T>: IntoSeries,
    {
        // Delegates creation to the trait implementation.
        data.into_series(name)
    }

    /// Returns the name of the column.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the logical data type of the column.
    pub fn dtype(&self) -> DataType {
        match &self.data {
            SeriesData::Int32(_) => DataType::Int32,
            SeriesData::Int64(_) => DataType::Int64,
            SeriesData::Float64(_) => DataType::Float64,
            SeriesData::Boolean(_) => DataType::Boolean,
            SeriesData::String(_) => DataType::String,
        }
    }

    /// Returns the number of elements (rows) in the column.
    pub fn len(&self) -> usize {
        match &self.data {
            SeriesData::Int32(v) => v.len(),
            SeriesData::Int64(v) => v.len(),
            SeriesData::Float64(v) => v.len(),
            SeriesData::Boolean(v) => v.len(),
            SeriesData::String(v) => v.len(),
        }
    }

    /// Checks if the column contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Checks if the value at a specific index is null.
    pub fn is_null(&self, index: usize) -> bool {
        // Panics on out-of-bounds, which is fine.
        match &self.data {
            SeriesData::Int32(v) => v[index].is_none(),
            SeriesData::Int64(v) => v[index].is_none(),
            SeriesData::Float64(v) => v[index].is_none(),
            SeriesData::Boolean(v) => v[index].is_none(),
            SeriesData::String(v) => v[index].is_none(),
        }
    }

    /// Performs a 'greater than' comparison against a scalar value.
    ///
    /// This creates a new boolean mask. Nulls in the original
    /// data will result in nulls in the mask.
    pub fn gt_i64(&self, value: i64) -> Result<Series, CrossbowError> {
        // This operation will be a closure.
        let op = |a: &i64| a > &value;

        // Need to dispatch based on the internal data.
        let new_data = match &self.data {
            SeriesData::Int64(v) => {
                // OK, map over the vector.
                // If `Some(val)`, apply the op. If `None`, keep it `None`.
                let mask: Vec<Option<bool>> =
                    v.iter().map(|opt_val| opt_val.as_ref().map(op)).collect();
                Ok(mask)
            }
            SeriesData::Float64(v) => {
                // Also good to allow comparing an int with a float column.
                let f_value = value as f64;
                let op_f = |a: &f64| a > &f_value;
                let mask: Vec<Option<bool>> =
                    v.iter().map(|opt_val| opt_val.as_ref().map(op_f)).collect();
                Ok(mask)
            }
            // Other types aren't comparable to an i64 this way.
            _ => Err(CrossbowError::OperationNotSupported(format!(
                "Operation 'gt_i64' not supported for dtype {:?}",
                self.dtype()
            ))),
        }?;

        // Cool, now wrap this boolean vector in a new Series.
        Ok(Series::new_bool(self.name(), new_data))
    }

    // TODO: Add lt_i64, eq_i64, gt_f64, eq_str, etc. following this pattern.

    /// Creates a new object containing only the values
    /// corresponding to `true` entries in the mask.
    pub fn filter(&self, mask: &Series) -> Result<Series, CrossbowError> {
        // First, validate the mask.
        // It must be boolean.
        let mask_slice = match &mask.data {
            SeriesData::Boolean(v) => v.as_slice(),
            _ => {
                return Err(CrossbowError::OperationNotSupported(
                    "Filter mask must be of type Boolean".to_string(),
                ));
            }
        };

        // And it must be the same length.
        if self.len() != mask.len() {
            // Just reuse the existing error for this.
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        // OK, dispatch to the macro for each data type.
        let new_data = match &self.data {
            SeriesData::Int32(v) => {
                let new_vec: Vec<Option<i32>> = apply_boolean_mask!(v, mask_slice);
                SeriesData::Int32(new_vec)
            }
            SeriesData::Int64(v) => {
                let new_vec: Vec<Option<i64>> = apply_boolean_mask!(v, mask_slice);
                SeriesData::Int64(new_vec)
            }
            SeriesData::Float64(v) => {
                let new_vec: Vec<Option<f64>> = apply_boolean_mask!(v, mask_slice);
                SeriesData::Float64(new_vec)
            }
            SeriesData::Boolean(v) => {
                // Yep, can filter a boolean series too.
                let new_vec: Vec<Option<bool>> = apply_boolean_mask!(v, mask_slice);
                SeriesData::Boolean(new_vec)
            }
            SeriesData::String(v) => {
                let new_vec: Vec<Option<String>> = apply_boolean_mask!(v, mask_slice);
                SeriesData::String(new_vec)
            }
        };

        // Wrap the new data in a Series.
        Ok(Series {
            name: self.name.clone(),
            data: new_data,
        })
    }

    // Removed the `get_value_as_string` helper function.
}

// Removed the `impl std::fmt::Display for Series` block.

/// A trait for converting collections into a Series.
///
/// This allows `Series::from` to be generic.
pub trait IntoSeries {
    /// Converts the collection into a Series.
    fn into_series(self, name: &str) -> Series;
}

// --- String/&str Implementations ---
// We need to handle Vec<&str>, Vec<Option<&str>>, and Vec<String>

impl IntoSeries for Vec<&str> {
    fn into_series(self, name: &str) -> Series {
        let data: Vec<Option<String>> = self.into_iter().map(|s| Some(s.to_string())).collect();
        Series::new_string(name, data)
    }
}

impl IntoSeries for Vec<Option<&str>> {
    fn into_series(self, name: &str) -> Series {
        let data: Vec<Option<String>> = self
            .into_iter()
            .map(|opt_s| opt_s.map(|s| s.to_string()))
            .collect();
        Series::new_string(name, data)
    }
}
impl IntoSeries for Vec<String> {
    fn into_series(self, name: &str) -> Series {
        let data: Vec<Option<String>> = self.into_iter().map(Some).collect();
        Series::new_string(name, data)
    }
}

/// This macro implements `IntoSeries` for numeric types.
/// It handles both `Vec<T>` and `Vec<Option<T>>`.
macro_rules! impl_into_series_for_numerics {
    // $T = Rust type (e.g., i32)
    // $C = Constructor function (e.g., new_int32)
    ($T:ty, $C:ident) => {
        // Implementation for Vec<T> (no nulls)
        impl IntoSeries for Vec<$T> {
            fn into_series(self, name: &str) -> Series {
                // Convert non-nullable vec into a vec of `Some(T)`
                let data: Vec<Option<$T>> = self.into_iter().map(Some).collect();
                Series::$C(name, data)
            }
        }
        // Implementation for Vec<Option<T>> (with nulls)
        impl IntoSeries for Vec<Option<$T>> {
            fn into_series(self, name: &str) -> Series {
                // This one is a direct pass-through
                Series::$C(name, self)
            }
        }
    };
}

// Register the types we want to support.
impl_into_series_for_numerics!(bool, new_bool);
impl_into_series_for_numerics!(i32, new_int32);
impl_into_series_for_numerics!(i64, new_int64);
impl_into_series_for_numerics!(f64, new_f64);

impl Series {
    #[cfg(test)] // Only compile this for tests
    pub(crate) fn as_i32_slice(&self) -> Option<&[Option<i32>]> {
        match &self.data {
            SeriesData::Int32(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    #[cfg(test)] // Only compile this for tests
    pub(crate) fn as_i64_slice(&self) -> Option<&[Option<i64>]> {
        match &self.data {
            SeriesData::Int64(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    #[cfg(test)] // Only compile this for tests
    pub(crate) fn as_f64_slice(&self) -> Option<&[Option<f64>]> {
        match &self.data {
            SeriesData::Float64(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    #[cfg(test)] // Only compile this for tests
    pub(crate) fn as_bool_slice(&self) -> Option<&[Option<bool>]> {
        match &self.data {
            SeriesData::Boolean(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    #[cfg(test)] // Only compile this for tests
    pub(crate) fn as_string_slice(&self) -> Option<&[Option<String>]> {
        match &self.data {
            SeriesData::String(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

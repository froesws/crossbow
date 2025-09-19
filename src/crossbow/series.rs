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
//! MIT License
//!
//! Copyright (c) [2025] [William Froes]
//! Project: [crossbow]
//! Developed at: [Uergs -- Universidade Estadual do Rio Grande do Sul]
use arrow::array::{Array, ArrayRef};
use arrow::datatypes::DataType;
use std::sync::Arc;

#[cfg(test)]
mod unit_test;

/// Represents a named, strongly-typed column of data.
///
/// A `Series` is the primary unit of data in this crate. It is a wrapper
/// around an [`ArrayRef`] from the `arrow-rs` crate, which allows it to
/// efficiently hold data of different types (like `i32`, `&str`, etc.).
///
/// # Examples
///
/// The easiest way to create a `Series` is by using the [`Series::from`] method.
///
/// ```
/// # use crossbow::Series; // The # hides this line from the final output but is needed for the test
/// let s = Series::from("numbers", vec![1, 2, 3]);
/// assert_eq!(s.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct Series {
    name: String,
    data: ArrayRef,
}

impl Series {
    /// Creates a new Series with the given name and data.
    ///
    /// # Arguments
    /// * `name` - The name of the Series.
    /// * `data` - An [`ArrayRef`] containing the data for the [`Series`].
    ///
    /// # Returns
    /// A new instance of [`Series`].
    ///
    /// # Example
    /// ```
    /// use arrow::array::Int32Array;
    /// use std::sync::Arc;
    /// use crossbow::Series;
    ///
    /// let int_data = Int32Array::from(vec![Some(1), Some(2), Some(3)]);
    /// let series = Series::new("numbers", Arc::new(int_data));
    /// assert_eq!(series.name(), "numbers");
    /// assert_eq!(series.len(), 3);
    /// assert_eq!(series.dtype(), &arrow::datatypes::DataType::Int32);
    /// ```
    pub fn new(name: impl Into<String>, data: ArrayRef) -> Self {
        Series {
            name: name.into(),
            data,
        }
    }

    /// Creates a Series from a vector of values and a name.
    /// This is a convenience method that leverages the `IntoSeries` trait
    /// to convert the vector into an appropriate Arrow array.
    ///
    /// # Arguments
    /// * `data` - A vector of values to be converted into a Series.
    /// * `name` - The name of the Series.
    ///
    /// # Returns
    /// A Series containing the data from the vector.
    ///
    /// # Example
    /// ```
    /// use crossbow::Series;
    ///
    /// let series = Series::from(vec![1, 2, 3], "numbers");
    /// assert_eq!(series.name(), "numbers");
    /// assert_eq!(series.len(), 3);
    /// assert_eq!(series.dtype(), &arrow::datatypes::DataType::Int32 );
    /// ```
    /// You can also use it with Option types:
    /// ```
    /// use crossbow::Series;
    ///
    /// let series_opt = Series::from(vec![Some(1), None, Some(3)], "optional_numbers");
    /// assert_eq!(series_opt.name(), "optional_numbers");
    /// assert_eq!(series_opt.len(), 3);
    /// assert_eq!(series_opt.dtype(), &arrow::datatypes::DataType::Int32);
    /// ```
    pub fn from<T>(data: Vec<T>, name: &str) -> Series
    where
        T: 'static,
        Vec<T>: IntoSeries,
    {
        data.into_series(name)
    }

    /// Returns the name of this [`Series`].
    ///
    /// # Returns
    /// A [`&str`] slice representing the name of the Series.
    ///
    /// # Example
    /// ```
    /// use crossbow::Series;
    ///
    /// let series = Series::from(vec![1, 2, 3], "numbers");
    /// assert_eq!(series.name(), "numbers");
    /// ```
    ///
    /// # Note
    /// The name is stored as a `String` internally, but this method returns a `&str` for convenience.
    /// This avoids unnecessary cloning of the name when you just need to read it.
    /// If you need ownership of the name, you can always clone it using `series.name().to_string()`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the data of this [`Series`].
    ///
    /// # Returns
    /// A reference to the underlying [`ArrayRef`].
    ///
    /// # Example
    /// ```
    /// use arrow::array::Int32Array;
    /// use std::sync::Arc;
    /// use crossbow::Series;
    ///
    /// let int_data = Int32Array::from(vec![Some(1), Some(
    /// 2), Some(3)]);
    /// let series = Series::new("numbers", Arc::new(int_data.clone()));
    /// assert_eq!(series.data().as_ref(), &int_data);
    /// ```
    /// # Note
    /// This method provides direct access to the underlying Arrow array.
    /// Be cautious when using it, as modifying the array directly can lead to inconsistencies
    /// with the Series' metadata (like name and length).
    /// In most cases, you should use the other methods provided by the Series struct
    #[allow(dead_code)]
    pub fn data(&self) -> &ArrayRef {
        &self.data
    }

    /// Returns the data type of the underlying Arrow array.
    ///
    /// # Returns
    /// A reference to the [`DataType`] of the Series.
    ///
    /// # Example
    /// ```
    /// use crossbow::Series;
    /// use arrow::datatypes::DataType;
    ///
    /// let series = Series::from(vec![1, 2, 3], "numbers");
    /// assert_eq!(series.dtype(), &DataType::Int32);
    /// ```
    ///
    /// # Note
    /// The data type is derived from the underlying Arrow array.
    /// This method is useful for understanding the type of data stored in the Series,
    /// especially when working with heterogeneous data.
    pub fn dtype(&self) -> &DataType {
        self.data.data_type()
    }

    /// Returns the length of the Series.   
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Attempts to downcast the internal data to a specific Arrow array type.
    ///
    /// # Type Parameters
    /// * `T`: The target Arrow array type to downcast to (e.g.,`Int32Array`, `Float64Array`, etc.).
    ///
    /// # Returns
    /// * `Some(&T)` if the downcast is successful, otherwise `None`.
    ///
    /// # Example
    /// ```
    /// let series = Series::from(vec![1, 2, 3], "my_series");
    /// if let Some(int_array) = series.as_primitive::<arrow::array::Int32Array>() {
    ///     // Do something with the downcasted array
    /// }
    /// ```
    pub fn as_primitive<T>(&self) -> Option<&T>
    where
        T: 'static,
    {
        self.data.as_any().downcast_ref::<T>()
    }

    /// Helper function to get a string representation of a value at a given index.
    /// This is used for displaying the Series.
    ///
    /// # Arguments
    /// * `index` - The index of the value to retrieve.
    ///
    /// # Returns
    /// A string representation of the value at the given index.
    /// If the value is null, returns "null".
    /// If the type is unsupported, returns a message indicating so.
    ///
    /// # Example
    /// ```
    /// let value_str = series.get_value_as_string(0);
    /// println!("Value at index 0: {}", value_str);
    /// ```
    /// # Panics
    /// Panics if the index is out of bounds.
    ///
    /// # Note
    /// This method currently supports Int32, Float64, Utf8 (String), and Boolean types.
    /// You can extend it to support more types as needed.  
    fn get_value_as_string(&self, index: usize) -> String {
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

impl std::fmt::Display for Series {
    /// Custom display implementation for Series.
    /// Displays the Series in a tabular format with index and values.
    /// If the Series is longer than 10 elements, it shows the first 5 and last 5 elements.
    /// Also displays the data type at the bottom.
    ///
    /// # Example
    /// ```
    /// let series = Series::from(vec![1, 2, 3, 4, 5], "numbers");
    /// println!("{}", series);
    /// ```
    ///
    /// # Note
    /// This implementation currently supports Int32, Float64, Utf8 (String), and Boolean types.
    /// You can extend it to support more types as needed.
    ///
    /// # Panics
    /// Panics if the internal data cannot be downcast to the expected types.
    /// This should not happen if the Series is constructed correctly.  
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

pub trait IntoSeries {
    /// Converts a vector of values into a Series with the given name.
    /// This is a convenience method for creating a Series from a vector.
    ///
    /// # Arguments
    /// * `name` - The name of the Series.
    /// # Returns
    /// A Series containing the data from the vector.
    ///
    /// # Example
    /// ```
    /// let series = vec![1, 2, 3].into_series("numbers");
    /// assert_eq!(series.name(), "numbers");
    /// assert_eq!(series.len(), 3);
    /// assert_eq!(series.dtype(), &DataType::Int32);
    /// ```
    /// You can also use it with Option types:
    /// ```
    /// let series_opt = vec![Some(1), None, Some(3)].into_series("optional_numbers");
    /// assert_eq!(series_opt.name(), "optional_numbers");
    /// assert_eq!(series_opt.len(), 3);
    /// assert_eq!(series_opt.dtype(), &DataType::Int32);
    /// ```
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
// Você pode adicionar para String também se precisar
impl IntoSeries for Vec<String> {
    fn into_series(self, name: &str) -> Series {
        let array =
            arrow::array::StringArray::from(self.iter().map(|s| s.as_str()).collect::<Vec<&str>>());
        Series::new(name, Arc::new(array))
    }
}

/// This macro helps to implement the `IntoSeries` trait for numeric types.
/// It generates implementations for both `Vec<T>` and `Vec<Option<T>>` where `T` is a numeric type.
/// # Arguments
/// * `$T`: The Rust primitive type (e.g., `i32`, `f64`, etc.).
/// * `$A`: The corresponding Arrow array type (e.g., `Int32Array`, `Float64Array`, etc.).
///
/// # Example
/// ```
/// let series = vec![1, 2, 3].into_series("numbers");
/// assert_eq!(series.name(), "numbers");
/// assert_eq!(series.len(), 3);
/// assert_eq!(series.dtype(), &DataType::Int32);
/// ```
///
/// You can also use it with Option types:
/// ```
/// let series_opt = vec![Some(1), None, Some(3)].into_series("optional_numbers");
/// assert_eq!(series_opt.name(), "optional_numbers");
/// assert_eq!(series_opt.len(), 3);
/// assert_eq!(series_opt.dtype(), &DataType::Int32);
/// ```
macro_rules! impl_into_series_for_numerics {
    ($T:ty, $A:ty) => {
        // Implementação para Vec<T> (sem nulos)
        impl IntoSeries for Vec<$T> {
            fn into_series(self, name: &str) -> Series {
                let array = <$A>::from(self);
                Series::new(name, Arc::new(array))
            }
        }
        // Implementação para Vec<Option<T>> (com nulos)
        impl IntoSeries for Vec<Option<$T>> {
            fn into_series(self, name: &str) -> Series {
                let array = <$A>::from(self);
                Series::new(name, Arc::new(array))
            }
        }
    };
}

impl_into_series_for_numerics!(i32, arrow::array::Int32Array);
impl_into_series_for_numerics!(f64, arrow::array::Float64Array);
impl_into_series_for_numerics!(bool, arrow::array::BooleanArray);

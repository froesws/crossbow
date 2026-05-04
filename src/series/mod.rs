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
//! Copyright (c) \[2025\] \[William Froes]
//! Project: \[crossbow\]
//! Developed at: \[Uergs -- Universidade Estadual do Rio Grande do Sul\]
use crate::CrossbowError;
use arrow::array::{Array, ArrayRef};
use arrow::datatypes::DataType;
use std::sync::Arc;

macro_rules! impl_comparison_op {
    ($func_name:ident, $kernel:path, $doc:expr) => {
        #[doc = $doc]
        pub fn $func_name(&self, value: impl arrow::array::Datum) -> Result<Series, CrossbowError> {
            let lhs: &dyn arrow::array::Array = self.data.as_ref();
            let boolean_array = $kernel(&lhs, &value)?;
            Ok(Series::new(self.name(), std::sync::Arc::new(boolean_array)))
        }
    };
}

pub mod aggregation;
pub mod arithmetic;
pub mod comparison;
pub mod null_utils;

#[cfg(test)]
mod unit_test;

/// A named, strongly-typed column of data backed by an Apache Arrow array.
///
/// `Series` is the foundational data structure in crossbow. It wraps an
/// [`ArrayRef`] from the `arrow-rs` crate, providing type-safe access
/// and a rich set of transformation, comparison, arithmetic, null-handling,
/// and aggregation operations.
///
/// # Examples
///
/// ```
/// use crossbow::Series;
///
/// let s = Series::from("age", vec![25i32, 30, 35]);
/// assert_eq!(s.len(), 3);
/// assert_eq!(s.name(), "age");
/// ```
#[derive(Debug, Clone)]
pub struct Series {
    name: String,
    data: ArrayRef,
    pub(crate) date_format: Option<String>,
}

/// Converts days since Unix epoch (1970-01-01) to year, month, day.
/// Uses O(1) algorithm (Howard Hinnant / civil_from_days).
pub(crate) fn epoch_days_to_ymd(epoch_days: i64) -> (i32, u32, u32) {
    let z = epoch_days + 719468i64;
    let era = if z >= 0 {
        z / 146097
    } else {
        (z - 146096) / 146097
    };
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d as u32)
}

/// Converts (year, month, day) back to days since Unix epoch.
pub(crate) fn ymd_to_epoch_days(year: i64, month: u32, day: u32) -> i32 {
    let (era, yoe): (i64, i64) = if month > 2 {
        (year.div_euclid(400), year.rem_euclid(400))
    } else {
        let y = year - 1;
        (y.div_euclid(400), y.rem_euclid(400))
    };
    let doy: i64 =
        (153 * (if month > 2 {
            month as i64 - 3
        } else {
            month as i64 + 9
        }) + 2)
            / 5
            + day as i64
            - 1;
    let doe: i64 = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days: i64 = era * 146097 + doe - 719468;
    days as i32
}

fn format_date(ymd: (i32, u32, u32), ms_of_day: u32, fmt: &str) -> String {
    let (y, m, d) = ymd;
    let h = ms_of_day / 3_600_000;
    let min = (ms_of_day % 3_600_000) / 60_000;
    let s = (ms_of_day % 60_000) / 1_000;
    fmt.replace("YYYY", &format!("{:04}", y))
        .replace("YY", &format!("{:02}", y % 100))
        .replace("MM", &format!("{:02}", m))
        .replace("DD", &format!("{:02}", d))
        .replace("HH", &format!("{:02}", h))
        .replace("mm", &format!("{:02}", min))
        .replace("ss", &format!("{:02}", s))
}

fn date64_body(index: usize, data: &ArrayRef, date_format: &Option<String>) -> String {
    let arr = data
        .as_any()
        .downcast_ref::<arrow::array::Date64Array>()
        .expect("Date64 dtype mismatch");
    let ms = arr.value(index as usize);
    let days = ms.div_euclid(86_400_000);
    let ms_of_day = ms.rem_euclid(86_400_000) as u32;
    let ymd = epoch_days_to_ymd(days);
    let fmt = date_format.as_deref().unwrap_or("YYYY-MM-DD HH:mm:ss");
    format_date(ymd, ms_of_day, fmt)
}

fn timestamp_ms_body(index: usize, data: &ArrayRef, date_format: &Option<String>) -> String {
    let arr = data
        .as_any()
        .downcast_ref::<arrow::array::TimestampMillisecondArray>()
        .expect("TimestampMillisecond dtype mismatch");
    let ms = arr.value(index as usize);
    let days = ms.div_euclid(86_400_000);
    let ms_of_day = ms.rem_euclid(86_400_000) as u32;
    let ymd = epoch_days_to_ymd(days);
    let fmt = date_format.as_deref().unwrap_or("YYYY-MM-DD HH:mm:ss");
    format_date(ymd, ms_of_day, fmt)
}

macro_rules! simple_value_string {
    ($slf:expr, $idx:expr, $array_ty:ty, $label:expr) => {
        match $slf.data.as_any().downcast_ref::<$array_ty>() {
            Some(array) => array.value($idx).to_string(),
            None => concat!("Invalid ", $label, " array").to_string(),
        }
    };
}

impl Series {
    /// Creates a new `Series` from an Arrow array reference.
    ///
    /// Prefer [`Series::from`] for constructing from Rust `Vec<T>` values.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![1i32, 2, 3]);
    /// assert_eq!(s.len(), 3);
    /// ```
    pub fn new(name: impl Into<String>, data: ArrayRef) -> Self {
        Series {
            name: name.into(),
            data,
            date_format: None,
        }
    }

    /// Creates a `Series` from a `Vec<T>` and a column name.
    ///
    /// Convenience constructor using the [`IntoSeries`] trait. Supports all
    /// integer, float, bool, `&str`, and `String` vectors (including `Option` variants).
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("score", vec![10.0, 20.5, 15.0]);
    /// assert_eq!(s.len(), 3);
    /// ```
    pub fn from<T>(name: &str, data: Vec<T>) -> Series
    where
        T: 'static,
        Vec<T>: IntoSeries,
    {
        data.into_series(name)
    }

    /// Creates a `Date32` series from days since Unix epoch (1970-01-01).
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let dates = Series::from_date32("epoch_days", vec![19701, 19702, 19703]);
    /// assert_eq!(dates.len(), 3);
    /// ```
    pub fn from_date32(name: &str, days_since_epoch: Vec<i32>) -> Series {
        Series {
            name: name.to_string(),
            data: Arc::new(arrow::array::Date32Array::from(days_since_epoch)),
            date_format: None,
        }
    }

    /// Creates a `Date64` series from milliseconds since Unix epoch.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let dates = Series::from_date64("epoch_ms", vec![1700000000000i64, 1700000086400000]);
    /// assert_eq!(dates.len(), 2);
    /// ```
    pub fn from_date64(name: &str, ms_since_epoch: Vec<i64>) -> Series {
        Series {
            name: name.to_string(),
            data: Arc::new(arrow::array::Date64Array::from(ms_since_epoch)),
            date_format: None,
        }
    }

    /// Creates a timestamp-without-timezone (`Timestamp(Millisecond)`) series.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let ts = Series::from_timestamp_ms("stamps", vec![1700000000000i64, 1700000086400000]);
    /// assert_eq!(ts.len(), 2);
    /// ```
    pub fn from_timestamp_ms(name: &str, ms_since_epoch: Vec<i64>) -> Series {
        Series {
            name: name.to_string(),
            data: Arc::new(arrow::array::TimestampMillisecondArray::from(
                ms_since_epoch,
            )),
            date_format: None,
        }
    }

    /// Returns the column name of this `Series`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("temperature", vec![22.5, 23.0]);
    /// assert_eq!(s.name(), "temperature");
    /// ```
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the underlying Arrow array.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    /// use arrow::datatypes::DataType;
    ///
    /// let s = Series::from("x", vec![1i32, 2]);
    /// assert_eq!(*s.dtype(), DataType::Int32);
    /// ```
    #[allow(dead_code)]
    pub fn data(&self) -> &ArrayRef {
        &self.data
    }

    /// Returns the Arrow data type of this column.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    /// use arrow::datatypes::DataType;
    ///
    /// let s = Series::from("val", vec![1.0, 2.0]);
    /// assert_eq!(*s.dtype(), DataType::Float64);
    /// ```
    pub fn dtype(&self) -> &DataType {
        self.data.data_type()
    }

    /// Returns the number of elements (including nulls).
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![1i32, 2, 3, 4, 5]);
    /// assert_eq!(s.len(), 5);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the `Series` contains zero elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    /// use std::sync::Arc;
    /// use arrow::array::Int32Array;
    ///
    /// let s = Series::new("empty", Arc::new(Int32Array::from(Vec::<i32>::new())));
    /// assert!(s.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Attempts to downcast the internal array to a concrete Arrow array type.
    ///
    /// Returns `Some(&T)` on success, `None` if the type does not match.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let s = Series::from("x", vec![1i32, 2, 3]);
    /// let arr = s.as_primitive::<arrow::array::Int32Array>().unwrap();
    /// assert_eq!(arr.value(0), 1);
    /// ```
    pub fn as_primitive<T>(&self) -> Option<&T>
    where
        T: 'static,
    {
        self.data.as_any().downcast_ref::<T>()
    }

    /// Sets a custom date format for display. Returns a new `Series`.
    ///
    /// Tokens: `YYYY` (year), `YY` (2-digit year), `MM` (month), `DD` (day),
    /// `HH` (24h hour), `mm` (minute), `ss` (second).
    ///
    /// Default: `YYYY-MM-DD` for Date32, `YYYY-MM-DD HH:mm:ss` for
    /// Date64 and Timestamp.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let dates = Series::from_date32("d", vec![19701])
    ///     .with_date_format("DD/MM/YYYY");
    /// assert_eq!(dates.value_at(0).unwrap(), "10/12/2023");
    /// ```
    pub fn with_date_format(&self, format: &str) -> Self {
        Series {
            name: self.name.clone(),
            data: self.data.clone(),
            date_format: Some(format.to_string()),
        }
    }

    /// Returns `true` if the `Series` holds date or timestamp data.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let d = Series::from_date32("d", vec![19701]);
    /// assert!(d.is_date());
    /// ```
    pub fn is_date(&self) -> bool {
        matches!(
            self.dtype(),
            DataType::Date32
                | DataType::Date64
                | DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None)
        )
    }

    /// Parses a `Utf8` string column into a `Date32` column using the
    /// given format. Supported tokens: `YYYY`, `MM`, `DD`. Returns an
    /// error if any value cannot be parsed.
    ///
    /// Primarily useful for converting date strings loaded from CSV.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::Series;
    ///
    /// let strings = Series::from("dates", vec!["2023-12-10", "2024-01-15"]);
    /// let dates = strings.try_into_date32("YYYY-MM-DD").unwrap();
    /// assert_eq!(dates.len(), 2);
    /// assert!(dates.is_date());
    /// assert_eq!(dates.value_at(0).unwrap(), "2023-12-10");
    /// ```
    pub fn try_into_date32(&self, date_format: &str) -> Result<Series, CrossbowError> {
        let arr = self
            .data
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or_else(|| {
                CrossbowError::TypeMismatch("try_into_date32 requires a Utf8 column".to_string())
            })?;

        let y_pos = date_format
            .find("YYYY")
            .ok_or_else(|| CrossbowError::InvalidSchema("format must contain YYYY".to_string()))?;
        let m_pos = date_format
            .find("MM")
            .ok_or_else(|| CrossbowError::InvalidSchema("format must contain MM".to_string()))?;
        let d_pos = date_format
            .find("DD")
            .ok_or_else(|| CrossbowError::InvalidSchema("format must contain DD".to_string()))?;

        let mut builder = arrow::array::Date32Builder::with_capacity(self.len());
        for i in 0..self.len() {
            if arr.is_null(i) {
                builder.append_null();
                continue;
            }
            let s = arr.value(i);
            let year: i32 = s[y_pos..y_pos + 4]
                .parse()
                .map_err(|_| CrossbowError::TypeMismatch(format!("invalid year in '{}'", s)))?;
            let month: u32 = s[m_pos..m_pos + 2]
                .parse()
                .map_err(|_| CrossbowError::TypeMismatch(format!("invalid month in '{}'", s)))?;
            let day: u32 = s[d_pos..d_pos + 2]
                .parse()
                .map_err(|_| CrossbowError::TypeMismatch(format!("invalid day in '{}'", s)))?;

            let epoch_days = ymd_to_epoch_days(year as i64, month, day);
            builder.append_value(epoch_days);
        }
        Ok(Series::new(
            format!("{}_date32", self.name()),
            Arc::new(builder.finish()),
        ))
    }

    pub(crate) fn get_value_as_string(&self, index: usize) -> String {
        if self.data.is_null(index) {
            return "null".to_string();
        }

        match self.dtype() {
            DataType::Int32 => simple_value_string!(self, index, arrow::array::Int32Array, "Int32"),
            DataType::Int64 => simple_value_string!(self, index, arrow::array::Int64Array, "Int64"),
            DataType::Float32 => {
                simple_value_string!(self, index, arrow::array::Float32Array, "Float32")
            }
            DataType::Float64 => {
                simple_value_string!(self, index, arrow::array::Float64Array, "Float64")
            }
            DataType::Utf8 => {
                match self
                    .data
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                {
                    Some(array) => format!("\"{}\"", array.value(index)),
                    None => "Invalid String array".to_string(),
                }
            }
            DataType::Boolean => {
                simple_value_string!(self, index, arrow::array::BooleanArray, "Boolean")
            }
            DataType::Date32 => {
                let arr = self
                    .data
                    .as_any()
                    .downcast_ref::<arrow::array::Date32Array>()
                    .expect("Date32 dtype mismatch");
                let days = arr.value(index) as i64;
                let ymd = epoch_days_to_ymd(days);
                let fmt = self.date_format.as_deref().unwrap_or("YYYY-MM-DD");
                format_date(ymd, 0, fmt)
            }
            DataType::Date64 => date64_body(index, &self.data, &self.date_format),
            DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None) => {
                timestamp_ms_body(index, &self.data, &self.date_format)
            }
            other_type => format!("Unsupported type: {:?}", other_type),
        }
    }
}

/// Trait for converting a `Vec<T>` into a named `Series`.
///
/// Implemented for all common Rust types (`i8`-`i64`, `u8`-`u64`, `f32`, `f64`,
/// `bool`, `&str`, `String`) and their `Option` variants. Uses the
/// `impl_into_series_for_numerics!` macro for numeric types.
///
/// # Examples
///
/// ```
/// use crossbow::IntoSeries;
///
/// let s = vec![1i32, 2, 3].into_series("nums");
/// assert_eq!(s.name(), "nums");
/// assert_eq!(s.len(), 3);
/// ```
pub trait IntoSeries {
    /// Consumes the vector, returning a `Series` with the given name.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::IntoSeries;
    ///
    /// let s = vec![1.0, 2.0, 3.0].into_series("values");
    /// assert_eq!(s.len(), 3);
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

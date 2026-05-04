//! Ergonomic typed-scalar constructors for comparison operations.
//!
//! Use [`Numeric`] to create correctly-typed scalar values for use with
//! [`crate::Series`] comparison methods like [`crate::Series::greater_than`].
//!
//! # Examples
//!
//! ```
//! use crossbow::{Series, Numeric};
//!
//! let s = Series::from("age", vec![18i32, 25, 42]);
//! let adult = s.greater_than(Numeric::int32(21)).unwrap();
//! ```

/// Typed-scalar factory for building Arrow scalars from Rust primitives.
pub struct Numeric;

impl Numeric {
    /// Creates an `Int32` scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{Series, Numeric};
    ///
    /// let s = Series::from("x", vec![10i32, 20, 30]);
    /// let mask = s.greater_than(Numeric::int32(15)).unwrap();
    /// ```
    pub fn int32(v: i32) -> arrow::array::Scalar<arrow::array::Int32Array> {
        arrow::array::Scalar::new(arrow::array::Int32Array::from(vec![v]))
    }

    /// Creates an `Int64` scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{Series, Numeric};
    ///
    /// let s = Series::from("x", vec![100i64, 200, 300]);
    /// let mask = s.less_than(Numeric::int64(250)).unwrap();
    /// ```
    pub fn int64(v: i64) -> arrow::array::Scalar<arrow::array::Int64Array> {
        arrow::array::Scalar::new(arrow::array::Int64Array::from(vec![v]))
    }

    /// Creates a `Float32` scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{Series, Numeric};
    ///
    /// let s = Series::from("x", vec![1.5f32, 2.5, 3.5]);
    /// let mask = s.greater_than(Numeric::float32(2.0)).unwrap();
    /// ```
    pub fn float32(v: f32) -> arrow::array::Scalar<arrow::array::Float32Array> {
        arrow::array::Scalar::new(arrow::array::Float32Array::from(vec![v]))
    }

    /// Creates a `Float64` scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{Series, Numeric};
    ///
    /// let s = Series::from("x", vec![1.0, 2.0, 3.0]);
    /// let mask = s.less_than(Numeric::float64(2.5)).unwrap();
    /// ```
    pub fn float64(v: f64) -> arrow::array::Scalar<arrow::array::Float64Array> {
        arrow::array::Scalar::new(arrow::array::Float64Array::from(vec![v]))
    }

    /// Creates a `Date32` scalar (days since Unix epoch).
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{Series, Numeric};
    ///
    /// let dates = Series::from_date32("d", vec![19700, 19701, 19702]);
    /// let mask = dates.greater_than(Numeric::date32(19700)).unwrap();
    /// ```
    pub fn date32(days: i32) -> arrow::array::Scalar<arrow::array::Date32Array> {
        arrow::array::Scalar::new(arrow::array::Date32Array::from(vec![days]))
    }

    /// Creates a `Date64` scalar (milliseconds since Unix epoch).
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{Series, Numeric};
    ///
    /// let ms_per_day: i64 = 86_400_000;
    /// let dates = Series::from_date64("d", vec![19700 * ms_per_day, 19701 * ms_per_day]);
    /// let mask = dates.less_than(Numeric::date64(19701 * ms_per_day)).unwrap();
    /// ```
    pub fn date64(ms: i64) -> arrow::array::Scalar<arrow::array::Date64Array> {
        arrow::array::Scalar::new(arrow::array::Date64Array::from(vec![ms]))
    }

    /// Creates a `Timestamp(Millisecond)` scalar (milliseconds since Unix epoch).
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{Series, Numeric};
    ///
    /// let ms: i64 = 1700000000000;
    /// let ts = Series::from_timestamp_ms("ts", vec![ms, ms + 1000]);
    /// let mask = ts.greater_than(Numeric::timestamp_millis(ms)).unwrap();
    /// ```
    pub fn timestamp_millis(
        ms: i64,
    ) -> arrow::array::Scalar<arrow::array::TimestampMillisecondArray> {
        arrow::array::Scalar::new(arrow::array::TimestampMillisecondArray::from(vec![ms]))
    }
}

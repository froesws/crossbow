//! Group-by operations for `DataFrame`.

use crate::{CrossbowError, DataFrame, Series, expected_array};
use arrow::array::Array;
use arrow::datatypes::DataType;
use std::collections::HashMap;

macro_rules! group_sum_body {
    ($col:expr, $array_ty:ty, $indices:ident, $sums:ident) => {{
        let arr = $col
            .as_primitive::<$array_ty>()
            .ok_or_else(|| expected_array(stringify!($array_ty)))?;
        let mut sum: f64 = 0.0;
        for &idx in $indices {
            if arr.is_valid(idx) {
                sum += arr.value(idx) as f64;
            }
        }
        $sums.push(sum);
    }};
}

macro_rules! group_mean_body {
    ($col:expr, $array_ty:ty, $indices:ident, $means:ident) => {{
        let arr = $col
            .as_primitive::<$array_ty>()
            .ok_or_else(|| expected_array(stringify!($array_ty)))?;
        let mut sum = 0.0;
        let mut count = 0;
        for &idx in $indices {
            if arr.is_valid(idx) {
                sum += arr.value(idx) as f64;
                count += 1;
            }
        }
        let mean_val = if count > 0 { sum / count as f64 } else { 0.0 };
        $means.push(mean_val);
    }};
}

macro_rules! group_minmax_body {
    ($col:expr, $array_ty:ty, $indices:ident, $out:ident, $cmp:tt, $init:expr) => {{
        let arr = $col
            .as_primitive::<$array_ty>()
            .ok_or_else(|| expected_array(stringify!($array_ty)))?;
        let mut val = $init;
        let mut has_valid = false;
        for &idx in $indices {
            if arr.is_valid(idx) {
                let v = arr.value(idx) as f64;
                if v $cmp val {
                    val = v;
                }
                has_valid = true;
            }
        }
        $out.push(if has_valid { val } else { f64::NAN });
    }};
}

macro_rules! group_minmax_arms {
    ($agg_col:ident, $indices:ident, $out:ident, $cmp:tt, $init:expr) => {
        match $agg_col.dtype() {
            DataType::Int32 => group_minmax_body!(
                $agg_col,
                arrow::array::Int32Array,
                $indices,
                $out,
                $cmp,
                $init
            ),
            DataType::Int64 => group_minmax_body!(
                $agg_col,
                arrow::array::Int64Array,
                $indices,
                $out,
                $cmp,
                $init
            ),
            DataType::Float32 => group_minmax_body!(
                $agg_col,
                arrow::array::Float32Array,
                $indices,
                $out,
                $cmp,
                $init
            ),
            DataType::Float64 => group_minmax_body!(
                $agg_col,
                arrow::array::Float64Array,
                $indices,
                $out,
                $cmp,
                $init
            ),
            DataType::Date32 => group_minmax_body!(
                $agg_col,
                arrow::array::Date32Array,
                $indices,
                $out,
                $cmp,
                $init
            ),
            DataType::Date64 => group_minmax_body!(
                $agg_col,
                arrow::array::Date64Array,
                $indices,
                $out,
                $cmp,
                $init
            ),
            DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None) => {
                group_minmax_body!(
                    $agg_col,
                    arrow::array::TimestampMillisecondArray,
                    $indices,
                    $out,
                    $cmp,
                    $init
                )
            }
            _ => {
                return Err(CrossbowError::OperationNotSupported(format!(
                    "aggregation not supported for {:?}",
                    $agg_col.dtype()
                )));
            }
        }
    };
}

impl DataFrame {
    /// Groups rows by the values in a column.
    ///
    /// Returns a [`GroupedDataFrame`] that supports `count`, `sum`, `mean`,
    /// `min`, and `max` aggregation per group.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let cat = Series::from("cat", vec!["a", "b", "a"]);
    /// let val = Series::from("val", vec![1i32, 2, 3]);
    /// let df = DataFrame::new(vec![cat, val]).unwrap();
    /// let grouped = df.group_by("cat").unwrap();
    /// let result = grouped.sum("val").unwrap();
    /// assert_eq!(result.shape(), (2, 2));
    /// ```
    pub fn group_by(&self, column_name: &str) -> Result<GroupedDataFrame, CrossbowError> {
        let group_col = self.select(column_name)?;
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

        for i in 0..group_col.len() {
            let key = group_col.get_value_as_string(i);
            groups.entry(key).or_default().push(i);
        }

        Ok(GroupedDataFrame {
            dataframe: self.clone(),
            group_column: column_name.to_string(),
            groups,
        })
    }
}

/// A grouped `DataFrame` produced by [`DataFrame::group_by`].
///
/// Supports `count()`, `sum()`, `mean()`, `min()`, and `max()`
/// aggregation per group.
///
/// # Examples
///
/// ```
/// use crossbow::{DataFrame, Series};
///
/// let cat = Series::from("cat", vec!["a", "b", "a"]);
/// let val = Series::from("val", vec![1i32, 2, 3]);
/// let df = DataFrame::new(vec![cat, val]).unwrap();
/// let grouped = df.group_by("cat").unwrap();
/// let result = grouped.count().unwrap();
/// assert_eq!(result.shape(), (2, 2));
/// ```
#[derive(Debug, Clone)]
pub struct GroupedDataFrame {
    dataframe: DataFrame,
    group_column: String,
    groups: HashMap<String, Vec<usize>>,
}

impl GroupedDataFrame {
    /// Counts rows in each group. Returns a `DataFrame` with the grouping
    /// key and a `count` column.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let cat = Series::from("cat", vec!["x", "x", "y"]);
    /// let val = Series::from("val", vec![1i32, 2, 3]);
    /// let df = DataFrame::new(vec![cat, val]).unwrap();
    /// let grouped = df.group_by("cat").unwrap();
    /// let counts = grouped.count().unwrap();
    /// assert_eq!(counts.shape(), (2, 2));
    /// ```
    pub fn count(&self) -> Result<DataFrame, CrossbowError> {
        let mut keys = Vec::new();
        let mut counts = Vec::new();

        for (key, indices) in &self.groups {
            keys.push(key.clone());
            counts.push(indices.len() as i32);
        }

        let key_series = Series::from(&self.group_column, keys);
        let count_series = Series::from("count", counts);

        DataFrame::new(vec![key_series, count_series])
    }

    /// Computes the sum of a numeric column for each group.
    ///
    /// Supports `Int32`, `Int64`, `Float32`, and `Float64` columns.
    /// The result column is always `Float64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let cat = Series::from("cat", vec!["a", "b", "a"]);
    /// let val = Series::from("val", vec![10i32, 20, 30]);
    /// let df = DataFrame::new(vec![cat, val]).unwrap();
    /// let grouped = df.group_by("cat").unwrap();
    /// let result = grouped.sum("val").unwrap();
    /// // a=40, b=20
    /// assert_eq!(result.shape().0, 2);
    /// ```
    pub fn sum(&self, column_name: &str) -> Result<DataFrame, CrossbowError> {
        let agg_col = self.dataframe.select(column_name)?;

        let mut keys = Vec::new();
        let mut sums: Vec<f64> = Vec::new();

        for (key, indices) in &self.groups {
            keys.push(key.clone());

            match agg_col.dtype() {
                DataType::Int32 => {
                    group_sum_body!(agg_col, arrow::array::Int32Array, indices, sums)
                }
                DataType::Int64 => {
                    group_sum_body!(agg_col, arrow::array::Int64Array, indices, sums)
                }
                DataType::Float32 => {
                    group_sum_body!(agg_col, arrow::array::Float32Array, indices, sums)
                }
                DataType::Float64 => {
                    group_sum_body!(agg_col, arrow::array::Float64Array, indices, sums)
                }
                _ => {
                    return Err(CrossbowError::OperationNotSupported(format!(
                        "sum() not supported for {:?}",
                        agg_col.dtype()
                    )));
                }
            }
        }

        let key_series = Series::from(&self.group_column, keys);
        let sum_series = Series::from(&format!("{}_sum", column_name), sums);

        DataFrame::new(vec![key_series, sum_series])
    }

    /// Computes the arithmetic mean of a numeric column for each group.
    ///
    /// Supports `Int32`, `Int64`, `Float32`, and `Float64` columns.
    /// The result column is always `Float64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let cat = Series::from("cat", vec!["a", "a", "b"]);
    /// let val = Series::from("val", vec![10i32, 20, 30]);
    /// let df = DataFrame::new(vec![cat, val]).unwrap();
    /// let grouped = df.group_by("cat").unwrap();
    /// let result = grouped.mean("val").unwrap();
    /// // a=15, b=30
    /// assert_eq!(result.shape().0, 2);
    /// ```
    pub fn mean(&self, column_name: &str) -> Result<DataFrame, CrossbowError> {
        let agg_col = self.dataframe.select(column_name)?;

        let mut keys = Vec::new();
        let mut means: Vec<f64> = Vec::new();

        for (key, indices) in &self.groups {
            keys.push(key.clone());

            match agg_col.dtype() {
                DataType::Int32 => {
                    group_mean_body!(agg_col, arrow::array::Int32Array, indices, means)
                }
                DataType::Int64 => {
                    group_mean_body!(agg_col, arrow::array::Int64Array, indices, means)
                }
                DataType::Float32 => {
                    group_mean_body!(agg_col, arrow::array::Float32Array, indices, means)
                }
                DataType::Float64 => {
                    group_mean_body!(agg_col, arrow::array::Float64Array, indices, means)
                }
                _ => {
                    return Err(CrossbowError::OperationNotSupported(format!(
                        "mean() not supported for {:?}",
                        agg_col.dtype()
                    )));
                }
            }
        }

        let key_series = Series::from(&self.group_column, keys);
        let mean_series = Series::from(&format!("{}_mean", column_name), means);

        DataFrame::new(vec![key_series, mean_series])
    }

    /// Computes the minimum value of a numeric or date column for each group.
    ///
    /// Supports all numeric and date types. Returns `f64::NAN` for
    /// groups with no valid values. The result column is always `Float64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let cat = Series::from("cat", vec!["a", "a", "b"]);
    /// let val = Series::from("val", vec![10i32, 20, 30]);
    /// let df = DataFrame::new(vec![cat, val]).unwrap();
    /// let grouped = df.group_by("cat").unwrap();
    /// let result = grouped.min("val").unwrap();
    /// assert_eq!(result.shape().0, 2);
    /// ```
    pub fn min(&self, column_name: &str) -> Result<DataFrame, CrossbowError> {
        let agg_col = self.dataframe.select(column_name)?;

        let mut keys = Vec::new();
        let mut mins: Vec<f64> = Vec::new();

        for (key, indices) in &self.groups {
            keys.push(key.clone());
            group_minmax_arms!(agg_col, indices, mins, <, f64::INFINITY);
        }

        let key_series = Series::from(&self.group_column, keys);
        let min_series = Series::from(&format!("{}_min", column_name), mins);

        DataFrame::new(vec![key_series, min_series])
    }

    /// Computes the maximum value of a numeric or date column for each group.
    ///
    /// Supports all numeric and date types. Returns `f64::NAN` for
    /// groups with no valid values. The result column is always `Float64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let cat = Series::from("cat", vec!["a", "a", "b"]);
    /// let val = Series::from("val", vec![10i32, 20, 30]);
    /// let df = DataFrame::new(vec![cat, val]).unwrap();
    /// let grouped = df.group_by("cat").unwrap();
    /// let result = grouped.max("val").unwrap();
    /// assert_eq!(result.shape().0, 2);
    /// ```
    pub fn max(&self, column_name: &str) -> Result<DataFrame, CrossbowError> {
        let agg_col = self.dataframe.select(column_name)?;

        let mut keys = Vec::new();
        let mut maxs: Vec<f64> = Vec::new();

        for (key, indices) in &self.groups {
            keys.push(key.clone());
            group_minmax_arms!(agg_col, indices, maxs, >, f64::NEG_INFINITY);
        }

        let key_series = Series::from(&self.group_column, keys);
        let max_series = Series::from(&format!("{}_max", column_name), maxs);

        DataFrame::new(vec![key_series, max_series])
    }
}

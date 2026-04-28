//! Group-by operations for `DataFrame`.

use std::collections::HashMap;
use arrow::array::Array;
use crate::{CrossbowError, DataFrame, Series, expected_array};

impl DataFrame {
    /// Groups rows by the values in a column.
    ///
    /// Returns a [`GroupedDataFrame`] that supports `count`, `sum`, and `mean`
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
/// Supports `count()`, `sum()`, and `mean()` aggregation per group.
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
    /// Counts rows in each group. Returns a `DataFrame` with the grouping key and a `count` column.
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
        let mut sums = Vec::new();

        for (key, indices) in &self.groups {
            keys.push(key.clone());

            match agg_col.dtype() {
                arrow::datatypes::DataType::Int32 => {
                    let arr = agg_col.as_primitive::<arrow::array::Int32Array>()
                        .ok_or_else(|| expected_array("Int32Array"))?;
                    let mut sum = 0i64;
                    for &idx in indices {
                        if arr.is_valid(idx) {
                            sum += arr.value(idx) as i64;
                        }
                    }
                    sums.push(Some(sum as i32));
                }
                arrow::datatypes::DataType::Float64 => {
                    let arr = agg_col.as_primitive::<arrow::array::Float64Array>()
                        .ok_or_else(|| expected_array("Float64Array"))?;
                    let mut sum = 0.0;
                    for &idx in indices {
                        if arr.is_valid(idx) {
                            sum += arr.value(idx);
                        }
                    }
                    sums.push(Some(sum as i32));
                }
                _ => return Err(CrossbowError::OperationNotSupported(
                    format!("sum() not supported for {:?}", agg_col.dtype())
                )),
            }
        }

        let key_series = Series::from(&self.group_column, keys);
        let sum_series = Series::from(&format!("{}_sum", column_name), sums);

        DataFrame::new(vec![key_series, sum_series])
    }

    /// Computes the arithmetic mean of a numeric column for each group.
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
        let mut means = Vec::new();

        for (key, indices) in &self.groups {
            keys.push(key.clone());

            match agg_col.dtype() {
                arrow::datatypes::DataType::Int32 => {
                    let arr = agg_col.as_primitive::<arrow::array::Int32Array>()
                        .ok_or_else(|| expected_array("Int32Array"))?;
                    let mut sum = 0.0;
                    let mut count = 0;
                    for &idx in indices {
                        if arr.is_valid(idx) {
                            sum += arr.value(idx) as f64;
                            count += 1;
                        }
                    }
                    let mean_val = if count > 0 { sum / count as f64 } else { 0.0 };
                    means.push(Some(mean_val));
                }
                arrow::datatypes::DataType::Float64 => {
                    let arr = agg_col.as_primitive::<arrow::array::Float64Array>()
                        .ok_or_else(|| expected_array("Float64Array"))?;
                    let mut sum = 0.0;
                    let mut count = 0;
                    for &idx in indices {
                        if arr.is_valid(idx) {
                            sum += arr.value(idx);
                            count += 1;
                        }
                    }
                    let mean_val = if count > 0 { sum / count as f64 } else { 0.0 };
                    means.push(Some(mean_val));
                }
                _ => return Err(CrossbowError::OperationNotSupported(
                    format!("mean() not supported for {:?}", agg_col.dtype())
                )),
            }
        }

        let key_series = Series::from(&self.group_column, keys);
        let mean_series = Series::from(&format!("{}_mean", column_name), means);

        DataFrame::new(vec![key_series, mean_series])
    }
}

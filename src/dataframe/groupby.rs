//! Group-by operations for `DataFrame`.

use std::collections::HashMap;
use arrow::array::Array;
use crate::{CrossbowError, DataFrame, Series};

impl DataFrame {
    /// Groups rows by the values in a column.
    ///
    /// Returns a [`GroupedDataFrame`] that supports `count`, `sum`, and `mean`
    /// aggregation per group.
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
#[derive(Debug, Clone)]
pub struct GroupedDataFrame {
    dataframe: DataFrame,
    group_column: String,
    groups: HashMap<String, Vec<usize>>,
}

impl GroupedDataFrame {
    /// Counts rows in each group. Returns a `DataFrame` with the grouping key and a `count` column.
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
    pub fn sum(&self, column_name: &str) -> Result<DataFrame, CrossbowError> {
        let agg_col = self.dataframe.select(column_name)?;

        let mut keys = Vec::new();
        let mut sums = Vec::new();

        for (key, indices) in &self.groups {
            keys.push(key.clone());

            match agg_col.dtype() {
                arrow::datatypes::DataType::Int32 => {
                    let arr = agg_col.as_primitive::<arrow::array::Int32Array>()
                        .ok_or_else(|| CrossbowError::TypeMismatch("Expected Int32Array".to_string()))?;
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
                        .ok_or_else(|| CrossbowError::TypeMismatch("Expected Float64Array".to_string()))?;
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
    pub fn mean(&self, column_name: &str) -> Result<DataFrame, CrossbowError> {
        let agg_col = self.dataframe.select(column_name)?;

        let mut keys = Vec::new();
        let mut means = Vec::new();

        for (key, indices) in &self.groups {
            keys.push(key.clone());

            match agg_col.dtype() {
                arrow::datatypes::DataType::Int32 => {
                    let arr = agg_col.as_primitive::<arrow::array::Int32Array>()
                        .ok_or_else(|| CrossbowError::TypeMismatch("Expected Int32Array".to_string()))?;
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
                        .ok_or_else(|| CrossbowError::TypeMismatch("Expected Float64Array".to_string()))?;
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

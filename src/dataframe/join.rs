//! Join operations for `DataFrame`.
//!
//! Supports inner, left, and outer joins on a single key column per side.
//! Colliding column names are automatically renamed with a `_right` suffix.
//!
//! # Examples
//!
//! ```
//! use crossbow::DataFrame;
//! use crossbow::Series;
//!
//! let left = DataFrame::new(vec![
//!     Series::from("id", vec![1i32, 2, 3]),
//!     Series::from("name", vec!["a", "b", "c"]),
//! ]).unwrap();
//!
//! let right = DataFrame::new(vec![
//!     Series::from("id", vec![1i32, 2, 4]),
//!     Series::from("score", vec![10i32, 20, 40]),
//! ]).unwrap();
//!
//! let joined = left.join_inner(&right, "id", "id").unwrap();
//! assert_eq!(joined.shape(), (2, 3)); // id, name, score
//! ```
use crate::{CrossbowError, DataFrame, Series};
use crate::{build_column_opt, build_string_column_opt};
use arrow::array::Array;
use arrow::datatypes::DataType;
use std::collections::HashMap;
use std::sync::Arc;

// Enum representing the three supported join strategies.
#[derive(Clone, Copy, PartialEq)]
enum JoinType {
    Inner,
    Outer,
    Left,
}

// Builds a HashMap from join key values to the list of row indices where they appear.
fn build_join_map(
    df: &DataFrame,
    key_col: &str,
) -> Result<HashMap<String, Vec<usize>>, CrossbowError> {
    let col = df.select(key_col)?;
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for i in 0..col.len() {
        let key = col.get_value_as_string(i);
        map.entry(key).or_default().push(i);
    }
    Ok(map)
}

// Returns a deduplicated column name for a right-side join column. Appends `_right`
// if the name already exists in the left DataFrame's column names.
fn right_column_name(left_names: &[String], right_name: &str) -> String {
    if left_names.contains(&right_name.to_string()) {
        format!("{}_right", right_name)
    } else {
        right_name.to_string()
    }
}

// Builds an Arrow array from optional indices dispatched by DataType. Uses the
// build_column_opt! / build_string_column_opt! macros for type-specific builder creation.
fn build_array_for_dtype(
    arr: &dyn Array,
    dtype: &DataType,
    indices: &[Option<usize>],
    capacity: usize,
) -> Arc<dyn Array> {
    match dtype {
        DataType::Int32 => build_column_opt!(
            arr,
            indices,
            arrow::array::Int32Builder,
            arrow::array::Int32Array,
            capacity
        ),
        DataType::Int64 => build_column_opt!(
            arr,
            indices,
            arrow::array::Int64Builder,
            arrow::array::Int64Array,
            capacity
        ),
        DataType::Float32 => build_column_opt!(
            arr,
            indices,
            arrow::array::Float32Builder,
            arrow::array::Float32Array,
            capacity
        ),
        DataType::Float64 => build_column_opt!(
            arr,
            indices,
            arrow::array::Float64Builder,
            arrow::array::Float64Array,
            capacity
        ),
        DataType::Boolean => build_column_opt!(
            arr,
            indices,
            arrow::array::BooleanBuilder,
            arrow::array::BooleanArray,
            capacity
        ),
        DataType::Utf8 => build_string_column_opt!(arr, indices, capacity),
        DataType::Date32 => build_column_opt!(
            arr,
            indices,
            arrow::array::Date32Builder,
            arrow::array::Date32Array,
            capacity
        ),
        DataType::Date64 => build_column_opt!(
            arr,
            indices,
            arrow::array::Date64Builder,
            arrow::array::Date64Array,
            capacity
        ),
        DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None) => build_column_opt!(
            arr,
            indices,
            arrow::array::TimestampMillisecondBuilder,
            arrow::array::TimestampMillisecondArray,
            capacity
        ),
        _ => panic!("Unsupported type in join: {:?}", dtype),
    }
}

// Builds the left-side array for a join result. Indices beyond `left_rows` are treated
// as sentinel values and produce null entries (for unmatched outer join rows).
fn build_left_array_for_dtype(
    arr: &dyn Array,
    dtype: &DataType,
    indices: &[usize],
    capacity: usize,
    left_rows: usize,
) -> Arc<dyn Array> {
    let opt_indices: Vec<Option<usize>> = indices
        .iter()
        .map(|&li| if li < left_rows { Some(li) } else { None })
        .collect();
    build_array_for_dtype(arr, dtype, &opt_indices, capacity)
}

// Core join engine: pairs matching rows from two DataFrames according to the join type,
// builds aligned index lists, and constructs new columns from the resulting row indices.
fn join_impl(
    left: &DataFrame,
    right: &DataFrame,
    left_on: &str,
    right_on: &str,
    join_type: JoinType,
) -> Result<DataFrame, CrossbowError> {
    let right_map = build_join_map(right, right_on)?;

    let left_names: Vec<String> = left
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let mut out_left_indices: Vec<usize> = Vec::new();
    let mut out_right_indices: Vec<Option<usize>> = Vec::new();

    for i in 0..left.shape().0 {
        let key = left.select(left_on)?.get_value_as_string(i);
        if let Some(ridx_list) = right_map.get(&key) {
            for &ridx in ridx_list {
                out_left_indices.push(i);
                out_right_indices.push(Some(ridx));
            }
        } else if join_type != JoinType::Inner {
            out_left_indices.push(i);
            out_right_indices.push(None);
        }
    }

    if join_type == JoinType::Outer {
        let mut seen_right: Vec<bool> = vec![false; right.shape().0];
        for &o in &out_right_indices {
            if let Some(idx) = o {
                seen_right[idx] = true;
            }
        }
        for j in 0..right.shape().0 {
            if !seen_right[j] {
                out_left_indices.push(left.shape().0);
                out_right_indices.push(Some(j));
            }
        }
    }

    let total_rows = out_left_indices.len();
    let left_rows = left.shape().0;
    let mut new_columns: Vec<Series> = Vec::new();

    for col in left.columns() {
        let arr = col.data();
        let built = build_left_array_for_dtype(
            arr.as_ref(),
            col.dtype(),
            &out_left_indices,
            total_rows,
            left_rows,
        );
        new_columns.push(Series::new(col.name(), built));
    }

    for col in right.columns() {
        if col.name() == right_on {
            continue;
        }
        let arr = col.data();
        let out_name = right_column_name(&left_names, col.name());
        let built =
            build_array_for_dtype(arr.as_ref(), col.dtype(), &out_right_indices, total_rows);
        new_columns.push(Series::new(&out_name, built));
    }

    DataFrame::new(new_columns)
}

impl DataFrame {
    /// Inner join: returns only rows where the join key matches in both DataFrames.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let left = DataFrame::new(vec![
    ///     Series::from("id", vec![1i32, 2, 3]),
    ///     Series::from("name", vec!["a", "b", "c"]),
    /// ]).unwrap();
    /// let right = DataFrame::new(vec![
    ///     Series::from("id", vec![1i32, 2]),
    ///     Series::from("val", vec![10i32, 20]),
    /// ]).unwrap();
    /// let joined = left.join_inner(&right, "id", "id").unwrap();
    /// assert_eq!(joined.shape(), (2, 3));
    /// ```
    pub fn join_inner(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
    ) -> Result<DataFrame, CrossbowError> {
        join_impl(self, other, left_on, right_on, JoinType::Inner)
    }

    /// Outer join: returns all rows from both DataFrames.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let left = DataFrame::new(vec![
    ///     Series::from("id", vec![1i32, 2]),
    ///     Series::from("name", vec!["a", "b"]),
    /// ]).unwrap();
    /// let right = DataFrame::new(vec![
    ///     Series::from("id", vec![2i32, 3]),
    ///     Series::from("val", vec![20i32, 30]),
    /// ]).unwrap();
    /// let joined = left.join_outer(&right, "id", "id").unwrap();
    /// assert_eq!(joined.shape(), (3, 3));
    /// ```
    pub fn join_outer(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
    ) -> Result<DataFrame, CrossbowError> {
        join_impl(self, other, left_on, right_on, JoinType::Outer)
    }

    /// Left join: returns all rows from the left DataFrame.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let left = DataFrame::new(vec![
    ///     Series::from("id", vec![1i32, 2, 3]),
    ///     Series::from("name", vec!["a", "b", "c"]),
    /// ]).unwrap();
    /// let right = DataFrame::new(vec![
    ///     Series::from("id", vec![1i32, 3]),
    ///     Series::from("val", vec![10i32, 30]),
    /// ]).unwrap();
    /// let joined = left.join_left(&right, "id", "id").unwrap();
    /// assert_eq!(joined.shape(), (3, 3));
    /// ```
    pub fn join_left(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
    ) -> Result<DataFrame, CrossbowError> {
        join_impl(self, other, left_on, right_on, JoinType::Left)
    }
}

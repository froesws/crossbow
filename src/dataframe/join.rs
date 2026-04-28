use std::collections::HashMap;
use std::sync::Arc;
use arrow::array::Array;
use arrow::datatypes::DataType;
use crate::{CrossbowError, DataFrame, Series};

#[derive(Clone, Copy, PartialEq)]
enum JoinType {
    Inner,
    Outer,
    Left,
}

fn build_join_map(df: &DataFrame, key_col: &str) -> Result<HashMap<String, Vec<usize>>, CrossbowError> {
    let col = df.select(key_col)?;
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for i in 0..col.len() {
        let key = col.get_value_as_string(i);
        map.entry(key).or_default().push(i);
    }
    Ok(map)
}

fn right_column_name(left_names: &[String], right_name: &str) -> String {
    if left_names.contains(&right_name.to_string()) {
        format!("{}_right", right_name)
    } else {
        right_name.to_string()
    }
}

macro_rules! build_column_from_indices {
    ($arr:expr, $indices:expr, $builder_ty:ty, $arr_downcast:ty, $cap:expr) => {{
        let a = $arr.as_any().downcast_ref::<$arr_downcast>().unwrap();
        let mut b = <$builder_ty>::with_capacity($cap);
        for &oi in $indices {
            match oi {
                Some(idx) => {
                    if a.is_valid(idx) {
                        b.append_value(a.value(idx));
                    } else {
                        b.append_null();
                    }
                }
                None => b.append_null(),
            }
        }
        Arc::new(b.finish()) as Arc<dyn Array>
    }};
}

macro_rules! build_column_all_left {
    ($arr:expr, $indices:expr, $builder_ty:ty, $arr_downcast:ty, $cap:expr, $lrow:expr) => {{
        let a = $arr.as_any().downcast_ref::<$arr_downcast>().unwrap();
        let mut b = <$builder_ty>::with_capacity($cap);
        for &li in $indices {
            if li < $lrow {
                if a.is_valid(li) {
                    b.append_value(a.value(li));
                } else {
                    b.append_null();
                }
            } else {
                b.append_null();
            }
        }
        Arc::new(b.finish()) as Arc<dyn Array>
    }};
}

fn build_array_for_dtype(
    arr: &dyn Array,
    dtype: &DataType,
    indices: &[Option<usize>],
    capacity: usize,
) -> Arc<dyn Array> {
    match dtype {
        DataType::Int32 => build_column_from_indices!(arr, indices, arrow::array::Int32Builder, arrow::array::Int32Array, capacity),
        DataType::Int64 => build_column_from_indices!(arr, indices, arrow::array::Int64Builder, arrow::array::Int64Array, capacity),
        DataType::Float32 => build_column_from_indices!(arr, indices, arrow::array::Float32Builder, arrow::array::Float32Array, capacity),
        DataType::Float64 => build_column_from_indices!(arr, indices, arrow::array::Float64Builder, arrow::array::Float64Array, capacity),
        DataType::Boolean => build_column_from_indices!(arr, indices, arrow::array::BooleanBuilder, arrow::array::BooleanArray, capacity),
        DataType::Utf8 => {
            let a = arr.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
            let mut b = arrow::array::StringBuilder::with_capacity(capacity, capacity * 16);
            for &oi in indices {
                match oi {
                    Some(idx) => {
                        if a.is_valid(idx) { b.append_value(a.value(idx)); } else { b.append_null(); }
                    }
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish()) as Arc<dyn Array>
        }
        _ => panic!("Unsupported type in join: {:?}", dtype),
    }
}

fn build_left_array_for_dtype(
    arr: &dyn Array,
    dtype: &DataType,
    indices: &[usize],
    capacity: usize,
    left_rows: usize,
) -> Arc<dyn Array> {
    match dtype {
        DataType::Int32 => build_column_all_left!(arr, indices, arrow::array::Int32Builder, arrow::array::Int32Array, capacity, left_rows),
        DataType::Int64 => build_column_all_left!(arr, indices, arrow::array::Int64Builder, arrow::array::Int64Array, capacity, left_rows),
        DataType::Float32 => build_column_all_left!(arr, indices, arrow::array::Float32Builder, arrow::array::Float32Array, capacity, left_rows),
        DataType::Float64 => build_column_all_left!(arr, indices, arrow::array::Float64Builder, arrow::array::Float64Array, capacity, left_rows),
        DataType::Boolean => build_column_all_left!(arr, indices, arrow::array::BooleanBuilder, arrow::array::BooleanArray, capacity, left_rows),
        DataType::Utf8 => {
            let a = arr.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
            let mut b = arrow::array::StringBuilder::with_capacity(capacity, capacity * 16);
            for &li in indices {
                if li < left_rows {
                    if a.is_valid(li) { b.append_value(a.value(li)); } else { b.append_null(); }
                } else {
                    b.append_null();
                }
            }
            Arc::new(b.finish()) as Arc<dyn Array>
        }
        _ => panic!("Unsupported type in join: {:?}", dtype),
    }
}

fn join_impl(
    left: &DataFrame,
    right: &DataFrame,
    left_on: &str,
    right_on: &str,
    join_type: JoinType,
) -> Result<DataFrame, CrossbowError> {
    let right_map = build_join_map(right, right_on)?;

    let left_names: Vec<String> = left.get_column_names().iter().map(|s| s.to_string()).collect();

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
            if let Some(idx) = o { seen_right[idx] = true; }
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
        let arr_ref: &dyn Array = arr.as_ref();
        let built = build_left_array_for_dtype(arr_ref, col.dtype(), &out_left_indices, total_rows, left_rows);
        new_columns.push(Series::new(col.name(), built));
    }

    for col in right.columns() {
        if col.name() == right_on { continue; }
        let arr = col.data();
        let arr_ref: &dyn Array = arr.as_ref();
        let out_name = right_column_name(&left_names, col.name());
        let built = build_array_for_dtype(arr_ref, col.dtype(), &out_right_indices, total_rows);
        new_columns.push(Series::new(&out_name, built));
    }

    DataFrame::new(new_columns)
}

impl DataFrame {
    /// Inner join: returns only rows where the join key matches in both DataFrames.
    ///
    /// Columns from the left `DataFrame` keep their names. Columns from the right
    /// are included except the join key. If a right column name collides with a
    /// left column, `_right` is appended.
    pub fn join_inner(&self, other: &DataFrame, left_on: &str, right_on: &str) -> Result<DataFrame, CrossbowError> {
        join_impl(self, other, left_on, right_on, JoinType::Inner)
    }

    /// Outer join: returns all rows from both DataFrames. Rows without a match
    /// have nulls in columns from the other side.
    pub fn join_outer(&self, other: &DataFrame, left_on: &str, right_on: &str) -> Result<DataFrame, CrossbowError> {
        join_impl(self, other, left_on, right_on, JoinType::Outer)
    }

    /// Left join: returns all rows from the left `DataFrame`. Rows without a
    /// match in the right have nulls in right-side columns.
    pub fn join_left(&self, other: &DataFrame, left_on: &str, right_on: &str) -> Result<DataFrame, CrossbowError> {
        join_impl(self, other, left_on, right_on, JoinType::Left)
    }
}

//! Column-based sorting for `DataFrame`.

use std::sync::Arc;
use arrow::array::Array;
use crate::{CrossbowError, DataFrame, Series};

impl DataFrame {
    /// Sorts the `DataFrame` by a column.
    ///
    /// Set `ascending: true` for smallest-first, `false` for largest-first.
    /// Null values sort first.
    pub fn sort_by(&self, column_name: &str, ascending: bool) -> Result<DataFrame, CrossbowError> {
        let sort_col = self.select(column_name)?;

        let mut indices: Vec<usize> = (0..sort_col.len()).collect();

        match sort_col.dtype() {
            arrow::datatypes::DataType::Int32 => {
                let arr = sort_col.as_primitive::<arrow::array::Int32Array>()
                    .ok_or_else(|| CrossbowError::TypeMismatch("Expected Int32Array".to_string()))?;
                if ascending {
                    indices.sort_by(|&a, &b| {
                        let val_a = if arr.is_valid(a) { Some(arr.value(a)) } else { None };
                        let val_b = if arr.is_valid(b) { Some(arr.value(b)) } else { None };
                        val_a.cmp(&val_b)
                    });
                } else {
                    indices.sort_by(|&a, &b| {
                        let val_a = if arr.is_valid(a) { Some(arr.value(a)) } else { None };
                        let val_b = if arr.is_valid(b) { Some(arr.value(b)) } else { None };
                        val_b.cmp(&val_a)
                    });
                }
            }
            arrow::datatypes::DataType::Int64 => {
                let arr = sort_col.as_primitive::<arrow::array::Int64Array>()
                    .ok_or_else(|| CrossbowError::TypeMismatch("Expected Int64Array".to_string()))?;
                if ascending {
                    indices.sort_by(|&a, &b| {
                        let val_a = if arr.is_valid(a) { Some(arr.value(a)) } else { None };
                        let val_b = if arr.is_valid(b) { Some(arr.value(b)) } else { None };
                        val_a.cmp(&val_b)
                    });
                } else {
                    indices.sort_by(|&a, &b| {
                        let val_a = if arr.is_valid(a) { Some(arr.value(a)) } else { None };
                        let val_b = if arr.is_valid(b) { Some(arr.value(b)) } else { None };
                        val_b.cmp(&val_a)
                    });
                }
            }
            arrow::datatypes::DataType::Float32 => {
                let arr = sort_col.as_primitive::<arrow::array::Float32Array>()
                    .ok_or_else(|| CrossbowError::TypeMismatch("Expected Float32Array".to_string()))?;
                if ascending {
                    indices.sort_by(|&a, &b| {
                        let val_a = if arr.is_valid(a) { Some(arr.value(a)) } else { None };
                        let val_b = if arr.is_valid(b) { Some(arr.value(b)) } else { None };
                        val_a.partial_cmp(&val_b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                } else {
                    indices.sort_by(|&a, &b| {
                        let val_a = if arr.is_valid(a) { Some(arr.value(a)) } else { None };
                        let val_b = if arr.is_valid(b) { Some(arr.value(b)) } else { None };
                        val_b.partial_cmp(&val_a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
            arrow::datatypes::DataType::Float64 => {
                let arr = sort_col.as_primitive::<arrow::array::Float64Array>()
                    .ok_or_else(|| CrossbowError::TypeMismatch("Expected Float64Array".to_string()))?;

                if ascending {
                    indices.sort_by(|&a, &b| {
                        let val_a = if arr.is_valid(a) { Some(arr.value(a)) } else { None };
                        let val_b = if arr.is_valid(b) { Some(arr.value(b)) } else { None };
                        val_a.partial_cmp(&val_b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                } else {
                    indices.sort_by(|&a, &b| {
                        let val_a = if arr.is_valid(a) { Some(arr.value(a)) } else { None };
                        let val_b = if arr.is_valid(b) { Some(arr.value(b)) } else { None };
                        val_b.partial_cmp(&val_a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
            arrow::datatypes::DataType::Utf8 => {
                let arr = sort_col.data().as_any().downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| CrossbowError::TypeMismatch("Expected StringArray".to_string()))?;

                if ascending {
                    indices.sort_by(|&a, &b| {
                        let val_a = if arr.is_valid(a) { Some(arr.value(a)) } else { None };
                        let val_b = if arr.is_valid(b) { Some(arr.value(b)) } else { None };
                        val_a.cmp(&val_b)
                    });
                } else {
                    indices.sort_by(|&a, &b| {
                        let val_a = if arr.is_valid(a) { Some(arr.value(a)) } else { None };
                        let val_b = if arr.is_valid(b) { Some(arr.value(b)) } else { None };
                        val_b.cmp(&val_a)
                    });
                }
            }
            _ => {
                return Err(CrossbowError::OperationNotSupported(
                    format!("Sorting not supported for {:?}", sort_col.dtype())
                ));
            }
        }

        let mut sorted_columns = Vec::new();
        for series in &self.columns {
            match series.dtype() {
                arrow::datatypes::DataType::Int32 => {
                    let arr = series.as_primitive::<arrow::array::Int32Array>()
                        .ok_or_else(|| CrossbowError::TypeMismatch("Expected Int32Array".to_string()))?;
                    let mut b = arrow::array::Int32Builder::with_capacity(indices.len());
                    for &idx in &indices {
                        if arr.is_valid(idx) {
                            b.append_value(arr.value(idx));
                        } else {
                            b.append_null();
                        }
                    }
                    sorted_columns.push(Series::new(series.name(), Arc::new(b.finish())));
                }
                arrow::datatypes::DataType::Int64 => {
                    let arr = series.as_primitive::<arrow::array::Int64Array>()
                        .ok_or_else(|| CrossbowError::TypeMismatch("Expected Int64Array".to_string()))?;
                    let mut b = arrow::array::Int64Builder::with_capacity(indices.len());
                    for &idx in &indices {
                        if arr.is_valid(idx) {
                            b.append_value(arr.value(idx));
                        } else {
                            b.append_null();
                        }
                    }
                    sorted_columns.push(Series::new(series.name(), Arc::new(b.finish())));
                }
                arrow::datatypes::DataType::Float32 => {
                    let arr = series.as_primitive::<arrow::array::Float32Array>()
                        .ok_or_else(|| CrossbowError::TypeMismatch("Expected Float32Array".to_string()))?;
                    let mut b = arrow::array::Float32Builder::with_capacity(indices.len());
                    for &idx in &indices {
                        if arr.is_valid(idx) {
                            b.append_value(arr.value(idx));
                        } else {
                            b.append_null();
                        }
                    }
                    sorted_columns.push(Series::new(series.name(), Arc::new(b.finish())));
                }
                arrow::datatypes::DataType::Float64 => {
                    let arr = series.as_primitive::<arrow::array::Float64Array>()
                        .ok_or_else(|| CrossbowError::TypeMismatch("Expected Float64Array".to_string()))?;
                    let mut b = arrow::array::Float64Builder::with_capacity(indices.len());
                    for &idx in &indices {
                        if arr.is_valid(idx) {
                            b.append_value(arr.value(idx));
                        } else {
                            b.append_null();
                        }
                    }
                    sorted_columns.push(Series::new(series.name(), Arc::new(b.finish())));
                }
                arrow::datatypes::DataType::Utf8 => {
                    let arr = series.data().as_any().downcast_ref::<arrow::array::StringArray>()
                        .ok_or_else(|| CrossbowError::TypeMismatch("Expected StringArray".to_string()))?;
                    let mut b = arrow::array::StringBuilder::with_capacity(indices.len(), indices.len() * 32);
                    for &idx in &indices {
                        if arr.is_valid(idx) {
                            b.append_value(arr.value(idx));
                        } else {
                            b.append_null();
                        }
                    }
                    sorted_columns.push(Series::new(series.name(), Arc::new(b.finish())));
                }
                _ => {
                    return Err(CrossbowError::OperationNotSupported(
                        format!("Sorting not supported for {:?}", series.dtype())
                    ));
                }
            }
        }

        DataFrame::new(sorted_columns)
    }
}

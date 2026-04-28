//! Column-based sorting for `DataFrame`.

use std::sync::Arc;
use arrow::array::Array;
use crate::{CrossbowError, DataFrame, Series};
use crate::{build_column, build_string_column, expected_array};

impl DataFrame {
    /// Sorts the `DataFrame` by a column. Set `ascending: true` for
    /// smallest-first, `false` for largest-first. Null values sort first.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("val", vec![3i32, 1, 2]);
    /// let df = DataFrame::new(vec![a]).unwrap();
    /// let sorted = df.sort_by("val", true).unwrap();
    /// let row = sorted.get_row(0).unwrap();
    /// assert_eq!(row[0], "1");
    /// ```
    pub fn sort_by(&self, column_name: &str, ascending: bool) -> Result<DataFrame, CrossbowError> {
        let sort_col = self.select(column_name)?;

        let mut indices: Vec<usize> = (0..sort_col.len()).collect();

        match sort_col.dtype() {
            arrow::datatypes::DataType::Int32 => {
                let arr = sort_col.as_primitive::<arrow::array::Int32Array>()
                    .ok_or_else(|| expected_array("Int32Array"))?;
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
                    .ok_or_else(|| expected_array("Int64Array"))?;
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
                    .ok_or_else(|| expected_array("Float32Array"))?;
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
                    .ok_or_else(|| expected_array("Float64Array"))?;

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
                    .ok_or_else(|| expected_array("StringArray"))?;

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

        let n = indices.len();
        let mut sorted_columns = Vec::new();
        for series in &self.columns {
            let arr = series.data();
            let built = match series.dtype() {
                arrow::datatypes::DataType::Int32 => build_column!(arr.as_ref(), &indices, arrow::array::Int32Builder, arrow::array::Int32Array, n),
                arrow::datatypes::DataType::Int64 => build_column!(arr.as_ref(), &indices, arrow::array::Int64Builder, arrow::array::Int64Array, n),
                arrow::datatypes::DataType::Float32 => build_column!(arr.as_ref(), &indices, arrow::array::Float32Builder, arrow::array::Float32Array, n),
                arrow::datatypes::DataType::Float64 => build_column!(arr.as_ref(), &indices, arrow::array::Float64Builder, arrow::array::Float64Array, n),
                arrow::datatypes::DataType::Utf8 => build_string_column!(arr.as_ref(), &indices, n),
                _ => return Err(CrossbowError::OperationNotSupported(
                    format!("Sorting not supported for {:?}", series.dtype())
                )),
            };
            sorted_columns.push(Series::new(series.name(), built));
        }

        DataFrame::new(sorted_columns)
    }
}

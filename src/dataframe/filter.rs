use std::sync::Arc;
use arrow::array::Array;
use crate::{CrossbowError, DataFrame, Series};

impl DataFrame {
    pub fn filter_by_mask(&self, mask: &Series) -> Result<DataFrame, CrossbowError> {
        if mask.len() != self.shape().0 {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        let mask_array = mask.data().as_any().downcast_ref::<arrow::array::BooleanArray>()
            .ok_or_else(|| CrossbowError::TypeMismatch("Filter mask must be boolean".to_string()))?;

        let mut filtered_columns = Vec::new();
        for series in &self.columns {
            let indices: Vec<usize> = (0..mask_array.len())
                .filter(|&i| mask_array.is_valid(i) && mask_array.value(i))
                .collect();

            match series.dtype() {
                arrow::datatypes::DataType::Int32 => {
                    let arr = series.as_primitive::<arrow::array::Int32Array>()
                        .ok_or_else(|| CrossbowError::TypeMismatch("Expected Int32Array".to_string()))?;
                    let mut b = arrow::array::Int32Builder::new();
                    for &idx in &indices {
                        if arr.is_valid(idx) {
                            b.append_value(arr.value(idx));
                        } else {
                            b.append_null();
                        }
                    }
                    let arr = b.finish();
                    filtered_columns.push(Series::new(series.name(), Arc::new(arr)));
                }
                arrow::datatypes::DataType::Float64 => {
                    let arr = series.as_primitive::<arrow::array::Float64Array>()
                        .ok_or_else(|| CrossbowError::TypeMismatch("Expected Float64Array".to_string()))?;
                    let mut b = arrow::array::Float64Builder::new();
                    for &idx in &indices {
                        if arr.is_valid(idx) {
                            b.append_value(arr.value(idx));
                        } else {
                            b.append_null();
                        }
                    }
                    let arr = b.finish();
                    filtered_columns.push(Series::new(series.name(), Arc::new(arr)));
                }
                arrow::datatypes::DataType::Utf8 => {
                    let arr = series.data().as_any().downcast_ref::<arrow::array::StringArray>()
                        .ok_or_else(|| CrossbowError::TypeMismatch("Expected StringArray".to_string()))?;
                    let mut b = arrow::array::StringBuilder::new();
                    for &idx in &indices {
                        if arr.is_valid(idx) {
                            b.append_value(arr.value(idx));
                        } else {
                            b.append_null();
                        }
                    }
                    let arr = b.finish();
                    filtered_columns.push(Series::new(series.name(), Arc::new(arr)));
                }
                _ => {
                    return Err(CrossbowError::OperationNotSupported(
                        format!("Filtering not supported for {:?}", series.dtype())
                    ));
                }
            };
        }

        DataFrame::new(filtered_columns)
    }
}

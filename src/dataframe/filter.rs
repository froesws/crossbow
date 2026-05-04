//! Boolean-mask filtering for `DataFrame`.

use crate::{CrossbowError, DataFrame, Series};
use crate::{build_column, build_string_column};
use arrow::array::Array;
use arrow::datatypes::DataType;
use std::sync::Arc;

impl DataFrame {
    /// Filters rows using a boolean mask `Series`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("a", vec![1i32, 2, 3, 4]);
    /// let mask = Series::from("mask", vec![true, false, true, false]);
    /// let df = DataFrame::new(vec![a]).unwrap();
    /// let filtered = df.filter_by_mask(&mask).unwrap();
    /// assert_eq!(filtered.shape(), (2, 1));
    /// ```
    pub fn filter_by_mask(&self, mask: &Series) -> Result<DataFrame, CrossbowError> {
        if mask.len() != self.shape().0 {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        let mask_array = mask
            .data()
            .as_any()
            .downcast_ref::<arrow::array::BooleanArray>()
            .ok_or_else(|| {
                CrossbowError::TypeMismatch("Filter mask must be boolean".to_string())
            })?;

        let indices: Vec<usize> = (0..mask_array.len())
            .filter(|&i| mask_array.is_valid(i) && mask_array.value(i))
            .collect();

        let n = indices.len();
        let mut filtered_columns = Vec::new();
        for series in &self.columns {
            let arr = series.data();
            let built = match series.dtype() {
                DataType::Int32 => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::Int32Builder,
                    arrow::array::Int32Array,
                    n
                ),
                DataType::Int64 => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::Int64Builder,
                    arrow::array::Int64Array,
                    n
                ),
                DataType::Float32 => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::Float32Builder,
                    arrow::array::Float32Array,
                    n
                ),
                DataType::Float64 => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::Float64Builder,
                    arrow::array::Float64Array,
                    n
                ),
                DataType::Utf8 => build_string_column!(arr.as_ref(), &indices, n),
                DataType::Boolean => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::BooleanBuilder,
                    arrow::array::BooleanArray,
                    n
                ),
                DataType::Date32 => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::Date32Builder,
                    arrow::array::Date32Array,
                    n
                ),
                DataType::Date64 => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::Date64Builder,
                    arrow::array::Date64Array,
                    n
                ),
                DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None) => {
                    build_column!(
                        arr.as_ref(),
                        &indices,
                        arrow::array::TimestampMillisecondBuilder,
                        arrow::array::TimestampMillisecondArray,
                        n
                    )
                }
                _ => {
                    return Err(CrossbowError::OperationNotSupported(format!(
                        "Filtering not supported for {:?}",
                        series.dtype()
                    )));
                }
            };
            filtered_columns.push(Series::new(series.name(), built));
        }

        DataFrame::new(filtered_columns)
    }
}

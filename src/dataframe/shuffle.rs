//! Random row permutation for `DataFrame`.
//!
//! Implements Fisher-Yates shuffle to randomly reorder rows
//! while preserving row cohesion — every value in a row stays
//! with its original row-mates.

use crate::{CrossbowError, DataFrame, Series};
use crate::{build_column, build_string_column};
use arrow::array::Array;
use rand::Rng;
use std::sync::Arc;

impl DataFrame {
    /// Randomly shuffles all rows using the Fisher-Yates algorithm.
    ///
    /// Row cohesion is preserved — values within a row stay together,
    /// only the order of rows changes. The shuffle is in-place in
    /// terms of a new [`DataFrame`] being returned; the original is
    /// not mutated.
    ///
    /// Uses [`rand::thread_rng`] as the randomness source.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("x", vec![1i32, 2, 3]);
    /// let b = Series::from("y", vec!["a", "b", "c"]);
    /// let df = DataFrame::new(vec![a, b]).unwrap();
    /// let shuffled = df.shuffle().unwrap();
    /// assert_eq!(shuffled.shape(), (3, 2));
    /// ```
    pub fn shuffle(&self) -> Result<DataFrame, CrossbowError> {
        let (n_rows, _) = self.shape();
        if n_rows <= 1 {
            return Ok(self.clone());
        }

        let mut indices: Vec<usize> = (0..n_rows).collect();
        let mut rng = rand::thread_rng();
        let len = indices.len();
        for i in (1..len).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let n = indices.len();
        let mut shuffled_columns = Vec::new();
        for series in &self.columns {
            let arr = series.data();
            let built = match series.dtype() {
                arrow::datatypes::DataType::Int32 => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::Int32Builder,
                    arrow::array::Int32Array,
                    n
                ),
                arrow::datatypes::DataType::Int64 => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::Int64Builder,
                    arrow::array::Int64Array,
                    n
                ),
                arrow::datatypes::DataType::Float32 => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::Float32Builder,
                    arrow::array::Float32Array,
                    n
                ),
                arrow::datatypes::DataType::Float64 => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::Float64Builder,
                    arrow::array::Float64Array,
                    n
                ),
                arrow::datatypes::DataType::Utf8 => build_string_column!(arr.as_ref(), &indices, n),
                arrow::datatypes::DataType::Boolean => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::BooleanBuilder,
                    arrow::array::BooleanArray,
                    n
                ),
                arrow::datatypes::DataType::Date32 => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::Date32Builder,
                    arrow::array::Date32Array,
                    n
                ),
                arrow::datatypes::DataType::Date64 => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::Date64Builder,
                    arrow::array::Date64Array,
                    n
                ),
                arrow::datatypes::DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Millisecond,
                    None,
                ) => build_column!(
                    arr.as_ref(),
                    &indices,
                    arrow::array::TimestampMillisecondBuilder,
                    arrow::array::TimestampMillisecondArray,
                    n
                ),
                _ => {
                    return Err(CrossbowError::OperationNotSupported(format!(
                        "Shuffle not supported for {:?}",
                        series.dtype()
                    )));
                }
            };
            shuffled_columns.push(Series::new(series.name(), built));
        }

        DataFrame::new(shuffled_columns)
    }
}

#[cfg(test)]
mod tests {
    use crate::{DataFrame, Series};

    #[test]
    fn test_shuffle_preserves_shape() {
        let a = Series::from("x", vec![1i32, 2, 3, 4, 5]);
        let b = Series::from("y", vec!["a", "b", "c", "d", "e"]);
        let df = DataFrame::new(vec![a, b]).unwrap();
        let shuffled = df.shuffle().unwrap();
        assert_eq!(df.shape(), shuffled.shape());
        assert_eq!(df.get_column_names(), shuffled.get_column_names());
    }

    #[test]
    fn test_shuffle_preserves_row_cohesion() {
        let a = Series::from("id", vec![1i32, 2, 3, 4, 5]);
        let b = Series::from("label", vec!["a", "b", "c", "d", "e"]);
        let df = DataFrame::new(vec![a, b]).unwrap();

        let original_rows: Vec<Vec<String>> =
            (0..df.shape().0).map(|i| df.get_row(i).unwrap()).collect();

        let shuffled = df.shuffle().unwrap();
        for i in 0..shuffled.shape().0 {
            let row = shuffled.get_row(i).unwrap();
            assert!(
                original_rows.contains(&row),
                "Row not found in original: {:?}",
                row
            );
        }
    }

    #[test]
    fn test_shuffle_single_row() {
        let a = Series::from("x", vec![42i32]);
        let df = DataFrame::new(vec![a]).unwrap();
        let shuffled = df.shuffle().unwrap();
        assert_eq!(shuffled.shape(), (1, 1));
        assert_eq!(shuffled.get_row(0).unwrap()[0], "42");
    }

    #[test]
    fn test_shuffle_empty_dataframe() {
        let df = DataFrame::new(vec![]).unwrap();
        let shuffled = df.shuffle().unwrap();
        assert_eq!(shuffled.shape(), (0, 0));
    }

    #[test]
    fn test_shuffle_zero_rows() {
        let a: Series = Series::from("x", Vec::<i32>::new());
        let df = DataFrame::new(vec![a]).unwrap();
        let shuffled = df.shuffle().unwrap();
        assert_eq!(shuffled.shape(), (0, 1));
    }

    #[test]
    fn test_shuffle_single_column() {
        let a = Series::from("x", vec![1i32, 2, 3, 4, 5, 6, 7, 8]);
        let df = DataFrame::new(vec![a]).unwrap();
        let shuffled = df.shuffle().unwrap();
        assert_eq!(shuffled.shape(), df.shape());
        let mut orig: Vec<String> = (0..df.shape().0)
            .map(|i| df.get_row(i).unwrap()[0].clone())
            .collect();
        let mut shuf: Vec<String> = (0..shuffled.shape().0)
            .map(|i| shuffled.get_row(i).unwrap()[0].clone())
            .collect();
        orig.sort();
        shuf.sort();
        assert_eq!(orig, shuf);
    }

    #[test]
    fn test_shuffle_with_bool_column() {
        let a = Series::from("num", vec![1i32, 2, 3]);
        let b = Series::from("flag", vec![true, false, true]);
        let df = DataFrame::new(vec![a, b]).unwrap();
        let shuffled = df.shuffle().unwrap();
        assert_eq!(shuffled.shape(), (3, 2));

        let orig_rows: Vec<Vec<String>> =
            (0..df.shape().0).map(|i| df.get_row(i).unwrap()).collect();
        for i in 0..shuffled.shape().0 {
            assert!(orig_rows.contains(&shuffled.get_row(i).unwrap()));
        }
    }
}

use std::fmt;
use arrow::array::Array;
use crate::{CrossbowError, Series};

#[cfg(test)]
mod unit_test;

#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: Vec<Series>,
}

impl DataFrame {
    pub fn new(columns: Vec<Series>) -> Result<Self, CrossbowError> {
        if columns.is_empty() {
            return Ok(DataFrame { columns });
        }

        let first_len = columns[0].len();
        if !columns.iter().all(|s| s.len() == first_len) {
            return Err(CrossbowError::MismatchedColumnLengths);
        }
        let mut names = std::collections::HashSet::new();
        for s in &columns {
            if !names.insert(s.name()) {
                return Err(CrossbowError::DuplicateColumnName(s.name().to_string()));
            }
        }

        Ok(DataFrame { columns })
    }

    pub fn shape(&self) -> (usize, usize) {
        if self.columns.is_empty() {
            (0, 0)
        } else {
            (self.columns[0].len(), self.columns.len())
        }
    }

    pub fn get_column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|s| s.name()).collect()
    }

    pub fn select(&self, name: &str) -> Result<&Series, CrossbowError> {
        self.columns
            .iter()
            .find(|s| s.name() == name)
            .ok_or_else(|| CrossbowError::ColumnNotFound(name.to_string()))
    }

    pub fn columns(&self) -> &[Series] {
        &self.columns
    }

    pub fn get_row(&self, index: usize) -> Result<Vec<String>, CrossbowError> {
        if index >= self.shape().0 {
            return Err(CrossbowError::IndexOutOfBounds(index));
        }

        let mut row = Vec::new();
        for series in &self.columns {
            row.push(series.get_value_as_string(index));
        }
        Ok(row)
    }

    pub fn get_rows(&self, indices: &[usize]) -> Result<Vec<Vec<String>>, CrossbowError> {
        let (n_rows, _) = self.shape();
        for &idx in indices {
            if idx >= n_rows {
                return Err(CrossbowError::IndexOutOfBounds(idx));
            }
        }

        let mut rows = Vec::new();
        for &idx in indices {
            let mut row = Vec::new();
            for series in &self.columns {
                row.push(series.get_value_as_string(idx));
            }
            rows.push(row);
        }
        Ok(rows)
    }

    pub fn add_column(&self, series: Series) -> Result<DataFrame, CrossbowError> {
        if series.len() != self.shape().0 {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        if self.get_column_names().contains(&series.name()) {
            return Err(CrossbowError::DuplicateColumnName(series.name().to_string()));
        }

        let mut new_columns = self.columns.clone();
        new_columns.push(series);
        DataFrame::new(new_columns)
    }

    pub fn remove_column(&self, name: &str) -> Result<DataFrame, CrossbowError> {
        let new_columns: Vec<Series> = self.columns
            .iter()
            .filter(|s| s.name() != name)
            .cloned()
            .collect();

        if new_columns.len() == self.columns.len() {
            return Err(CrossbowError::ColumnNotFound(name.to_string()));
        }

        DataFrame::new(new_columns)
    }

    pub fn rename_column(&self, old_name: &str, new_name: &str) -> Result<DataFrame, CrossbowError> {
        if !self.get_column_names().contains(&old_name) {
            return Err(CrossbowError::ColumnNotFound(old_name.to_string()));
        }

        if self.get_column_names().contains(&new_name) && old_name != new_name {
            return Err(CrossbowError::DuplicateColumnName(new_name.to_string()));
        }

        let new_columns: Vec<Series> = self.columns
            .iter()
            .map(|s| {
                if s.name() == old_name {
                    Series::new(new_name, s.data().clone())
                } else {
                    s.clone()
                }
            })
            .collect();

        DataFrame::new(new_columns)
    }
}

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
                    filtered_columns.push(Series::new(series.name(), std::sync::Arc::new(arr)));
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
                    filtered_columns.push(Series::new(series.name(), std::sync::Arc::new(arr)));
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
                    filtered_columns.push(Series::new(series.name(), std::sync::Arc::new(arr)));
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

impl DataFrame {
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
                    let mut b = arrow::array::Int32Builder::new();
                    for &idx in &indices {
                        if arr.is_valid(idx) {
                            b.append_value(arr.value(idx));
                        } else {
                            b.append_null();
                        }
                    }
                    sorted_columns.push(Series::new(series.name(), std::sync::Arc::new(b.finish())));
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
                    sorted_columns.push(Series::new(series.name(), std::sync::Arc::new(b.finish())));
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
                    sorted_columns.push(Series::new(series.name(), std::sync::Arc::new(b.finish())));
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

impl DataFrame {
    pub fn group_by(&self, column_name: &str) -> Result<GroupedDataFrame, CrossbowError> {
        let group_col = self.select(column_name)?;
        let mut groups: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();

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

#[derive(Debug, Clone)]
pub struct GroupedDataFrame {
    dataframe: DataFrame,
    group_column: String,
    groups: std::collections::HashMap<String, Vec<usize>>,
}

impl GroupedDataFrame {
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

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const HEAD: usize = 5;
        const TAIL: usize = 5;
        let (n_rows, n_cols) = self.shape();

        if n_cols == 0 {
            return write!(f, "DataFrame (0 columns, 0 rows)");
        }

        let mut col_widths: Vec<usize> = self.columns.iter().map(|s| s.name().len()).collect();
        let rows_to_display: Vec<usize> = if n_rows > HEAD + TAIL {
            (0..HEAD).chain(n_rows - TAIL..n_rows).collect()
        } else {
            (0..n_rows).collect()
        };

        for (i, series) in self.columns.iter().enumerate() {
            for row_idx in &rows_to_display {
                let val_str_len = series.get_value_as_string(*row_idx).len();
                if val_str_len > col_widths[i] {
                    col_widths[i] = val_str_len;
                }
            }
        }

        write!(f, "┌")?;
        for width in &col_widths {
            write!(f, "─{:-<width$}─", "")?;
        }
        writeln!(f, "┐")?;

        write!(f, "│")?;
        for (i, series) in self.columns.iter().enumerate() {
            write!(f, " {:^width$} │", series.name(), width = col_widths[i])?;
        }
        writeln!(f)?;

        write!(f, "├")?;
        for width in &col_widths {
            write!(f, "─{:-<width$}─", "")?;
        }
        writeln!(f, "┤")?;

        if n_rows > HEAD + TAIL {
            for i in 0..HEAD {
                write!(f, "│")?;
                for (j, series) in self.columns.iter().enumerate() {
                    write!(
                        f,
                        " {:<width$} │",
                        series.get_value_as_string(i),
                        width = col_widths[j]
                    )?;
                }
                writeln!(f)?;
            }
            write!(f, "│")?;
            for width in &col_widths {
                write!(f, " {:^width$} │", "...", width = *width)?;
            }
            writeln!(f)?;
            for i in n_rows - TAIL..n_rows {
                write!(f, "│")?;
                for (j, series) in self.columns.iter().enumerate() {
                    write!(
                        f,
                        " {:<width$} │",
                        series.get_value_as_string(i),
                        width = col_widths[j]
                    )?;
                }
                writeln!(f)?;
            }
        } else {
            for i in 0..n_rows {
                write!(f, "│")?;
                for (j, series) in self.columns.iter().enumerate() {
                    write!(
                        f,
                        " {:<width$} │",
                        series.get_value_as_string(i),
                        width = col_widths[j]
                    )?;
                }
                writeln!(f)?;
            }
        }

        write!(f, "└")?;
        for width in &col_widths {
            write!(f, "─{:-<width$}─", "")?;
        }
        writeln!(f, "┘")?;

        write!(f, "Shape: ({}, {})", n_rows, n_cols)?;

        Ok(())
    }
}

use std::fmt;
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

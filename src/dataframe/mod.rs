//! DataFrame — the primary tabular data structure.
//!
//! A `DataFrame` is an ordered collection of named [`Series`] (columns),
//! all of equal length. It supports column management, row access,
//! filtering, sorting, grouping, and aggregation. Use [`DataFrame::new`] to
//! construct one, or [`crate::read_csv`] / [`crate::read_parquet`] to load
//! from files.

use std::fmt;
use crate::{CrossbowError, Series};

pub mod filter;
pub mod sort;
pub mod groupby;
pub mod join;

#[cfg(test)]
mod unit_test;

/// A tabular collection of named, equal-length columns ([`Series`]).
///
/// Use [`DataFrame::new`] to construct one from `Vec<Series>`.
/// Supports column management, row access, filtering, sorting, and grouping.
///
/// # Examples
///
/// ```
/// use crossbow::{DataFrame, Series};
///
/// let a = Series::from("a", vec![1i32, 2, 3]);
/// let b = Series::from("b", vec!["x", "y", "z"]);
/// let df = DataFrame::new(vec![a, b]).unwrap();
/// assert_eq!(df.shape(), (3, 2));
/// ```
#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: Vec<Series>,
}

impl DataFrame {
    /// Creates a new `DataFrame` from a vector of `Series`.
    ///
    /// All `Series` must have the same length. Column names must be unique.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("a", vec![1i32, 2, 3]);
    /// let b = Series::from("b", vec![4.0, 5.0, 6.0]);
    /// let df = DataFrame::new(vec![a, b]).unwrap();
    /// assert_eq!(df.shape(), (3, 2));
    /// ```
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

    /// Returns the shape as `(rows, columns)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("a", vec![1i32, 2, 3]);
    /// let df = DataFrame::new(vec![a]).unwrap();
    /// assert_eq!(df.shape(), (3, 1));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        if self.columns.is_empty() {
            (0, 0)
        } else {
            (self.columns[0].len(), self.columns.len())
        }
    }

    /// Returns a vector of all column names in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("x", vec![1i32]);
    /// let b = Series::from("y", vec![2i32]);
    /// let df = DataFrame::new(vec![a, b]).unwrap();
    /// assert_eq!(df.get_column_names(), vec!["x", "y"]);
    /// ```
    pub fn get_column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|s| s.name()).collect()
    }

    /// Selects a column by name. Returns [`CrossbowError::ColumnNotFound`] if missing.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("a", vec![1i32, 2, 3]);
    /// let df = DataFrame::new(vec![a]).unwrap();
    /// let col = df.select("a").unwrap();
    /// assert_eq!(col.len(), 3);
    /// ```
    pub fn select(&self, name: &str) -> Result<&Series, CrossbowError> {
        self.columns
            .iter()
            .find(|s| s.name() == name)
            .ok_or_else(|| CrossbowError::ColumnNotFound(name.to_string()))
    }

    /// Returns a reference to all columns in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("a", vec![1i32]);
    /// let df = DataFrame::new(vec![a]).unwrap();
    /// assert_eq!(df.columns().len(), 1);
    /// ```
    pub fn columns(&self) -> &[Series] {
        &self.columns
    }

    /// Returns a single row as `Vec<String>`. Index out of range returns an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("a", vec![10i32]);
    /// let b = Series::from("b", vec!["hello"]);
    /// let df = DataFrame::new(vec![a, b]).unwrap();
    /// let row = df.get_row(0).unwrap();
    /// assert_eq!(row, vec!["10", "\"hello\""]);
    /// ```
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

    /// Returns multiple rows by index. Fails if any index is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("a", vec![1i32, 2, 3]);
    /// let df = DataFrame::new(vec![a]).unwrap();
    /// let rows = df.get_rows(&[0, 2]).unwrap();
    /// assert_eq!(rows.len(), 2);
    /// ```
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

    /// Adds a new column. Must have the same row count and a unique name.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("a", vec![1i32, 2]);
    /// let df = DataFrame::new(vec![a]).unwrap();
    /// let b = Series::from("b", vec![3.0, 4.0]);
    /// let df2 = df.add_column(b).unwrap();
    /// assert_eq!(df2.shape(), (2, 2));
    /// ```
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

    /// Removes a column by name. Returns error if the column does not exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("a", vec![1i32]);
    /// let b = Series::from("b", vec![2i32]);
    /// let df = DataFrame::new(vec![a, b]).unwrap();
    /// let df2 = df.remove_column("a").unwrap();
    /// assert_eq!(df2.shape(), (1, 1));
    /// ```
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

    /// Renames a column. Fails if `old_name` is missing or `new_name` is taken.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbow::{DataFrame, Series};
    ///
    /// let a = Series::from("old", vec![1i32]);
    /// let df = DataFrame::new(vec![a]).unwrap();
    /// let df2 = df.rename_column("old", "new").unwrap();
    /// assert_eq!(df2.get_column_names(), vec!["new"]);
    /// ```
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

use std::fmt;

use crate::{CrossbowError, Series};

#[cfg(test)]
mod unit_test;
/// This struct represents a DataFrame, which is a collection of Series (columns).
///
/// each Series in the DataFrame must have the same length.
/// Each Series must have a unique name.
#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: Vec<Series>,
}

impl DataFrame {
    /// Creates a new DataFrame from a vector of Series.
    ///     
    /// # Errors
    /// Returns a `CrossbowError` if:
    /// - The columns have mismatched lengths.
    /// - There are duplicate column names.
    ///
    /// # Example
    /// ```
    /// # use crossbow::series::Series;
    /// # use crossbow::dataframe::DataFrame;
    ///
    /// let s1 = Series::from("col A", vec![1, 2, 3]);
    /// let s2 = Series::from("col B", vec!["a", "b", "c"]);
    /// let df = DataFrame::new(vec![s1, s2]).unwrap();
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

    /// Returns the shape of the DataFrame as (number of rows, number of columns).
    /// If the DataFrame is empty, returns (0, 0).
    ///
    /// # Example
    /// ```
    /// use crossbow::series::Series;
    /// use crossbow::dataframe::DataFrame;
    ///  
    /// let s1 = Series::from("col A", vec![1, 2, 3]);
    /// let s2 = Series::from("col B", vec!["a", "b", "c"]);
    ///
    /// let df = DataFrame::new(vec![s1, s2]).unwrap();
    /// assert_eq!(df.shape(), (3, 2));
    /// ```
    ///
    ///
    pub fn shape(&self) -> (usize, usize) {
        if self.columns.is_empty() {
            (0, 0)
        } else {
            (self.columns[0].len(), self.columns.len())
        }
    }

    /// Returns a vector of column names in the DataFrame.
    ///
    /// # Example
    /// ```
    /// use crossbow::series::Series;
    /// use crossbow::dataframe::DataFrame;
    ///     
    /// let s1 = Series::from("col A", vec![1, 2, 3]);
    /// let s2 = Series::from("col B", vec!["a", "b", "c"]);
    /// let df = DataFrame::new(vec![s1, s2]).unwrap();
    /// let col_names = df.get_column_names();
    ///
    /// assert_eq!(col_names, vec!["col A", "col B"]);
    /// ```
    pub fn get_column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|s| s.name()).collect()
    }

    /// Selects a column by name and returns a reference to the corresponding Series.
    ///
    /// # Errors
    /// Returns a `CrossbowError::ColumnNotFound` if the column name does not
    /// exist in the DataFrame.
    ///
    /// # Example
    /// ```
    /// use crossbow::series::Series;
    /// use crossbow::dataframe::DataFrame;
    ///    
    /// let s1 = Series::from("col A", vec![1, 2, 3]);
    /// let s2 = Series::from("col B", vec!["a", "
    /// b", "c"]);
    ///
    /// let df = DataFrame::new(vec![s1.clone(), s2.clone()]).unwrap();
    /// let col_a = df.select("col A").unwrap();
    ///
    /// assert_eq!(col_a.name(), "col A");
    /// assert_eq!(col_a.len(), 3);
    /// ```
    pub fn select(&self, name: &str) -> Result<&Series, CrossbowError> {
        self.columns
            .iter()
            .find(|s| s.name() == name)
            .ok_or_else(|| CrossbowError::ColumnNotFound(name.to_string()))
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

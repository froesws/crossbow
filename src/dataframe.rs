// Removed `use std::fmt;`
use std::path::Path;

// Use our own error and series
use crate::{CrossbowError, Series, types::DataType};

#[cfg(test)]
mod unit_tests;

/// This struct represents a collection of named columns.
///
/// It ensures all columns have the same length and unique names.
#[derive(Debug, Clone)]
pub struct DataFrame {
    // Use Vec<Series> to preserve column order.
    columns: Vec<Series>,
}

// --- Column Builders ---
// This enum holds the temporary vectors for each column.
enum ColumnBuilder {
    Boolean(Vec<Option<bool>>),
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    Float64(Vec<Option<f64>>),
    String(Vec<Option<String>>),
}

impl DataFrame {
    /// Creates a new collection from a vector of columns.
    ///
    /// # Errors
    /// Fails if:
    /// - Columns have different lengths.
    /// - Column names are duplicated.
    pub fn new(columns: Vec<Series>) -> Result<Self, CrossbowError> {
        if columns.is_empty() {
            // It's fine to create an empty DF.
            return Ok(DataFrame { columns });
        }

        // Check lengths.
        let first_len = columns[0].len();
        if !columns.iter().all(|s| s.len() == first_len) {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        // Check for duplicate names.
        let mut names = std::collections::HashSet::new();
        for s in &columns {
            if !names.insert(s.name()) {
                return Err(CrossbowError::DuplicateColumnName(s.name().to_string()));
            }
        }

        Ok(DataFrame { columns })
    }

    /// Returns the shape (rows, columns) of the data.
    pub fn shape(&self) -> (usize, usize) {
        if self.columns.is_empty() {
            (0, 0)
        } else {
            // All columns have the same length, so just grab the first.
            (self.columns[0].len(), self.columns.len())
        }
    }

    /// Returns a vector of all column names in order.
    pub fn get_column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|s| s.name()).collect()
    }

    /// Selects a single column by its name.
    ///
    /// # Errors
    /// Fails if the column name does not exist.
    pub fn select(&self, name: &str) -> Result<&Series, CrossbowError> {
        // Find the series with the matching name.
        self.columns
            .iter()
            .find(|s| s.name() == name)
            .ok_or_else(|| CrossbowError::ColumnNotFound(name.to_string()))
    }

    /// Filters the entire collection of columns using a boolean mask.
    ///
    /// This will return a new collection where each column
    /// has been filtered by the provided mask.
    pub fn filter(&self, mask: &Series) -> Result<Self, CrossbowError> {
        let (n_rows, _) = self.shape();

        // Quick check on length before we iterate.
        if n_rows != mask.len() {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        // This will hold the new, filtered columns.
        let mut new_columns = Vec::with_capacity(self.columns.len());

        // Filter each column.
        for col in &self.columns {
            // The series filter method does all the hard work.
            let new_col = col.filter(mask)?;
            new_columns.push(new_col);
        }

        // Build the new DataFrame.
        // The `new` function already handles validation.
        DataFrame::new(new_columns)
    }

    // --- CSV Reading ---

    /// Reads a CSV file from a path.
    ///
    /// This will infer data types from the first 100 rows
    /// and then read the entire file into memory.
    pub fn read_csv<P: AsRef<Path>>(path: P) -> Result<Self, CrossbowError> {
        let mut rdr = csv::Reader::from_path(&path)?;
        let headers = rdr.headers()?.clone();

        // --- Type Inference ---
        // Read the first 100 records to guess types.
        let mut records = Vec::new();
        for result in rdr.records().take(100) {
            records.push(result?);
        }

        let mut data_types: Vec<DataType> = Vec::with_capacity(headers.len());
        for col_idx in 0..headers.len() {
            // Start with the most restrictive type and widen.
            let mut dtype = DataType::Boolean; // 1. Try Boolean

            for record in &records {
                let field = &record[col_idx];
                if field.is_empty() {
                    continue;
                } // Ignore empty fields for inference

                if dtype == DataType::Boolean && field.parse::<bool>().is_err() {
                    dtype = DataType::Int64;
                }
                if dtype == DataType::Int64 && field.parse::<i64>().is_err() {
                    dtype = DataType::Float64;
                }
                if dtype == DataType::Float64 && field.parse::<f64>().is_err() {
                    dtype = DataType::String;
                    break; // Can't get any wider.
                }
            }
            data_types.push(dtype);
        }

        let mut builders: Vec<ColumnBuilder> = data_types
            .iter()
            .map(|dtype| {
                match dtype {
                    DataType::Boolean => ColumnBuilder::Boolean(Vec::new()),
                    DataType::Int64 => ColumnBuilder::Int64(Vec::new()),
                    DataType::Float64 => ColumnBuilder::Float64(Vec::new()),
                    DataType::String => ColumnBuilder::String(Vec::new()),
                    DataType::Int32 => ColumnBuilder::Int32(Vec::new()), // Handle this case
                }
            })
            .collect();

        // --- File Parsing ---

        // First, process the records we already read for inference.
        for record in &records {
            for (col_idx, builder) in builders.iter_mut().enumerate() {
                let field = &record[col_idx];
                Self::parse_and_push(field, builder);
            }
        }

        // Now, reset the reader and process the rest of the file.
        let mut rdr = csv::Reader::from_path(&path)?;
        for result in rdr.records().skip(records.len()) {
            // Skip what we already parsed
            let record = result?;
            for (col_idx, builder) in builders.iter_mut().enumerate() {
                let field = &record[col_idx];
                Self::parse_and_push(field, builder);
            }
        }

        // --- Final Assembly ---
        // Convert all builders into Series.
        let mut columns = Vec::new();
        for (col_idx, builder) in builders.into_iter().enumerate() {
            let name = &headers[col_idx];
            let series = match builder {
                ColumnBuilder::Boolean(v) => Series::new_bool(name, v),
                ColumnBuilder::Int32(v) => Series::new_int32(name, v),
                ColumnBuilder::Int64(v) => Series::new_int64(name, v),
                ColumnBuilder::Float64(v) => Series::new_f64(name, v),
                ColumnBuilder::String(v) => Series::new_string(name, v),
            };
            columns.push(series);
        }

        // `new` will validate all lengths.
        DataFrame::new(columns)
    }

    /// Internal helper for the CSV reader.
    /// Parses a string field and pushes it into the correct builder.
    fn parse_and_push(field: &str, builder: &mut ColumnBuilder) {
        // Empty strings become nulls.
        if field.is_empty() {
            match builder {
                ColumnBuilder::Boolean(v) => v.push(None),
                ColumnBuilder::Int32(v) => v.push(None),
                ColumnBuilder::Int64(v) => v.push(None),
                ColumnBuilder::Float64(v) => v.push(None),
                ColumnBuilder::String(v) => v.push(None),
            }
            return;
        }

        // Parse based on the builder's type.
        // `ok()` converts a `Result` to `Option`, which is what we want.
        match builder {
            ColumnBuilder::Boolean(v) => v.push(field.parse::<bool>().ok()),
            ColumnBuilder::Int32(v) => v.push(field.parse::<i32>().ok()),
            ColumnBuilder::Int64(v) => v.push(field.parse::<i64>().ok()),
            ColumnBuilder::Float64(v) => v.push(field.parse::<f64>().ok()),
            ColumnBuilder::String(v) => v.push(Some(field.to_string())),
        }
    }
}

// Removed the `impl fmt::Display for DataFrame` block.

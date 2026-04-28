//! I/O module - input/output operations
//! Organized by data format: CSV and Parquet

pub mod csv;     // CSV file reading and writing
pub mod parquet; // Parquet file reading and writing

// Placeholder I/O functions (to be refactored into csv.rs and parquet.rs)
// For now, these are stubs - the actual implementations remain in the legacy io.rs pattern

use crate::{DataFrame, error::CrossbowError};

/// Reads a CSV file and returns a DataFrame.
///
/// This function attempts to infer the types of columns from the data.
pub fn read_csv(_path: &str) -> Result<DataFrame, CrossbowError> {
    Err(CrossbowError::OperationNotSupported(
        "CSV reading moved - use crate::read_csv or see src/dataframe.rs".to_string()
    ))
}

/// Reads a CSV file with explicit type inference.
pub fn read_csv_infer(_path: &str) -> Result<DataFrame, CrossbowError> {
    Err(CrossbowError::OperationNotSupported(
        "CSV reading moved - use crate::read_csv_infer or see src/dataframe.rs".to_string()
    ))
}

/// Reads a Parquet file and returns a DataFrame.
pub fn read_parquet(_path: &str) -> Result<DataFrame, CrossbowError> {
    Err(CrossbowError::OperationNotSupported(
        "Parquet reading not yet implemented".to_string()
    ))
}

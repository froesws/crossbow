use arrow::error::ArrowError;
use std::io;

/// Central error type for all crossbow operations.
///
/// Every fallible operation in the crate returns `Result<_, CrossbowError>`.
/// Implements `Display`, `Debug`, and `From<ArrowError>` + `From<io::Error>`.
#[derive(Debug)]
pub enum CrossbowError {
    /// Columns in a `DataFrame` have different row counts.
    MismatchedColumnLengths,
    /// Two or more columns share the same name.
    DuplicateColumnName(String),
    /// Referenced column does not exist in the `DataFrame`.
    ColumnNotFound(String),
    /// The requested operation is not implemented for this data type or context.
    OperationNotSupported(String),
    /// A filesystem or I/O error occurred.
    IoError(String),
    /// Arrow schema does not match expectations.
    SchemaMismatch(String),
    /// Integer arithmetic overflow (e.g. `i32::MAX + 1`).
    ArithmeticOverflow(String),
    /// A function received a Series whose Arrow data type is incompatible.
    TypeMismatch(String),
    /// The provided schema is invalid.
    InvalidSchema(String),
    /// Attempted division or modulo by zero.
    DivisionByZero,
    /// A row or column index is out of range.
    IndexOutOfBounds(usize),
}

impl std::fmt::Display for CrossbowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrossbowError::MismatchedColumnLengths => {
                write!(f, "All columns in a DataFrame must have the same length.")
            }
            CrossbowError::DuplicateColumnName(name) => {
                write!(f, "Duplicate column name found: {}", name)
            }
            CrossbowError::ColumnNotFound(name) => write!(f, "Column not found: {}", name),
            CrossbowError::OperationNotSupported(msg) => {
                write!(f, "Operation not supported: {}", msg)
            }
            CrossbowError::IoError(msg) => write!(f, "I/O error: {}", msg),
            CrossbowError::SchemaMismatch(msg) => write!(f, "Schema mismatch: {}", msg),
            CrossbowError::ArithmeticOverflow(msg) => write!(f, "Arithmetic overflow: {}", msg),
            CrossbowError::TypeMismatch(msg) => write!(f, "Type mismatch: {}", msg),
            CrossbowError::InvalidSchema(msg) => write!(f, "Invalid schema: {}", msg),
            CrossbowError::DivisionByZero => write!(f, "Division by zero"),
            CrossbowError::IndexOutOfBounds(idx) => write!(f, "Index out of bounds: {}", idx),
        }
    }
}

impl From<ArrowError> for CrossbowError {
    fn from(err: ArrowError) -> Self {
        CrossbowError::OperationNotSupported(format!("Arrow error: {}", err))
    }
}

impl From<io::Error> for CrossbowError {
    fn from(err: io::Error) -> Self {
        CrossbowError::IoError(err.to_string())
    }
}

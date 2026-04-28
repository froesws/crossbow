use arrow::error::ArrowError;
use std::io;

#[derive(Debug)]
pub enum CrossbowError {
    MismatchedColumnLengths,
    DuplicateColumnName(String),
    ColumnNotFound(String),
    OperationNotSupported(String),
    IoError(String),
    SchemaMismatch(String),
    ArithmeticOverflow(String),
    TypeMismatch(String),
    InvalidSchema(String),
    DivisionByZero,
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

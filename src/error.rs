use arrow::error::ArrowError;

#[derive(Debug)]
pub enum CrossbowError {
    MismatchedColumnLengths,
    DuplicateColumnName(String),
    ColumnNotFound(String),
    OperationNotSupported(String),
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
        }
    }
}

impl From<ArrowError> for CrossbowError {
    fn from(err: ArrowError) -> Self {
        CrossbowError::OperationNotSupported(format!("Arrow error: {}", err))
    }
}

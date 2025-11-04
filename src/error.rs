#[derive(Debug)]
pub enum CrossbowError {
    MismatchedColumnLengths,
    DuplicateColumnName(String),
    ColumnNotFound(String),
    OperationNotSupported(String),
    MismatchedDataTypes(String, String),
    CsvError(csv::Error),
    IoError(std::io::Error),
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
            CrossbowError::MismatchedDataTypes(expected, found) => {
                write!(
                    f,
                    "Mismatched data types: expected {}, found {}",
                    expected, found
                )
            }
            CrossbowError::CsvError(error) => {
                write!(f, "CSV error: {}", error)
            }
            CrossbowError::IoError(error) => {
                write!(f, "I/O error: {}", error)
            }
        }
    }
}

// Added this implementation for std::io::Error
impl From<std::io::Error> for CrossbowError {
    fn from(err: std::io::Error) -> Self {
        CrossbowError::IoError(err)
    }
}

// Added this implementation for csv::Error
impl From<csv::Error> for CrossbowError {
    fn from(err: csv::Error) -> Self {
        CrossbowError::CsvError(err)
    }
}

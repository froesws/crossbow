// src/types.rs

/// Represents the logical data types supported by the library.
/// This is used for metadata and dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    Int32,
    Int64,
    Float64,
    Boolean,
    String,
}

// This enum replaces Arrow's `ArrayRef`.
// It's the internal, type-safe storage for our column data.
// Using Vec<Option<T>> is the simplest way to handle nulls.
#[derive(Debug, Clone)]
pub enum SeriesData {
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    Float64(Vec<Option<f64>>),
    Boolean(Vec<Option<bool>>),
    String(Vec<Option<String>>),
}

//! MIT License
//!
//! Copyright (c) [2025] [William Froes]
//! Project: [crossbow]
//! Developed at: [Uergs -- Universidade Estadual do Rio Grande do Sul]

// Core data structures (new modular organization - in development)
// Note: These are being created as part of SRP refactoring
// For now, the legacy modules (series, dataframe) are the main exports
pub mod core;

// Error handling
pub mod error;
pub use error::CrossbowError;

// Legacy modules (main implementation)
pub mod series;
pub use series::Series;

pub mod dataframe;
pub use dataframe::DataFrame;

// Operations modules (SRP-based organization - in development) 
pub mod ops;

// I/O operations
pub mod io;
pub use io::{read_csv, read_csv_infer, read_parquet};

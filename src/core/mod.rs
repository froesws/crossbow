//! Core data structures (new modular organization - in development)
//! Note: These are being created as part of SRP refactoring
//! For now, the legacy modules (series, dataframe) are the main exports

pub mod series;
pub use series::Series;

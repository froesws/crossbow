//! # crossbow
//!
//! A tabular data manipulation library built on Apache Arrow.
//!
//! crossbow provides [`DataFrame`] and [`Series`] types for loading, filtering,
//! sorting, joining, grouping, and aggregating structured data. It supports CSV
//! and Parquet I/O with automatic type inference.
//!
//! ## Quick start
//!
//! ```
//! use crossbow::{DataFrame, Series};
//!
//! let a = Series::from("a", vec![1i32, 2, 3]);
//! let b = Series::from("b", vec![4.0, 5.0, 6.0]);
//! let df = DataFrame::new(vec![a, b]).unwrap();
//! assert_eq!(df.shape(), (3, 2));
//! ```
//!
//! MIT License
//!
//! Copyright (c) [2025] [William Froes]
//! Project: [crossbow]
//! Developed at: [Uergs -- Universidade Estadual do Rio Grande do Sul]

pub mod utils;
pub mod error;
pub mod series;
pub mod dataframe;
pub mod io;

pub use error::CrossbowError;
pub use series::Series;
pub use series::IntoSeries;
pub use dataframe::DataFrame;
pub use io::csv::{read_csv, write_csv};
pub use io::parquet::{read_parquet, write_parquet};
pub use utils::expected_array;

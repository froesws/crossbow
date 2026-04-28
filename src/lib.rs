//! MIT License
//!
//! Copyright (c) [2025] [William Froes]
//! Project: [crossbow]
//! Developed at: [Uergs -- Universidade Estadual do Rio Grande do Sul]

pub mod error;
pub mod series;
pub mod dataframe;
pub mod io;

pub use error::CrossbowError;
pub use series::Series;
pub use dataframe::DataFrame;
pub use io::csv::{read_csv, write_csv};
pub use io::parquet::{read_parquet, write_parquet};

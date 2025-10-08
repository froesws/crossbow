//! MIT License
//!
//! Copyright (c) [2025] [William Froes]
//! Project: [crossbow]
//! Developed at: [Uergs -- Universidade Estadual do Rio Grande do Sul]

pub mod dataframe;
pub mod error;
pub mod series;

pub use self::dataframe::DataFrame;
pub use self::error::CrossbowError;
pub use self::series::Series;

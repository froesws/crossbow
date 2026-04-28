//! MIT License
//!
//! Copyright (c) [2025] [William Froes]
//! Project: [crossbow]
//! Developed at: [Uergs -- Universidade Estadual do Rio Grande do Sul]

// Error handling
pub mod error;
pub use error::CrossbowError;

// Legacy modules (main implementation)
pub mod series;
pub use series::Series;

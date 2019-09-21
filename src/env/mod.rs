use std::time::Duration;

pub mod backpropagation;
pub mod genetic;

pub use self::{backpropagation::Backpropagation, genetic::Genetic};

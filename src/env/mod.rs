use std::time::Duration;

pub mod genetic;

pub use self::genetic::Genetic;

#[derive(Copy, Clone, Debug, PartialEq)]
/// When should the network optimization process be stopped
pub enum EndCondition {
    /// stop after `n` iterations
    Iterations { n: u32 },
    /// stop after `delta` time has passed
    Time { delta: Duration },
    /// stop once the total error is less than `err`
    MinError { err: f32 },
    /// stop once the change of total error is less than `delta` in the last `n` generations
    DeltaError { delta: f32, n: u32 },
    /// stop once `stop` is written to the console
    ConsoleInput,
}

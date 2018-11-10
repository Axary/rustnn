//! all errors used in this crate


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CreationError {
    NotEnoughLayers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RunError {
    WrongInputCount,
}

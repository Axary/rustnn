//! a collection of useful functions which are somewhat relevant to neural networks

/// calculates a binary sigmoidal function of `t`, amplified by a factor `a` and returning a value between `0.0` and `1.0`.
/// 
/// `sig(t, a) = 1 / (1 + e ^ (-t * a))`
#[inline(always)]
pub fn binary_sigmoid(t: f32, a: f32) -> f32 {
    1.0 / (1.0 + (-t * a).exp())
}

/// calculates a bipolar sigmoidal function of `t`, amplified by a factor `a`, returning a value between `-1.0` and `1.0`.
/// 
/// `sig(t, a) = 2 / (1 + e ^ (-t * a)) -1`
#[inline(always)]
pub fn bipolar_sigmoid(t: f32, a: f32) -> f32 {
    2.0 / (1.0 + (-t * a).exp()) - 1.0
}

/// calculates the squared error of `actual`.
/// 
/// `E(ideal, actual) = sum((ideal - actual)^2)`
pub fn squared_error(ideal: &[f32], actual: &[f32]) -> f32 {
    ideal.iter().zip(actual).fold(0.0, |total, (i, a)| (total + (i - a).powi(2)))
}
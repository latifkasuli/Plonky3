//! DEEP-ALI out-of-domain sampling error.
//!
//! [`deep_ali_error`] retains the historical `soundcalc` diagnostic:
//!
//! `ε = L⁺ · (max_deg · (k + max_combo − 1) + (k − 1)) / |F|`.
//!
//! It is not the complete source expression. [2024/1553] Theorem 2 uses
//! `L²`, both Theorems 2 and 3 use `|G| - |D union H|`, and both include a
//! second quotient-segment degree branch. Use
//! [`deep_ali_round_two_error_ldr`] or [`deep_ali_round_two_error_udr`] for
//! that arithmetic after separately establishing the implementation-to-source
//! parameter correspondence.

use core::cmp::max;

use libm::{log2, nextafter};
use num_bigint::BigUint;

use crate::error::ErrorBits;
use crate::shape::{InstanceShape, StarkAirParams};

/// `-log2(ε_DEEP)` in bits. Returns 0 bits if inputs are degenerate.
pub fn deep_ali_error(air: &StarkAirParams, shape: &InstanceShape, list_size: f64) -> ErrorBits {
    if shape.modulus_bits == 0 || !list_size.is_finite() || list_size <= 0.0 {
        return ErrorBits::from_log2(0.0);
    }
    let k = (1u64 << shape.log_trace_length) as f64;
    let max_deg = air.max_constraint_degree.max(1) as f64;
    let combo = air.max_combo as f64;
    let factor = (max_deg * (k + combo - 1.0) + (k - 1.0)).max(1.0);
    let bits = shape.modulus_bits as f64 - log2(list_size) - log2(factor);
    ErrorBits::from_log2(bits.max(0.0))
}

/// Explicit inputs to the source-form DEEP-ALI round-two error in
/// [2024/1553] Theorems 2 and 3.
///
/// This struct records the paper's parameters; constructing it does not
/// establish that a concrete protocol uses the same quotient decomposition.
/// In particular, callers must separately justify `quotient_segment_count`
/// and `quotient_segment_degree_bound` for their implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeepAliRoundTwoParams {
    /// Cardinality of the DEEP challenge field `G`.
    pub field_cardinality: BigUint,
    /// Exact size of `D union H`, from which the DEEP point is excluded.
    pub evaluation_trace_domain_union_size: usize,
    /// Source low-degree bound `k`.
    pub low_degree_bound: usize,
    /// Source expanded low-degree bound `k+`.
    pub expanded_low_degree_bound: usize,
    /// Number `f` of quotient segments.
    pub quotient_segment_count: usize,
    /// Degree bound `ell` of each quotient segment.
    pub quotient_segment_degree_bound: usize,
}

/// Source-form DEEP-ALI round-two error in the unique-decoding regime
/// ([2024/1553] Theorem 3).
///
/// Returns `None` rather than manufacturing a number when a count is zero,
/// the expanded degree is smaller than `k`, or `|D union H| >= |G|`.
pub fn deep_ali_round_two_error_udr(
    air: &StarkAirParams,
    params: &DeepAliRoundTwoParams,
) -> Option<ErrorBits> {
    deep_ali_round_two_error(air, params, 1.0, false)
}

/// Source-form DEEP-ALI round-two error in the list-decoding regime
/// ([2024/1553] Theorem 2).
///
/// The theorem's list-size contribution is quadratic. `list_size` must be
/// the list-size bound justified for the same proximity regime as the other
/// source parameters. Returns `None` for invalid or degenerate inputs.
pub fn deep_ali_round_two_error_ldr(
    air: &StarkAirParams,
    params: &DeepAliRoundTwoParams,
    list_size: f64,
) -> Option<ErrorBits> {
    deep_ali_round_two_error(air, params, list_size, true)
}

fn deep_ali_round_two_error(
    air: &StarkAirParams,
    params: &DeepAliRoundTwoParams,
    list_size: f64,
    square_list_size: bool,
) -> Option<ErrorBits> {
    if air.max_constraint_degree == 0
        || params.low_degree_bound == 0
        || params.expanded_low_degree_bound < params.low_degree_bound
        || params.quotient_segment_count == 0
        || params.quotient_segment_degree_bound == 0
        || !list_size.is_finite()
        || list_size <= 0.0
    {
        return None;
    }

    let excluded = BigUint::from(params.evaluation_trace_domain_union_size);
    if params.field_cardinality <= excluded {
        return None;
    }
    let denominator = &params.field_cardinality - excluded;

    let k = BigUint::from(params.low_degree_bound);
    let k_minus_one = BigUint::from(params.low_degree_bound - 1);
    let k_plus_minus_one = BigUint::from(params.expanded_low_degree_bound - 1);
    let first_branch = BigUint::from(air.max_constraint_degree) * &k_plus_minus_one + &k_minus_one;
    let second_branch = k
        + BigUint::from(params.quotient_segment_count - 1)
            * BigUint::from(params.quotient_segment_degree_bound)
        + k_plus_minus_one;
    let selected = max(first_branch, second_branch);

    let list_power = if square_list_size { 2.0 } else { 0.0 };
    let list_size_log_upper = nextafter(log2(list_size), f64::INFINITY);
    let rational_bits =
        log2_biguint_lower_bound(&denominator) - log2_biguint_upper_bound(&selected);
    let bits = rational_bits - list_power * list_size_log_upper;
    Some(ErrorBits::from_log2(bits.max(0.0)))
}

/// Conservative floating-point `log2`: truncate to the leading 52 bits
/// before converting to `f64`. The result never exceeds the exact logarithm,
/// so its use in `log2(denominator) - log2(numerator)` cannot overstate the
/// denominator contribution.
fn log2_biguint_lower_bound(value: &BigUint) -> f64 {
    let bits = value.bits();
    debug_assert!(bits > 0);
    let shift = bits.saturating_sub(52);
    let leading = value >> shift;
    let leading_digits = leading.to_u64_digits();
    debug_assert_eq!(leading_digits.len(), 1);
    nextafter(
        log2(leading_digits[0] as f64) + shift as f64,
        f64::NEG_INFINITY,
    )
}

/// Conservative upper bound matching [`log2_biguint_lower_bound`]. Values
/// wider than 52 bits are truncated and then rounded up by one unit in the
/// retained prefix. Subtracting this result cannot overstate security bits.
fn log2_biguint_upper_bound(value: &BigUint) -> f64 {
    let bits = value.bits();
    debug_assert!(bits > 0);
    if bits <= 52 {
        let digits = value.to_u64_digits();
        debug_assert_eq!(digits.len(), 1);
        return nextafter(log2(digits[0] as f64), f64::INFINITY);
    }
    let shift = bits - 52;
    let leading = value >> shift;
    let leading_digits = leading.to_u64_digits();
    debug_assert_eq!(leading_digits.len(), 1);
    nextafter(
        log2((leading_digits[0] + 1) as f64) + shift as f64,
        f64::INFINITY,
    )
}

#[cfg(test)]
mod source_tests {
    use super::*;

    fn air() -> StarkAirParams {
        StarkAirParams {
            num_constraints: 7,
            max_constraint_degree: 3,
            max_combo: 2,
        }
    }

    fn params() -> DeepAliRoundTwoParams {
        DeepAliRoundTwoParams {
            field_cardinality: BigUint::from(1u8) << 128usize,
            evaluation_trace_domain_union_size: (1 << 20) + (1 << 16),
            low_degree_bound: 1 << 16,
            expanded_low_degree_bound: (1 << 16) + 2,
            quotient_segment_count: 4,
            quotient_segment_degree_bound: 1 << 16,
        }
    }

    #[test]
    fn source_form_selects_quotient_segment_branch() {
        let params = params();
        let result = deep_ali_round_two_error_udr(&air(), &params).unwrap();
        let first = 3 * (params.expanded_low_degree_bound - 1) + (params.low_degree_bound - 1);
        let second = params.low_degree_bound
            + (params.quotient_segment_count - 1) * params.quotient_segment_degree_bound
            + (params.expanded_low_degree_bound - 1);
        assert!(second > first);

        let denominator = 2f64.powi(128) - params.evaluation_trace_domain_union_size as f64;
        let expected = log2(denominator) - log2(second as f64);
        assert!(result.bits() <= expected);
        assert!((result.bits() - expected).abs() < 1e-12);
    }

    #[test]
    fn list_decoding_applies_quadratic_list_size() {
        let params = params();
        let list_size = 17.5;
        let udr = deep_ali_round_two_error_udr(&air(), &params).unwrap();
        let ldr = deep_ali_round_two_error_ldr(&air(), &params, list_size).unwrap();
        assert!((ldr.bits() - (udr.bits() - 2.0 * log2(list_size))).abs() < 1e-12);
    }

    #[test]
    fn source_form_fails_closed_on_invalid_inputs() {
        let mut invalid = params();
        invalid.field_cardinality = BigUint::from(invalid.evaluation_trace_domain_union_size);
        assert!(deep_ali_round_two_error_udr(&air(), &invalid).is_none());

        let mut invalid = params();
        invalid.expanded_low_degree_bound = invalid.low_degree_bound - 1;
        assert!(deep_ali_round_two_error_udr(&air(), &invalid).is_none());

        assert!(deep_ali_round_two_error_ldr(&air(), &params(), f64::NAN).is_none());
    }

    #[test]
    fn big_integer_rounding_is_conservative_in_both_directions() {
        let denominator = (BigUint::from(1u8) << 200usize) - BigUint::from(12345u32);
        let numerator = (BigUint::from(1u8) << 120usize) + BigUint::from(67890u32);

        let denominator_floor = log2_biguint_lower_bound(&denominator);
        let denominator_ceil = log2_biguint_upper_bound(&denominator);
        let numerator_floor = log2_biguint_lower_bound(&numerator);
        let numerator_ceil = log2_biguint_upper_bound(&numerator);

        assert!(denominator_floor <= denominator_ceil);
        assert!(numerator_floor <= numerator_ceil);
        assert!(denominator_floor - numerator_ceil <= denominator_ceil - numerator_floor);
        assert!(denominator_floor <= 200.0);
        assert!(denominator_ceil >= 200.0);
        assert!(numerator_floor <= 120.0);
        assert!(numerator_ceil > 120.0);
    }
}

//! AIR composition error.
//!
//! [`composition_error`] retains the conservative historical calculator
//! `L⁺ · num_constraints / 2^modulus_bits`.  The exact one-challenge root
//! count for `K = num_constraints` consecutive powers of `alpha` is instead
//! `(K - 1) / |EF|` per fixed non-zero constraint vector.  Use
//! [`single_alpha_constraint_combination_report`] when the exact challenge
//! field cardinality and the theorem-applicability boundary matter.
//!
//! Regime-independent — the list size L⁺ is passed in by the caller,
//! computed from the chosen proximity regime via [`crate::proximity`].

use libm::log2;
use num_bigint::BigUint;

use crate::error::ErrorBits;

/// `-log2(ε_ALI)` in bits. Returns 0 bits if inputs are degenerate.
pub fn composition_error(num_constraints: usize, list_size: f64, modulus_bits: usize) -> ErrorBits {
    if num_constraints == 0 || modulus_bits == 0 || !list_size.is_finite() || list_size <= 0.0 {
        return ErrorBits::from_log2(0.0);
    }
    let bits = modulus_bits as f64 - log2(list_size) - log2(num_constraints as f64);
    ErrorBits::from_log2(bits.max(0.0))
}

/// Exact root-count bookkeeping for Plonky3's single-`alpha` AIR fold.
///
/// For constraints emitted in one global order, Plonky3 evaluates
///
/// `C_fold(alpha) = sum_{i=0}^{K-1} alpha^(K-1-i) C_i`.
///
/// If the fixed coefficient vector `(C_0, ..., C_{K-1})` is non-zero, this is
/// a non-zero univariate polynomial of degree at most `K-1`.  Hence a uniform
/// `alpha` in the challenge field cancels it with probability at most
/// `(K-1)/|EF|`.  A union bound over a source-justified list of size `L` costs
/// `(K-1)L/|EF|`.
///
/// This report deliberately stops before that union bound: it does not accept
/// a caller-supplied list size and therefore cannot manufacture an affirmative
/// RbR or soundness result.  Callers must separately establish that the source
/// state function produces a non-zero coefficient polynomial for every
/// candidate and that the same list-size bound applies.  It also assumes an
/// ideal uniform challenge; it does not prove a Fiat-Shamir/ROM reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingleAlphaConstraintCombinationReport {
    pub num_constraints: usize,
    pub lowest_alpha_power: usize,
    pub highest_alpha_power: usize,
    pub schwartz_zippel_degree_bound: usize,
    pub challenge_field_cardinality: BigUint,
    pub per_candidate_error_numerator: usize,
    pub historical_calculator_numerator: usize,
    pub historical_calculator_factor_is_conservative: bool,
    pub independent_coefficients_assumed: bool,
    pub nonzero_coefficient_polynomial_required: bool,
    pub source_list_size_binding_established: bool,
    pub state_function_correspondence_established: bool,
    pub fiat_shamir_uniformity_established: bool,
    pub full_rbr_transfer_established: bool,
}

/// Build the exact single-`alpha` root-count report.
///
/// Returns `None` for fewer than two constraints or when the root-count
/// numerator is not strictly smaller than the challenge field.  These shapes
/// are refused rather than being reported as a meaningful positive-bit bound.
pub fn single_alpha_constraint_combination_report(
    num_constraints: usize,
    challenge_field_cardinality: BigUint,
) -> Option<SingleAlphaConstraintCombinationReport> {
    let schwartz_zippel_degree_bound = num_constraints.checked_sub(1)?;
    if schwartz_zippel_degree_bound == 0
        || challenge_field_cardinality <= BigUint::from(schwartz_zippel_degree_bound)
    {
        return None;
    }

    Some(SingleAlphaConstraintCombinationReport {
        num_constraints,
        lowest_alpha_power: 0,
        highest_alpha_power: schwartz_zippel_degree_bound,
        schwartz_zippel_degree_bound,
        challenge_field_cardinality,
        per_candidate_error_numerator: schwartz_zippel_degree_bound,
        historical_calculator_numerator: num_constraints,
        historical_calculator_factor_is_conservative: true,
        independent_coefficients_assumed: false,
        nonzero_coefficient_polynomial_required: true,
        source_list_size_binding_established: false,
        state_function_correspondence_established: false,
        fiat_shamir_uniformity_established: false,
        full_rbr_transfer_established: false,
    })
}

#[cfg(test)]
mod source_tests {
    use super::*;

    #[test]
    fn single_alpha_report_prices_the_exact_root_factor() {
        let report = single_alpha_constraint_combination_report(7, BigUint::from(101u8)).unwrap();
        assert_eq!(report.lowest_alpha_power, 0);
        assert_eq!(report.highest_alpha_power, 6);
        assert_eq!(report.schwartz_zippel_degree_bound, 6);
        assert_eq!(report.per_candidate_error_numerator, 6);
        assert_eq!(report.challenge_field_cardinality, BigUint::from(101u8));
        assert_eq!(report.historical_calculator_numerator, 7);
        assert!(report.historical_calculator_factor_is_conservative);
        assert!(!report.independent_coefficients_assumed);
        assert!(report.nonzero_coefficient_polynomial_required);
        assert!(!report.source_list_size_binding_established);
        assert!(!report.state_function_correspondence_established);
        assert!(!report.fiat_shamir_uniformity_established);
        assert!(!report.full_rbr_transfer_established);
    }

    #[test]
    fn single_alpha_report_fails_closed_on_degenerate_shapes() {
        assert!(single_alpha_constraint_combination_report(1, BigUint::from(101u8)).is_none());
        assert!(single_alpha_constraint_combination_report(7, BigUint::from(6u8)).is_none());
    }
}

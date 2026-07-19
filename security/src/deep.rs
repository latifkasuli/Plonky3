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
//! parameter correspondence. If the executed verifier samples from the full
//! challenge field and rejects only the trace domain, use
//! [`deep_ali_full_field_error_ldr`] or [`deep_ali_full_field_error_udr`] to
//! add the accepted forbidden-point event. Those functions are conditional on
//! an ideal uniform field challenge and do not establish Fiat-Shamir or the
//! quotient-segment proof transfer.

use core::cmp::max;

use libm::log2;
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

/// Exact degree bookkeeping for transferring the DEEP-ALI round-two
/// distinct-polynomial argument to disjoint-coset Lagrange selectors.
///
/// For `d` disjoint quotient domains of common size `h`, every selector
///
/// `L_i(X) = c_i * product_{j != i} Z_{H_j}(X)`
///
/// has degree `(d - 1)h`.  Consequently, arbitrary candidate chunks of
/// degree `<2h` recompose to degree `<(d + 1)h`.  This calculation does not
/// use the honest prover's zero-balance relation: RbR list candidates are
/// arbitrary low-degree codewords and cannot be assumed to satisfy it.
///
/// The source-style parameter translation uses `k=2N`, `k+=k+2`, `ell=h`,
/// and `f=d`.  The older `k=N`, `k+=N+2` translation is retained in the
/// report only as a checked negative control.  A successful report proves
/// this degree-envelope lemma; it does not establish the source theorem's
/// list-size regime, constraint-combination reduction, state-function
/// correspondence, or full RbR soundness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LagrangeCosetRbrDegreeReport {
    pub trace_domain_size: usize,
    pub quotient_chunk_domain_size: usize,
    pub quotient_chunk_count: usize,
    pub randomized_trace_degree_bound_exclusive: usize,
    pub randomized_chunk_degree_bound_exclusive: usize,
    pub selector_degree: usize,
    pub candidate_recomposition_degree_bound_exclusive: usize,
    pub source_low_degree_bound_k: usize,
    pub source_expanded_low_degree_bound_k_plus: usize,
    pub source_quotient_segment_count_f: usize,
    pub source_quotient_segment_length_ell: usize,
    pub source_candidate_recomposition_degree_bound_exclusive: usize,
    pub legacy_source_low_degree_bound_k: usize,
    pub legacy_source_expanded_low_degree_bound_k_plus: usize,
    pub legacy_candidate_recomposition_degree_bound_exclusive: usize,
    pub legacy_mapping_sufficient: bool,
    pub corrected_mapping_sufficient: bool,
    pub arbitrary_candidate_balance_assumed: bool,
    pub full_rbr_transfer_established: bool,
}

/// Build the checked Lagrange/coset degree-transfer report.
///
/// The concrete hiding construction currently uses equal quotient cosets with
/// `h=N` and randomizes both trace and quotient polynomials to degree `<2N`.
/// Shapes outside that exact regime fail closed instead of being silently
/// translated into the source theorem.
pub fn lagrange_coset_rbr_degree_report(
    trace_domain_size: usize,
    quotient_chunk_domain_size: usize,
    quotient_chunk_count: usize,
) -> Option<LagrangeCosetRbrDegreeReport> {
    if trace_domain_size < 2
        || quotient_chunk_domain_size != trace_domain_size
        || quotient_chunk_count <= 1
    {
        return None;
    }

    let randomized_trace_degree_bound_exclusive = trace_domain_size.checked_mul(2)?;
    let randomized_chunk_degree_bound_exclusive = quotient_chunk_domain_size.checked_mul(2)?;
    let selector_degree = quotient_chunk_count
        .checked_sub(1)?
        .checked_mul(quotient_chunk_domain_size)?;
    let candidate_recomposition_degree_bound_exclusive =
        selector_degree.checked_add(randomized_chunk_degree_bound_exclusive)?;

    let source_low_degree_bound_k = randomized_trace_degree_bound_exclusive;
    let source_expanded_low_degree_bound_k_plus = source_low_degree_bound_k.checked_add(2)?;
    let source_quotient_segment_count_f = quotient_chunk_count;
    let source_quotient_segment_length_ell = quotient_chunk_domain_size;
    let source_candidate_recomposition_degree_bound_exclusive =
        selector_degree.checked_add(source_expanded_low_degree_bound_k_plus)?;

    let legacy_source_low_degree_bound_k = trace_domain_size;
    let legacy_source_expanded_low_degree_bound_k_plus = trace_domain_size.checked_add(2)?;
    let legacy_candidate_recomposition_degree_bound_exclusive =
        selector_degree.checked_add(legacy_source_expanded_low_degree_bound_k_plus)?;

    let corrected_mapping_sufficient = candidate_recomposition_degree_bound_exclusive
        <= source_candidate_recomposition_degree_bound_exclusive;
    let legacy_mapping_sufficient = candidate_recomposition_degree_bound_exclusive
        <= legacy_candidate_recomposition_degree_bound_exclusive;
    if !corrected_mapping_sufficient || legacy_mapping_sufficient {
        return None;
    }

    Some(LagrangeCosetRbrDegreeReport {
        trace_domain_size,
        quotient_chunk_domain_size,
        quotient_chunk_count,
        randomized_trace_degree_bound_exclusive,
        randomized_chunk_degree_bound_exclusive,
        selector_degree,
        candidate_recomposition_degree_bound_exclusive,
        source_low_degree_bound_k,
        source_expanded_low_degree_bound_k_plus,
        source_quotient_segment_count_f,
        source_quotient_segment_length_ell,
        source_candidate_recomposition_degree_bound_exclusive,
        legacy_source_low_degree_bound_k,
        legacy_source_expanded_low_degree_bound_k_plus,
        legacy_candidate_recomposition_degree_bound_exclusive,
        legacy_mapping_sufficient,
        corrected_mapping_sufficient,
        arbitrary_candidate_balance_assumed: false,
        full_rbr_transfer_established: false,
    })
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

/// Conservative DEEP-ALI round-two error for an ideal uniform full-field
/// challenge in the unique-decoding regime.
///
/// The source theorem conditions on a challenge outside `D union H`. The
/// executed verifier rejects challenges in `H`, but an accepted challenge in
/// `(D union H) \\ H` is outside the theorem's sample space and is therefore
/// charged pessimistically with failure probability one. On the remaining
/// points, the source conditional error is weighted by their exact sampling
/// probability. Algebraically, if the source numerator is `A`, the resulting
/// bound is `(A + |D union H| - |H|) / |G|`.
///
/// This is a counting reduction only. It assumes an ideal uniform challenge
/// over `G` and that the source theorem applies on `G \\ (D union H)`; it does
/// not prove a Fiat-Shamir/ROM reduction or a quotient-segment correspondence.
pub fn deep_ali_full_field_error_udr(
    air: &StarkAirParams,
    params: &DeepAliRoundTwoParams,
    verifier_rejected_trace_domain_size: usize,
) -> Option<ErrorBits> {
    deep_ali_full_field_error(air, params, verifier_rejected_trace_domain_size, 1.0, false)
}

/// Conservative DEEP-ALI round-two error for an ideal uniform full-field
/// challenge in the list-decoding regime.
///
/// This applies the theorem's quadratic list-size factor using the same
/// directional power-of-two envelope as [`deep_ali_round_two_error_ldr`], then
/// adds the accepted forbidden-point event. See
/// [`deep_ali_full_field_error_udr`] for the reduction and its trust boundary.
pub fn deep_ali_full_field_error_ldr(
    air: &StarkAirParams,
    params: &DeepAliRoundTwoParams,
    verifier_rejected_trace_domain_size: usize,
    list_size: f64,
) -> Option<ErrorBits> {
    deep_ali_full_field_error(
        air,
        params,
        verifier_rejected_trace_domain_size,
        list_size,
        true,
    )
}

fn deep_ali_round_two_error(
    air: &StarkAirParams,
    params: &DeepAliRoundTwoParams,
    list_size: f64,
    square_list_size: bool,
) -> Option<ErrorBits> {
    let selected = deep_ali_selected_degree_numerator(air, params, list_size)?;
    let excluded = BigUint::from(params.evaluation_trace_domain_union_size);
    if params.field_cardinality <= excluded {
        return None;
    }
    let denominator = &params.field_cardinality - excluded;

    let list_power = if square_list_size { 2.0 } else { 0.0 };
    let list_size_log_upper = ceil_log2_f64(list_size);
    let rational_bits =
        log2_biguint_lower_bound(&denominator) - log2_biguint_upper_bound(&selected);
    let bits = rational_bits - list_power * list_size_log_upper;
    Some(ErrorBits::from_log2(bits.max(0.0)))
}

fn deep_ali_full_field_error(
    air: &StarkAirParams,
    params: &DeepAliRoundTwoParams,
    verifier_rejected_trace_domain_size: usize,
    list_size: f64,
    square_list_size: bool,
) -> Option<ErrorBits> {
    let selected = deep_ali_selected_degree_numerator(air, params, list_size)?;
    if verifier_rejected_trace_domain_size > params.evaluation_trace_domain_union_size {
        return None;
    }

    let excluded = BigUint::from(params.evaluation_trace_domain_union_size);
    if params.field_cardinality <= excluded {
        return None;
    }
    let accepted_forbidden = excluded - BigUint::from(verifier_rejected_trace_domain_size);
    let list_power = if square_list_size {
        (2.0 * ceil_log2_f64(list_size)) as usize
    } else {
        0
    };
    let corrected_numerator = (selected << list_power) + accepted_forbidden;
    if corrected_numerator >= params.field_cardinality {
        return Some(ErrorBits::from_log2(0.0));
    }
    let bits = log2_biguint_lower_bound(&params.field_cardinality)
        - log2_biguint_upper_bound(&corrected_numerator);
    Some(ErrorBits::from_log2(bits.max(0.0)))
}

fn deep_ali_selected_degree_numerator(
    air: &StarkAirParams,
    params: &DeepAliRoundTwoParams,
    list_size: f64,
) -> Option<BigUint> {
    if air.max_constraint_degree == 0
        || params.low_degree_bound == 0
        || params.expanded_low_degree_bound < params.low_degree_bound
        || params.quotient_segment_count == 0
        || params.quotient_segment_degree_bound == 0
        || !list_size.is_finite()
        || list_size < 1.0
    {
        return None;
    }

    let k = BigUint::from(params.low_degree_bound);
    let k_minus_one = BigUint::from(params.low_degree_bound - 1);
    let k_plus_minus_one = BigUint::from(params.expanded_low_degree_bound - 1);
    let first_branch = BigUint::from(air.max_constraint_degree) * &k_plus_minus_one + &k_minus_one;
    let second_branch = k
        + BigUint::from(params.quotient_segment_count - 1)
            * BigUint::from(params.quotient_segment_degree_bound)
        + k_plus_minus_one;
    Some(max(first_branch, second_branch))
}

/// Exact integer lower bound on `log2(value)`.
fn log2_biguint_lower_bound(value: &BigUint) -> f64 {
    let bits = value.bits();
    debug_assert!(bits > 0);
    (bits - 1) as f64
}

/// Exact integer upper bound on `log2(value)`.
fn log2_biguint_upper_bound(value: &BigUint) -> f64 {
    let bits = value.bits();
    debug_assert!(bits > 0);
    let is_power_of_two = value
        .to_u64_digits()
        .iter()
        .map(|digit| digit.count_ones())
        .sum::<u32>()
        == 1;
    if is_power_of_two {
        (bits - 1) as f64
    } else {
        bits as f64
    }
}

/// Exact integer upper bound on the logarithm of a positive normal `f64`.
/// Source list-size bounds smaller than one are invalid for this calculator.
fn ceil_log2_f64(value: f64) -> f64 {
    debug_assert!(value.is_finite() && value >= 1.0);
    let raw = value.to_bits();
    let exponent = ((raw >> 52) & 0x7ff) as i32 - 1023;
    let fraction = raw & ((1u64 << 52) - 1);
    if fraction == 0 {
        exponent as f64
    } else {
        (exponent + 1) as f64
    }
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
        assert!(expected - result.bits() < 2.0);
    }

    #[test]
    fn list_decoding_applies_quadratic_list_size() {
        let params = params();
        let list_size = 17.5;
        let udr = deep_ali_round_two_error_udr(&air(), &params).unwrap();
        let ldr = deep_ali_round_two_error_ldr(&air(), &params, list_size).unwrap();
        assert_eq!(ldr.bits(), udr.bits() - 2.0 * ceil_log2_f64(list_size));
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
    fn full_field_correction_matches_hand_computed_toy_partition() {
        let toy_air = StarkAirParams {
            num_constraints: 1,
            max_constraint_degree: 2,
            max_combo: 1,
        };
        let toy_params = DeepAliRoundTwoParams {
            field_cardinality: BigUint::from(101u8),
            evaluation_trace_domain_union_size: 11,
            low_degree_bound: 2,
            expanded_low_degree_bound: 3,
            quotient_segment_count: 2,
            quotient_segment_degree_bound: 2,
        };

        // The two degree branches are 5 and 6, so A=6. Of the eleven source-
        // excluded points, the verifier rejects three and pessimistically
        // charges the other eight. The corrected exact error is therefore
        // (6+8)/101 = 14/101. The directional integer-log envelope returns
        // floor(log2(101))-ceil(log2(14)) = 6-4 = 2 bits.
        let selected = deep_ali_selected_degree_numerator(&toy_air, &toy_params, 1.0).unwrap();
        assert_eq!(selected, BigUint::from(6u8));
        assert_eq!(
            &selected + BigUint::from(11usize - 3usize),
            BigUint::from(14u8)
        );
        let result = deep_ali_full_field_error_udr(&toy_air, &toy_params, 3).unwrap();
        assert_eq!(result.bits(), 2.0);
    }

    #[test]
    fn full_field_list_decoding_uses_a_directional_quadratic_envelope() {
        let params = params();
        let rejected_trace_size = 1 << 16;
        let list_size = 17.5;
        let selected = deep_ali_selected_degree_numerator(&air(), &params, list_size).unwrap();
        let accepted_forbidden =
            BigUint::from(params.evaluation_trace_domain_union_size - rejected_trace_size);
        let corrected_numerator = (selected << 10usize) + accepted_forbidden;

        let result =
            deep_ali_full_field_error_ldr(&air(), &params, rejected_trace_size, list_size).unwrap();
        let expected = log2_biguint_lower_bound(&params.field_cardinality)
            - log2_biguint_upper_bound(&corrected_numerator);
        assert_eq!(result.bits(), expected);
    }

    #[test]
    fn full_field_correction_fails_closed_on_impossible_set_sizes() {
        let mut invalid = params();
        let union_size = invalid.evaluation_trace_domain_union_size;
        assert!(deep_ali_full_field_error_udr(&air(), &invalid, union_size + 1).is_none());

        invalid.field_cardinality = BigUint::from(union_size);
        assert!(deep_ali_full_field_error_udr(&air(), &invalid, 1 << 16).is_none());
        assert!(deep_ali_full_field_error_ldr(&air(), &params(), 1 << 16, f64::NAN).is_none());
    }

    #[test]
    fn lagrange_coset_degree_transfer_matches_the_concrete_zk_shape() {
        let report = lagrange_coset_rbr_degree_report(1 << 16, 1 << 16, 8).unwrap();
        assert_eq!(report.randomized_trace_degree_bound_exclusive, 1 << 17);
        assert_eq!(report.randomized_chunk_degree_bound_exclusive, 1 << 17);
        assert_eq!(report.selector_degree, 7 << 16);
        assert_eq!(
            report.candidate_recomposition_degree_bound_exclusive,
            9 << 16
        );
        assert_eq!(report.source_low_degree_bound_k, 1 << 17);
        assert_eq!(
            report.source_expanded_low_degree_bound_k_plus,
            (1 << 17) + 2
        );
        assert_eq!(report.source_quotient_segment_count_f, 8);
        assert_eq!(report.source_quotient_segment_length_ell, 1 << 16);
        assert_eq!(
            report.source_candidate_recomposition_degree_bound_exclusive,
            (9 << 16) + 2
        );
        assert!(!report.legacy_mapping_sufficient);
        assert!(report.corrected_mapping_sufficient);
        assert!(!report.arbitrary_candidate_balance_assumed);
        assert!(!report.full_rbr_transfer_established);
    }

    #[test]
    fn legacy_n_plus_two_mapping_underbounds_arbitrary_candidates() {
        let report = lagrange_coset_rbr_degree_report(4, 4, 2).unwrap();
        // A selector has degree 4 and a candidate chunk may have degree 7,
        // so the recomposition may have degree 11 (exclusive bound 12).
        assert_eq!(report.selector_degree, 4);
        assert_eq!(report.candidate_recomposition_degree_bound_exclusive, 12);
        // The legacy k=N, k+=N+2 envelope has degree at most 9
        // (exclusive bound 10), so it cannot justify the candidate class.
        assert_eq!(
            report.legacy_candidate_recomposition_degree_bound_exclusive,
            10
        );
        assert!(!report.legacy_mapping_sufficient);
        // The corrected k=2N, k+=2N+2 envelope is conservative.
        assert_eq!(
            report.source_candidate_recomposition_degree_bound_exclusive,
            14
        );
        assert!(report.corrected_mapping_sufficient);
    }

    #[test]
    fn lagrange_coset_degree_transfer_fails_closed_on_other_shapes_and_overflow() {
        assert!(lagrange_coset_rbr_degree_report(0, 0, 2).is_none());
        assert!(lagrange_coset_rbr_degree_report(4, 2, 2).is_none());
        assert!(lagrange_coset_rbr_degree_report(4, 4, 1).is_none());
        assert!(lagrange_coset_rbr_degree_report(usize::MAX, usize::MAX, 2).is_none());
    }

    #[test]
    fn big_integer_rounding_is_conservative_in_both_directions() {
        let denominator = (BigUint::from(1u8) << 200usize) - BigUint::from(12345u32);
        let numerator = (BigUint::from(1u8) << 120usize) + BigUint::from(67890u32);

        let denominator_floor = log2_biguint_lower_bound(&denominator);
        let denominator_ceil = log2_biguint_upper_bound(&denominator);
        let numerator_floor = log2_biguint_lower_bound(&numerator);
        let numerator_ceil = log2_biguint_upper_bound(&numerator);

        assert_eq!(denominator_floor, 199.0);
        assert_eq!(denominator_ceil, 200.0);
        assert_eq!(numerator_floor, 120.0);
        assert_eq!(numerator_ceil, 121.0);
        assert_eq!(denominator_floor - numerator_ceil, 78.0);
    }

    #[test]
    fn floating_list_size_log_ceiling_is_directional_at_boundaries() {
        assert_eq!(ceil_log2_f64(1.0), 0.0);
        assert_eq!(ceil_log2_f64(2.0), 1.0);
        assert_eq!(ceil_log2_f64(f64::from_bits(2.0f64.to_bits() + 1)), 2.0);
        assert_eq!(ceil_log2_f64(17.5), 5.0);
    }
}

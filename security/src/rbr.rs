//! Exact ideal-IOP round-by-round state and failure-event bookkeeping for the
//! concrete hiding FRI profile.
//!
//! The source state proof is [2021/582, Theorem 5], as adapted to the hiding
//! family in [2024/1553, Theorem 2].  The printed round count in both sources
//! says `3 + |t|`, while the proof itself enumerates three pre-FRI events, one
//! event for every FRI reduction challenge, and a final query event, i.e.
//! `4 + |t|`.  This module follows the proof-consistent ledger and records the
//! discrepancy instead of silently dropping the last reduction event.
//!
//! Plonky3 combines the opening quotients with powers of one challenge.  The
//! batching event is therefore priced with the degree-`M` curve theorem
//! [2025/2055, Theorem 4.2], not as independent coefficients.  Each binary FRI
//! reduction event uses its line specialization, Theorem 1.5.
//!
//! This is an ideal public-coin IOP result.  It deliberately excludes the
//! Fiat--Shamir/ROM reduction, commitment binding, proof-of-work grinding,
//! zero knowledge, and deployment binding.

use alloc::vec::Vec;
use core::cmp::Ordering;

use num_bigint::BigUint;

/// One exact non-negative failure probability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactErrorFraction {
    pub numerator: BigUint,
    pub denominator: BigUint,
}

impl ExactErrorFraction {
    fn new(numerator: BigUint, denominator: BigUint) -> Option<Self> {
        (denominator != BigUint::ZERO && numerator < denominator).then_some(Self {
            numerator,
            denominator,
        })
    }

    fn cmp_value(&self, other: &Self) -> Ordering {
        (&self.numerator * &other.denominator).cmp(&(&other.numerator * &self.denominator))
    }
}

/// Stable identity of the largest event in the exact RbR vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FriRbrFailureEvent {
    ConstraintCombination,
    DeepEvaluation,
    OpeningBatchCombination,
    FriCommitRound(usize),
    FriQueries,
}

/// Complete proof-consistent event ledger for one fixed ideal-IOP profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FriRbrStateTransitionReport {
    pub trace_domain_size: usize,
    pub source_candidate_degree_bound: usize,
    pub fri_evaluation_domain_size: usize,
    pub verifier_rejected_trace_domain_size: usize,
    pub evaluation_trace_domain_union_size: usize,
    pub source_list_size_integer_bound: usize,
    pub constraint_count: usize,
    pub input_batch_count: usize,
    pub input_matrix_count: usize,
    pub input_matrix_widths: Vec<usize>,
    pub input_matrix_opening_point_counts: Vec<usize>,
    pub opening_batch_function_count: usize,
    pub opening_batch_curve_degree: usize,
    pub fri_log_arities: Vec<usize>,
    pub fri_input_height: usize,
    pub fri_input_degree_bound: usize,
    pub fri_log_blowup: usize,
    pub fri_max_log_arity: usize,
    pub fri_commit_round_count: usize,
    pub num_queries: usize,
    pub source_printed_round_count: usize,
    pub proof_consistent_round_count: usize,
    pub failure_event_count: usize,
    pub source_round_count_indexing_discrepancy: bool,
    pub source_round_count_interpretation_author_confirmed: bool,
    pub source_error_vector_reused_verbatim: bool,
    pub source_state_ledger_adapted_with_bchks25_bounds: bool,
    pub constraint_combination_error: ExactErrorFraction,
    pub deep_evaluation_error: ExactErrorFraction,
    pub opening_batch_error: ExactErrorFraction,
    pub fri_commit_round_errors: Vec<ExactErrorFraction>,
    pub fri_query_error: ExactErrorFraction,
    pub maximum_failure_event: FriRbrFailureEvent,
    pub ideal_iop_rbr_error: ExactErrorFraction,
    pub verifier_degree_bound_established: bool,
    pub complete_failure_event_ledger_established: bool,
    pub proof_consistent_ideal_iop_rbr_correspondence_established: bool,
    pub fiat_shamir_rom_uniformity_established: bool,
    pub commitment_binding_established: bool,
    pub zero_knowledge_established: bool,
}

/// Construct the exact report for the locked `beta=2`, rate-`1/4`, `m=3`
/// hiding family.
///
/// `fri_log_arities` is read from the verified proof.  This checker accepts
/// only binary folds and independently reconstructs the required schedule, so
/// a shortened, extended, or reordered event vector fails closed.  The
/// verifier must also bind the proof-carried degree to the expected base trace
/// degree before this function may return a report.
#[allow(clippy::too_many_arguments)]
pub fn exact_fri_rbr_state_transition_report(
    trace_domain_size: usize,
    source_candidate_degree_bound: usize,
    fri_evaluation_domain_size: usize,
    verifier_rejected_trace_domain_size: usize,
    evaluation_trace_domain_union_size: usize,
    source_expanded_candidate_degree_bound: usize,
    quotient_segment_count: usize,
    quotient_segment_length: usize,
    max_constraint_degree: usize,
    source_list_size_integer_bound: usize,
    constraint_count: usize,
    challenge_field_cardinality: BigUint,
    input_batch_count: usize,
    input_matrix_count: usize,
    input_matrix_widths: &[usize],
    input_matrix_opening_point_counts: &[usize],
    opening_batch_function_count: usize,
    all_input_matrices_share_one_height: bool,
    fri_input_height: usize,
    fri_input_degree_bound: usize,
    fri_log_blowup: usize,
    fri_max_log_arity: usize,
    fri_log_arities: &[usize],
    final_domain_size: usize,
    num_queries: usize,
    verifier_degree_bound_established: bool,
) -> Option<FriRbrStateTransitionReport> {
    if trace_domain_size < 4
        || !trace_domain_size.is_power_of_two()
        || source_candidate_degree_bound != trace_domain_size.checked_mul(2)?
        || fri_evaluation_domain_size != source_candidate_degree_bound.checked_mul(4)?
        || verifier_rejected_trace_domain_size != trace_domain_size
        || evaluation_trace_domain_union_size
            != fri_evaluation_domain_size.checked_add(trace_domain_size)?
        || source_expanded_candidate_degree_bound != source_candidate_degree_bound.checked_add(2)?
        || quotient_segment_count < 2
        || quotient_segment_length == 0
        || max_constraint_degree == 0
        || source_list_size_integer_bound == 0
        || constraint_count < 2
        || input_batch_count != 3
        || input_matrix_count != 10
        || input_matrix_widths.len() != input_matrix_count
        || input_matrix_opening_point_counts.len() != input_matrix_count
        || opening_batch_function_count < 2
        || !all_input_matrices_share_one_height
        || fri_input_height != fri_evaluation_domain_size
        || fri_input_degree_bound != source_candidate_degree_bound
        || fri_log_blowup != 2
        || fri_max_log_arity != 1
        || final_domain_size != 4
        || num_queries == 0
        || !verifier_degree_bound_established
        || fri_log_arities.is_empty()
        || fri_log_arities.iter().any(|&log_arity| log_arity != 1)
    {
        return None;
    }

    let recomputed_opening_batch_function_count = input_matrix_widths
        .iter()
        .zip(input_matrix_opening_point_counts)
        .try_fold(0usize, |acc, (&width, &points)| {
            if width == 0 || points == 0 {
                return None;
            }
            acc.checked_add(width.checked_mul(points)?)
        })?;
    if recomputed_opening_batch_function_count != opening_batch_function_count {
        return None;
    }

    let total_log_reduction = fri_log_arities
        .iter()
        .try_fold(0usize, |acc, value| acc.checked_add(*value))?;
    if fri_evaluation_domain_size.checked_shr(total_log_reduction as u32)? != final_domain_size {
        return None;
    }

    // BCHKS25 uses the slightly reduced RS rate `(dimension - 1) / n`.
    // Here the dimension is `2|H|`.  The rational square-root lower bound
    // `(k-2)/(2k)` is below sqrt((k-1)/(4k)), while 1/2 is a strict upper
    // bound.  Those two bounds prove that the line and curve theorem
    // multiplicities are respectively 3 and 6, and give an exact
    // conservative envelope for their displayed error formula.
    let sqrt_rate_lower = PositiveRational::new(
        BigUint::from(source_candidate_degree_bound.checked_sub(2)?),
        BigUint::from(source_candidate_degree_bound.checked_mul(2)?),
    )?;
    let actual_rate = PositiveRational::new(
        BigUint::from(source_candidate_degree_bound.checked_sub(1)?),
        BigUint::from(fri_evaluation_domain_size),
    )?;
    let one_quarter = PositiveRational::from_usize(1, 4)?;
    if sqrt_rate_lower.square().cmp_value(&actual_rate) == Ordering::Greater
        || actual_rate.cmp_value(&one_quarter) != Ordering::Less
        || sqrt_rate_lower.cmp_value(&PositiveRational::from_usize(35, 72)?) != Ordering::Greater
        || sqrt_rate_lower.cmp_value(&PositiveRational::from_usize(7, 15)?) != Ordering::Greater
    {
        return None;
    }

    let field = challenge_field_cardinality;
    if field <= BigUint::from(evaluation_trace_domain_union_size) {
        return None;
    }

    let constraint_numerator = constraint_count
        .checked_sub(1)?
        .checked_mul(source_list_size_integer_bound)?;
    let constraint_combination_error =
        ExactErrorFraction::new(BigUint::from(constraint_numerator), field.clone())?;

    let first_deep_degree = max_constraint_degree
        .checked_mul(source_expanded_candidate_degree_bound.checked_sub(1)?)?
        .checked_add(source_candidate_degree_bound.checked_sub(1)?)?;
    let second_deep_degree = source_candidate_degree_bound
        .checked_add(
            quotient_segment_count
                .checked_sub(1)?
                .checked_mul(quotient_segment_length)?,
        )?
        .checked_add(source_expanded_candidate_degree_bound.checked_sub(1)?)?;
    let deep_degree = first_deep_degree.max(second_deep_degree);
    let deep_list_factor =
        source_list_size_integer_bound.checked_mul(source_list_size_integer_bound)?;
    let accepted_forbidden =
        evaluation_trace_domain_union_size.checked_sub(verifier_rejected_trace_domain_size)?;
    let deep_numerator = deep_degree
        .checked_mul(deep_list_factor)?
        .checked_add(accepted_forbidden)?;
    let deep_evaluation_error =
        ExactErrorFraction::new(BigUint::from(deep_numerator), field.clone())?;

    let opening_batch_curve_degree = opening_batch_function_count.checked_sub(1)?;
    let opening_batch_numerator = proximity_curve_error_numerator_upper(
        fri_evaluation_domain_size,
        6,
        opening_batch_curve_degree,
        &sqrt_rate_lower,
    )?;
    let opening_batch_error = ExactErrorFraction::new(opening_batch_numerator, field.clone())?;

    let mut fri_commit_round_errors = Vec::with_capacity(fri_log_arities.len());
    let mut round_domain_size = fri_evaluation_domain_size;
    for &log_arity in fri_log_arities {
        let arity = 1usize.checked_shl(log_arity as u32)?;
        let curve_degree = arity.checked_sub(1)?;
        let theorem_m = if log_arity == 1 { 3 } else { 6 };
        let numerator = proximity_curve_error_numerator_upper(
            round_domain_size,
            theorem_m,
            curve_degree,
            &sqrt_rate_lower,
        )?;
        fri_commit_round_errors.push(ExactErrorFraction::new(numerator, field.clone())?);
        round_domain_size = round_domain_size.checked_div(arity)?;
    }
    if round_domain_size != final_domain_size {
        return None;
    }

    let query_exponent = u32::try_from(num_queries).ok()?;
    let fri_query_error = ExactErrorFraction::new(
        BigUint::from(7usize).pow(query_exponent),
        BigUint::from(12usize).pow(query_exponent),
    )?;

    let mut maximum_failure_event = FriRbrFailureEvent::ConstraintCombination;
    let mut ideal_iop_rbr_error = constraint_combination_error.clone();
    for (event, error) in [
        (FriRbrFailureEvent::DeepEvaluation, &deep_evaluation_error),
        (
            FriRbrFailureEvent::OpeningBatchCombination,
            &opening_batch_error,
        ),
    ] {
        if error.cmp_value(&ideal_iop_rbr_error) == Ordering::Greater {
            maximum_failure_event = event;
            ideal_iop_rbr_error = error.clone();
        }
    }
    for (round, error) in fri_commit_round_errors.iter().enumerate() {
        if error.cmp_value(&ideal_iop_rbr_error) == Ordering::Greater {
            maximum_failure_event = FriRbrFailureEvent::FriCommitRound(round);
            ideal_iop_rbr_error = error.clone();
        }
    }
    if fri_query_error.cmp_value(&ideal_iop_rbr_error) == Ordering::Greater {
        maximum_failure_event = FriRbrFailureEvent::FriQueries;
        ideal_iop_rbr_error = fri_query_error.clone();
    }

    let fri_commit_round_count = fri_log_arities.len();
    let source_printed_round_count = fri_commit_round_count.checked_add(3)?;
    let proof_consistent_round_count = fri_commit_round_count.checked_add(4)?;
    let failure_event_count = 3usize.checked_add(fri_commit_round_count)?.checked_add(1)?;
    if failure_event_count != proof_consistent_round_count {
        return None;
    }

    Some(FriRbrStateTransitionReport {
        trace_domain_size,
        source_candidate_degree_bound,
        fri_evaluation_domain_size,
        verifier_rejected_trace_domain_size,
        evaluation_trace_domain_union_size,
        source_list_size_integer_bound,
        constraint_count,
        input_batch_count,
        input_matrix_count,
        input_matrix_widths: input_matrix_widths.to_vec(),
        input_matrix_opening_point_counts: input_matrix_opening_point_counts.to_vec(),
        opening_batch_function_count,
        opening_batch_curve_degree,
        fri_log_arities: fri_log_arities.to_vec(),
        fri_input_height,
        fri_input_degree_bound,
        fri_log_blowup,
        fri_max_log_arity,
        fri_commit_round_count,
        num_queries,
        source_printed_round_count,
        proof_consistent_round_count,
        failure_event_count,
        source_round_count_indexing_discrepancy: true,
        source_round_count_interpretation_author_confirmed: false,
        source_error_vector_reused_verbatim: false,
        source_state_ledger_adapted_with_bchks25_bounds: true,
        constraint_combination_error,
        deep_evaluation_error,
        opening_batch_error,
        fri_commit_round_errors,
        fri_query_error,
        maximum_failure_event,
        ideal_iop_rbr_error,
        verifier_degree_bound_established,
        complete_failure_event_ledger_established: true,
        proof_consistent_ideal_iop_rbr_correspondence_established: true,
        fiat_shamir_rom_uniformity_established: false,
        commitment_binding_established: false,
        zero_knowledge_established: false,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PositiveRational {
    numerator: BigUint,
    denominator: BigUint,
}

impl PositiveRational {
    fn new(numerator: BigUint, denominator: BigUint) -> Option<Self> {
        (numerator != BigUint::ZERO && denominator != BigUint::ZERO).then_some(Self {
            numerator,
            denominator,
        })
    }

    fn from_usize(numerator: usize, denominator: usize) -> Option<Self> {
        Self::new(BigUint::from(numerator), BigUint::from(denominator))
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            numerator: &self.numerator * &other.denominator + &other.numerator * &self.denominator,
            denominator: &self.denominator * &other.denominator,
        }
    }

    fn mul(&self, other: &Self) -> Self {
        Self {
            numerator: &self.numerator * &other.numerator,
            denominator: &self.denominator * &other.denominator,
        }
    }

    fn div(&self, other: &Self) -> Self {
        Self {
            numerator: &self.numerator * &other.denominator,
            denominator: &self.denominator * &other.numerator,
        }
    }

    fn scale(&self, value: usize) -> Self {
        Self {
            numerator: &self.numerator * BigUint::from(value),
            denominator: self.denominator.clone(),
        }
    }

    fn pow(&self, exponent: u32) -> Self {
        Self {
            numerator: self.numerator.pow(exponent),
            denominator: self.denominator.pow(exponent),
        }
    }

    fn square(&self) -> Self {
        self.pow(2)
    }

    fn cmp_value(&self, other: &Self) -> Ordering {
        (&self.numerator * &other.denominator).cmp(&(&other.numerator * &self.denominator))
    }

    fn ceil(&self) -> BigUint {
        (&self.numerator + (&self.denominator - BigUint::from(1usize))) / &self.denominator
    }
}

fn proximity_curve_error_numerator_upper(
    domain_size: usize,
    theorem_m: usize,
    curve_degree: usize,
    sqrt_rate_lower: &PositiveRational,
) -> Option<BigUint> {
    if domain_size == 0 || theorem_m < 3 || curve_degree == 0 {
        return None;
    }
    let shifted_m = PositiveRational::from_usize(theorem_m.checked_mul(2)?.checked_add(1)?, 2)?;
    let gamma = PositiveRational::from_usize(5, 12)?;
    let rho_upper = PositiveRational::from_usize(1, 4)?;
    let leading = shifted_m
        .pow(5)
        .scale(2)
        .add(&shifted_m.mul(&gamma).mul(&rho_upper).scale(3));
    let denominator = sqrt_rate_lower.pow(3).scale(3);
    let main = leading.div(&denominator).scale(domain_size);
    let tail = shifted_m.div(sqrt_rate_lower);
    Some(main.add(&tail).scale(curve_degree).ceil())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn concrete_report() -> FriRbrStateTransitionReport {
        exact_fri_rbr_state_transition_report(
            1 << 16,
            1 << 17,
            1 << 19,
            1 << 16,
            (1 << 19) + (1 << 16),
            (1 << 17) + 2,
            8,
            1 << 16,
            3,
            12,
            2256,
            BigUint::from(2_013_265_921u64).pow(4),
            3,
            10,
            &[8, 2388, 8, 8, 8, 8, 8, 8, 8, 8],
            &[1; 10],
            2460,
            true,
            1 << 19,
            1 << 17,
            2,
            1,
            &[1; 17],
            4,
            100,
            true,
        )
        .unwrap()
    }

    #[test]
    fn concrete_ledger_covers_every_proof_consistent_state() {
        let report = concrete_report();
        assert_eq!(report.fri_commit_round_count, 17);
        assert_eq!(report.source_printed_round_count, 20);
        assert_eq!(report.proof_consistent_round_count, 21);
        assert_eq!(report.failure_event_count, 21);
        assert!(report.source_round_count_indexing_discrepancy);
        assert!(!report.source_round_count_interpretation_author_confirmed);
        assert!(!report.source_error_vector_reused_verbatim);
        assert!(report.source_state_ledger_adapted_with_bchks25_bounds);
        assert_eq!(
            report.constraint_combination_error.numerator,
            BigUint::from(27060usize)
        );
        assert_eq!(
            report.deep_evaluation_error.numerator,
            BigUint::from(104_333_456usize)
        );
        assert_eq!(report.opening_batch_curve_degree, 2459);
        assert_eq!(
            report.opening_batch_error.numerator,
            BigUint::from(79_790_622_118_799u64)
        );
        assert_eq!(report.fri_commit_round_errors.len(), 17);
        assert_eq!(
            report.maximum_failure_event,
            FriRbrFailureEvent::OpeningBatchCombination
        );
        assert_eq!(report.ideal_iop_rbr_error, report.opening_batch_error);
        assert!(report.complete_failure_event_ledger_established);
        assert!(report.proof_consistent_ideal_iop_rbr_correspondence_established);
        assert!(!report.fiat_shamir_rom_uniformity_established);
        assert!(!report.commitment_binding_established);
        assert!(!report.zero_knowledge_established);
    }

    #[test]
    fn schedule_and_degree_binding_fail_closed() {
        let mut schedule = [1usize; 17];
        schedule[4] = 2;
        assert!(
            exact_fri_rbr_state_transition_report(
                1 << 16,
                1 << 17,
                1 << 19,
                1 << 16,
                (1 << 19) + (1 << 16),
                (1 << 17) + 2,
                8,
                1 << 16,
                3,
                12,
                2256,
                BigUint::from(2_013_265_921u64).pow(4),
                3,
                10,
                &[8, 2388, 8, 8, 8, 8, 8, 8, 8, 8],
                &[1; 10],
                2460,
                true,
                1 << 19,
                1 << 17,
                2,
                1,
                &schedule,
                4,
                100,
                true,
            )
            .is_none()
        );
        assert!(
            exact_fri_rbr_state_transition_report(
                1 << 16,
                1 << 17,
                1 << 19,
                1 << 16,
                (1 << 19) + (1 << 16),
                (1 << 17) + 2,
                8,
                1 << 16,
                3,
                12,
                2256,
                BigUint::from(2_013_265_921u64).pow(4),
                3,
                10,
                &[8, 2388, 8, 8, 8, 8, 8, 8, 8, 8],
                &[1; 10],
                2460,
                true,
                1 << 19,
                1 << 17,
                2,
                1,
                &[1; 17],
                4,
                100,
                false,
            )
            .is_none()
        );
        assert!(
            exact_fri_rbr_state_transition_report(
                1 << 16,
                1 << 17,
                1 << 19,
                1 << 16,
                (1 << 19) + (1 << 16),
                (1 << 17) + 2,
                8,
                1 << 16,
                3,
                12,
                2256,
                BigUint::from(2_013_265_921u64).pow(4),
                3,
                10,
                &[8, 2387, 8, 8, 8, 8, 8, 8, 8, 8],
                &[1; 10],
                2460,
                true,
                1 << 19,
                1 << 17,
                2,
                1,
                &[1; 17],
                4,
                100,
                true,
            )
            .is_none()
        );
    }
}

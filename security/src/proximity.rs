//! Proximity-gap and list-size helpers shared across low-degree-test
//! modules.
//!
//! - UDR (unique-decoding regime): agreement parameter α = (1 + ρ⁺)/2,
//!   list size L⁺ = 1.
//! - LDR (list-decoding regime, BCHKS25 explicit-m): α = (1 + 1/(2m))·√ρ,
//!   proximity parameter γ = 1 − α, list size L⁺ = (m + 1/2)/√ρ.
//!
//! References:
//! - [2020/654] Proximity Gaps for Reed–Solomon Codes
//! - [2024/1553] STARK-based Signatures from the RPO Permutation
//! - [2025/2055] BCHKS25 Theorem 4.2

use libm::{ceil, pow, sqrt};

/// Legacy base-code parameter report for one concrete FRI analysis.
///
/// This deliberately stops short of claiming that the source RbR state
/// function matches the implementation's acceptance predicate. In the ZK
/// profile, the source candidate-degree envelope is `2 * |H|`, whereas the
/// base Reed--Solomon code used here has dimension `|H|`.  Its final eta
/// comparison is therefore a cross-regime diagnostic, not an application of
/// the RPO Protocol-2 theorem.  Use [`exact_hiding_rbr_state_list_report`] for
/// the theorem-applicable hiding family and its actual committed FRI domain.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LdrListParameterReport {
    pub trace_domain_size: usize,
    pub evaluation_domain_size: usize,
    pub log_blowup: usize,
    pub base_rs_dimension: usize,
    pub base_rs_rate_numerator: usize,
    pub base_rs_rate_denominator: usize,
    pub sqrt_rate_reciprocal: usize,
    pub proximity_m: usize,
    pub proximity_radius_numerator: usize,
    pub proximity_radius_denominator: usize,
    pub source_list_size_bound: usize,
    pub source_hiding_beta: usize,
    pub source_candidate_degree_bound: usize,
    pub source_expanded_candidate_degree_bound: usize,
    pub eta_comparison_left: usize,
    pub eta_comparison_right: usize,
    pub eta_positive: bool,
    pub analysis_m_selected_by_ldt_only: bool,
    pub full_composite_optimality_established: bool,
    pub base_rs_parameters_established: bool,
    pub source_list_formula_instantiated: bool,
    pub state_candidate_family_correspondence_established: bool,
    pub common_rs_and_list_size_regime_established: bool,
}

/// Exact Protocol-2 candidate-family and list-size report for one hiding FRI
/// execution.
///
/// The source code is `RS[G, D, beta * |H|]`.  In the concrete Plonky3 ZK
/// execution `beta = 2`, and the committed FRI codeword domain has size
/// `(beta * |H|) * 2^log_blowup`.  This distinction matters: the base trace
/// has dimension `|H|`, but it is not the code whose list is used by the
/// hiding-family state function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HidingRbrStateListReport {
    pub trace_domain_size: usize,
    pub source_hiding_beta: usize,
    pub source_candidate_degree_bound: usize,
    pub fri_log_blowup: usize,
    pub fri_evaluation_domain_size: usize,
    pub source_rs_rate_numerator: usize,
    pub source_rs_rate_denominator: usize,
    pub proximity_m: usize,
    pub source_agreement_numerator: usize,
    pub source_agreement_denominator: usize,
    pub source_distance_radius_numerator: usize,
    pub source_distance_radius_denominator: usize,
    pub source_expanded_candidate_degree_bound: usize,
    pub eta_comparison_left: usize,
    pub eta_comparison_right: usize,
    pub eta_positive: bool,
    pub source_list_bound_numerator: usize,
    pub source_list_bound_denominator: usize,
    pub source_list_size_integer_bound: usize,
    pub randomized_trace_degree_bound_exclusive: usize,
    pub randomized_chunk_degree_bound_exclusive: usize,
    pub mask_degree_bound_exclusive: usize,
    pub reduced_mask_degree_bound_exclusive: usize,
    pub reduced_opening_degree_bound_exclusive: usize,
    pub constraint_count: usize,
    pub priced_constraint_list_union_numerator: usize,
    pub source_protocol2_function_family_shape_established: bool,
    pub common_rs_and_list_size_regime_established: bool,
    pub state_candidate_family_correspondence_established: bool,
    pub fiat_shamir_uniformity_established: bool,
    pub full_rbr_transfer_established: bool,
    pub zero_knowledge_established: bool,
}

/// Bind the executed hiding-family polynomial shape to the list used by the
/// source Protocol-2 state argument.
///
/// For the source agreement
///
/// `alpha = (1 + 1/(2m)) * sqrt(rho)`
///
/// this function requires an even logarithmic blowup so the value is exact.
/// The finite-length candidate-list bound is the RPO/ethSTARK expression
///
/// `L = m / (rho - 2m/|D|) = m|D| / (k - 2m)`.
///
/// Since a list cardinality is integral, `floor(L)` is a valid integer upper
/// bound whenever the source proves `|P| <= L`.  The exact unreduced rational
/// is retained as well.  This function establishes the candidate-family/list
/// part of the ideal-IOP state argument only; it deliberately does not claim
/// Fiat--Shamir uniformity, the complete RbR theorem, or zero knowledge.
#[allow(clippy::too_many_arguments)]
pub fn exact_hiding_rbr_state_list_report(
    trace_domain_size: usize,
    fri_log_blowup: usize,
    proximity_m: usize,
    fri_evaluation_domain_size: usize,
    randomized_trace_degree_bound_exclusive: usize,
    randomized_chunk_degree_bound_exclusive: usize,
    mask_degree_bound_exclusive: usize,
    reduced_mask_degree_bound_exclusive: usize,
    constraint_count: usize,
) -> Option<HidingRbrStateListReport> {
    if trace_domain_size == 0
        || !trace_domain_size.is_power_of_two()
        || fri_log_blowup == 0
        || !fri_log_blowup.is_multiple_of(2)
        || proximity_m < 3
        || constraint_count < 2
    {
        return None;
    }

    let source_hiding_beta = 2usize;
    let source_candidate_degree_bound = trace_domain_size.checked_mul(source_hiding_beta)?;
    if proximity_m >= source_candidate_degree_bound / 2 {
        return None;
    }
    let source_expanded_candidate_degree_bound = source_candidate_degree_bound.checked_add(2)?;
    let source_rs_rate_denominator = 1usize.checked_shl(fri_log_blowup as u32)?;
    let expected_fri_domain =
        source_candidate_degree_bound.checked_mul(source_rs_rate_denominator)?;
    if fri_evaluation_domain_size != expected_fri_domain {
        return None;
    }

    let expected_mask_bound = source_candidate_degree_bound.checked_sub(1)?;
    let expected_reduced_mask_bound = source_candidate_degree_bound.checked_sub(2)?;
    let reduced_opening_degree_bound_exclusive = expected_mask_bound;
    if randomized_trace_degree_bound_exclusive != source_candidate_degree_bound
        || randomized_chunk_degree_bound_exclusive != source_candidate_degree_bound
        || mask_degree_bound_exclusive != expected_mask_bound
        || reduced_mask_degree_bound_exclusive != expected_reduced_mask_bound
    {
        return None;
    }

    let sqrt_rate_reciprocal = 1usize.checked_shl((fri_log_blowup / 2) as u32)?;
    let two_m = proximity_m.checked_mul(2)?;
    let source_agreement_numerator = two_m.checked_add(1)?;
    let source_agreement_denominator = two_m.checked_mul(sqrt_rate_reciprocal)?;
    let source_distance_radius_numerator =
        source_agreement_denominator.checked_sub(source_agreement_numerator)?;
    if source_distance_radius_numerator == 0 {
        return None;
    }

    // alpha > (k + 2)/|D|, compared without division.
    let eta_comparison_left = source_agreement_numerator.checked_mul(fri_evaluation_domain_size)?;
    let eta_comparison_right =
        source_agreement_denominator.checked_mul(source_expanded_candidate_degree_bound)?;
    let eta_positive = eta_comparison_left > eta_comparison_right;
    if !eta_positive {
        return None;
    }

    let source_list_bound_numerator = proximity_m.checked_mul(fri_evaluation_domain_size)?;
    let source_list_bound_denominator = source_candidate_degree_bound.checked_sub(two_m)?;
    let source_list_size_integer_bound =
        source_list_bound_numerator / source_list_bound_denominator;
    if source_list_size_integer_bound == 0 {
        return None;
    }
    let priced_constraint_list_union_numerator = constraint_count
        .checked_sub(1)?
        .checked_mul(source_list_size_integer_bound)?;

    Some(HidingRbrStateListReport {
        trace_domain_size,
        source_hiding_beta,
        source_candidate_degree_bound,
        fri_log_blowup,
        fri_evaluation_domain_size,
        source_rs_rate_numerator: 1,
        source_rs_rate_denominator,
        proximity_m,
        source_agreement_numerator,
        source_agreement_denominator,
        source_distance_radius_numerator,
        source_distance_radius_denominator: source_agreement_denominator,
        source_expanded_candidate_degree_bound,
        eta_comparison_left,
        eta_comparison_right,
        eta_positive,
        source_list_bound_numerator,
        source_list_bound_denominator,
        source_list_size_integer_bound,
        randomized_trace_degree_bound_exclusive,
        randomized_chunk_degree_bound_exclusive,
        mask_degree_bound_exclusive,
        reduced_mask_degree_bound_exclusive,
        reduced_opening_degree_bound_exclusive,
        constraint_count,
        priced_constraint_list_union_numerator,
        source_protocol2_function_family_shape_established: true,
        common_rs_and_list_size_regime_established: true,
        state_candidate_family_correspondence_established: true,
        fiat_shamir_uniformity_established: false,
        full_rbr_transfer_established: false,
        zero_knowledge_established: false,
    })
}

/// Instantiate the exact Johnson/list-size parameters for the concrete
/// power-of-four FRI rate while retaining the ZK candidate-family gap.
///
/// For `rho = 1 / 2^log_blowup` with even `log_blowup`, this computes
///
/// - `theta = 1 - (1 + 1/(2m)) * sqrt(rho)`, and
/// - `ell_m = (m + 1/2) / sqrt(rho)`.
///
/// It also records a legacy cross-regime `eta > 0` comparison against the
/// separately established hiding envelope `k = 2|H|`, `k+ = k + 2`.  That
/// comparison is not theorem-applicable because this report's domain has size
/// `4|H|`, while the hiding family is committed over `8|H|`. The selected `m`
/// is explicitly labeled as an LDT-only analysis choice: it is not a field
/// carried by the proof and this function does not establish the source state
/// function or a full RbR theorem.
pub fn exact_ldr_list_parameter_report(
    trace_domain_size: usize,
    evaluation_domain_size: usize,
    log_blowup: usize,
    proximity_m: usize,
    source_candidate_degree_bound: usize,
    source_expanded_candidate_degree_bound: usize,
) -> Option<LdrListParameterReport> {
    if trace_domain_size == 0
        || !trace_domain_size.is_power_of_two()
        || log_blowup == 0
        || !log_blowup.is_multiple_of(2)
        || proximity_m < 3
    {
        return None;
    }
    let blowup = 1usize.checked_shl(log_blowup as u32)?;
    let expected_evaluation_domain_size = trace_domain_size.checked_mul(blowup)?;
    if evaluation_domain_size != expected_evaluation_domain_size {
        return None;
    }
    let sqrt_rate_reciprocal = 1usize.checked_shl((log_blowup / 2) as u32)?;
    let expected_source_candidate_degree_bound = trace_domain_size.checked_mul(2)?;
    if source_candidate_degree_bound != expected_source_candidate_degree_bound
        || source_expanded_candidate_degree_bound != source_candidate_degree_bound.checked_add(2)?
    {
        return None;
    }

    let two_m = proximity_m.checked_mul(2)?;
    let two_m_plus_one = two_m.checked_add(1)?;
    let proximity_radius_denominator = two_m.checked_mul(sqrt_rate_reciprocal)?;
    let proximity_radius_numerator = proximity_radius_denominator.checked_sub(two_m_plus_one)?;
    if proximity_radius_numerator == 0 {
        return None;
    }
    let list_size_numerator = two_m_plus_one.checked_mul(sqrt_rate_reciprocal)?;
    if !list_size_numerator.is_multiple_of(2) {
        return None;
    }
    let source_list_size_bound = list_size_numerator / 2;

    // Compare alpha = (2m+1)/(2m*sqrt(1/rho)) with k+/|D| exactly.
    let eta_comparison_left = two_m_plus_one.checked_mul(evaluation_domain_size)?;
    let eta_comparison_right =
        source_expanded_candidate_degree_bound.checked_mul(proximity_radius_denominator)?;
    let eta_positive = eta_comparison_left > eta_comparison_right;
    if !eta_positive {
        return None;
    }

    Some(LdrListParameterReport {
        trace_domain_size,
        evaluation_domain_size,
        log_blowup,
        base_rs_dimension: trace_domain_size,
        base_rs_rate_numerator: 1,
        base_rs_rate_denominator: blowup,
        sqrt_rate_reciprocal,
        proximity_m,
        proximity_radius_numerator,
        proximity_radius_denominator,
        source_list_size_bound,
        source_hiding_beta: 2,
        source_candidate_degree_bound,
        source_expanded_candidate_degree_bound,
        eta_comparison_left,
        eta_comparison_right,
        eta_positive,
        analysis_m_selected_by_ldt_only: true,
        full_composite_optimality_established: false,
        base_rs_parameters_established: true,
        source_list_formula_instantiated: true,
        state_candidate_family_correspondence_established: false,
        common_rs_and_list_size_regime_established: false,
    })
}

/// Performance cap on the proximity parameter `m` searched in LDR
/// analyses. Matches Ethereum's `soundcalc`.
pub const LDR_M_CAP: usize = 1000;

/// UDR agreement parameter α = (1 + ρ⁺)/2, where ρ⁺ accounts for the
/// trace-side expansion from out-of-domain openings.
pub fn alpha_udr(log_trace_length: usize, log_blowup: usize, max_combo: usize) -> f64 {
    let k = (1u64 << log_trace_length) as f64;
    let n = (1u64 << (log_trace_length + log_blowup)) as f64;
    let rho_plus = (k + max_combo as f64) / n;
    (1.0 + rho_plus) * 0.5
}

/// LDR agreement parameter α = (1 + 1/(2m))·√ρ. BCHKS25 §4.2.
pub fn alpha_ldr_m(log_blowup: usize, m: usize) -> f64 {
    let rho = pow(2.0, -(log_blowup as f64));
    (1.0 + 0.5 / m as f64) * sqrt(rho)
}

/// UDR proximity parameter γ = 1 − α used in the multi-point quotient
/// soundness precondition from [2020/654] §4.1.3.
pub fn gamma_udr(log_trace_length: usize, log_blowup: usize, max_combo: usize) -> f64 {
    1.0 - alpha_udr(log_trace_length, log_blowup, max_combo)
}

/// LDR proximity parameter γ = 1 − √ρ·(1 + 1/(2m)). BCHKS25 §4.2.
pub fn gamma_ldr_m(log_blowup: usize, m: usize) -> f64 {
    let rho = pow(2.0, -(log_blowup as f64));
    1.0 - sqrt(rho) * (1.0 + 0.5 / m as f64)
}

/// UDR list size: L⁺ = 1.
pub const fn list_size_udr() -> f64 {
    1.0
}

/// LDR list size: L⁺ = (m + 1/2)/√ρ. Matches `soundcalc`
/// `johnson_bound::get_max_list_size` (explicit-m branch).
pub fn list_size_ldr_m(log_blowup: usize, m: usize) -> f64 {
    let rho = pow(2.0, -(log_blowup as f64));
    (m as f64 + 0.5) / sqrt(rho)
}

/// Largest proximity parameter `m` such that the η > 0 precondition of
/// Theorem 1 in [2021/582] holds. Caller applies [`LDR_M_CAP`].
pub fn compute_upper_m(trace_length: usize) -> usize {
    if trace_length == 0 {
        return 0;
    }
    let h = trace_length as f64;
    let ratio = (h + 2.0) / h;
    ceil(1.0 / (2.0 * (sqrt(ratio) - 1.0))) as usize
}

#[cfg(test)]
mod exact_list_parameter_tests {
    use super::*;

    #[test]
    fn concrete_rate_quarter_parameters_are_exact_and_state_scoped() {
        let report =
            exact_ldr_list_parameter_report(1 << 16, 1 << 18, 2, 3, 1 << 17, (1 << 17) + 2)
                .unwrap();

        assert_eq!(report.base_rs_rate_numerator, 1);
        assert_eq!(report.base_rs_rate_denominator, 4);
        assert_eq!(report.proximity_radius_numerator, 5);
        assert_eq!(report.proximity_radius_denominator, 12);
        assert_eq!(report.source_list_size_bound, 7);
        assert!(report.eta_positive);
        assert!(report.base_rs_parameters_established);
        assert!(report.source_list_formula_instantiated);
        assert!(!report.state_candidate_family_correspondence_established);
        assert!(!report.common_rs_and_list_size_regime_established);
    }

    #[test]
    fn rejects_wrong_domain_or_hiding_degree_envelope() {
        assert!(
            exact_ldr_list_parameter_report(1 << 16, 1 << 17, 2, 100, 1 << 17, (1 << 17) + 2,)
                .is_none()
        );
        assert!(
            exact_ldr_list_parameter_report(1 << 16, 1 << 18, 2, 100, 1 << 16, (1 << 16) + 2,)
                .is_none()
        );
    }

    #[test]
    fn concrete_hiding_state_list_regime_uses_the_committed_fri_domain() {
        let report = exact_hiding_rbr_state_list_report(
            1 << 16,
            2,
            3,
            1 << 19,
            1 << 17,
            1 << 17,
            (1 << 17) - 1,
            (1 << 17) - 2,
            2256,
        )
        .unwrap();

        assert_eq!(report.source_candidate_degree_bound, 1 << 17);
        assert_eq!(report.fri_evaluation_domain_size, 1 << 19);
        assert_eq!(report.source_rs_rate_numerator, 1);
        assert_eq!(report.source_rs_rate_denominator, 4);
        assert_eq!(report.source_agreement_numerator, 7);
        assert_eq!(report.source_agreement_denominator, 12);
        assert_eq!(report.source_distance_radius_numerator, 5);
        assert_eq!(report.source_distance_radius_denominator, 12);
        assert_eq!(report.source_list_bound_numerator, 3 * (1 << 19));
        assert_eq!(report.source_list_bound_denominator, (1 << 17) - 6);
        assert_eq!(report.source_list_size_integer_bound, 12);
        assert_eq!(report.priced_constraint_list_union_numerator, 2255 * 12);
        assert!(report.eta_positive);
        assert!(report.source_protocol2_function_family_shape_established);
        assert!(report.common_rs_and_list_size_regime_established);
        assert!(report.state_candidate_family_correspondence_established);
        assert!(!report.fiat_shamir_uniformity_established);
        assert!(!report.full_rbr_transfer_established);
        assert!(!report.zero_knowledge_established);
    }

    #[test]
    fn hiding_state_list_regime_rejects_the_old_base_domain_substitution() {
        assert!(
            exact_hiding_rbr_state_list_report(
                1 << 16,
                2,
                3,
                1 << 18,
                1 << 17,
                1 << 17,
                (1 << 17) - 1,
                (1 << 17) - 2,
                2256,
            )
            .is_none()
        );
    }
}

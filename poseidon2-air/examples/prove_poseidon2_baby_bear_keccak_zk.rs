use core::fmt::Debug;

use p3_baby_bear::{
    BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16,
    BABYBEAR_S_BOX_DEGREE, BabyBear, GenericPoseidon2LinearLayersBabyBear,
};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, HidingFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_merkle_tree::MerkleTreeHidingMmcs;
use p3_poseidon2_air::{RoundConstants, VectorizedPoseidon2Air};
use p3_security::air::single_alpha_constraint_combination_report;
use p3_security::deep::lagrange_coset_rbr_degree_report;
use p3_security::fri::best_ldr_m;
use p3_security::proximity::{exact_hiding_rbr_state_list_report, exact_ldr_list_parameter_report};
use p3_security::rbr::{FriRbrFailureEvent, exact_fri_rbr_state_transition_report};
use p3_security::{InstanceShape, StarkAirParams};
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::{
    AirLayout, StarkConfig, StarkGenericConfig, get_constraint_layout, prove,
    verify_with_expected_base_degree_bits,
};
use rand::SeedableRng;
use rand::rngs::{SmallRng, StdRng, SysRng};
#[cfg(target_family = "unix")]
use tikv_jemallocator::Jemalloc;
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

#[cfg(target_family = "unix")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const WIDTH: usize = 16;
const SBOX_DEGREE: u64 = BABYBEAR_S_BOX_DEGREE;
const SBOX_REGISTERS: usize = 1;
const HALF_FULL_ROUNDS: usize = BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS;
const PARTIAL_ROUNDS: usize = BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16;

const NUM_ROWS: usize = 1 << 16;
const VECTOR_LEN: usize = 1 << 3;
const NUM_PERMUTATIONS: usize = NUM_ROWS * VECTOR_LEN;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Dft = p3_dft::Radix2DitParallel<BabyBear>;

fn main() -> Result<(), impl Debug> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type ByteHash = Keccak256Hash;
    let byte_hash = ByteHash {};

    type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
    let u64_hash = U64Hash::new(KeccakF {});

    type FieldHash = SerializingHasher<U64Hash>;
    let field_hash = FieldHash::new(u64_hash);

    type MyCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;
    let compress = MyCompress::new(u64_hash);

    type ValMmcs = MerkleTreeHidingMmcs<
        [Val; p3_keccak::VECTOR_LEN],
        [u64; p3_keccak::VECTOR_LEN],
        FieldHash,
        MyCompress,
        StdRng,
        2,
        4,
        4,
    >;
    let mut rng = SmallRng::seed_from_u64(1);
    let constants = RoundConstants::from_rng(&mut rng);
    let mut sys_rng = SysRng;
    let mmcs_rng = StdRng::try_from_rng(&mut sys_rng)
        .expect("the OS entropy source must seed the hiding MMCS RNG");
    let val_mmcs = ValMmcs::new(field_hash, compress, 0, mmcs_rng);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    let challenger = Challenger::from_hasher(vec![], byte_hash);

    let air: VectorizedPoseidon2Air<
        Val,
        GenericPoseidon2LinearLayersBabyBear,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > = VectorizedPoseidon2Air::new(constants);

    let fri_params = FriParameters::new_benchmark_zk(challenge_mmcs);
    let fri_security_regime = fri_params.security_regime();

    let trace = air.generate_vectorized_trace_rows(NUM_PERMUTATIONS, fri_params.log_blowup);

    let dft = Dft::default();

    type Pcs = HidingFriPcs<Val, Dft, ValMmcs, ChallengeMmcs, StdRng>;
    let pcs_rng = StdRng::try_from_rng(&mut sys_rng)
        .expect("the OS entropy source must seed the hiding PCS RNG");
    let pcs = Pcs::new(dft, val_mmcs, fri_params, 4, pcs_rng);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let proof = prove(&config, &air, trace, &[]);

    verify_with_expected_base_degree_bits(&config, &air, &proof, &[], NUM_ROWS.ilog2() as usize)
        .expect("the concrete ZK proof must verify at the fixed base trace degree");

    let degree_reports = config.pcs().quotient_degree_reports();
    assert_eq!(
        degree_reports.len(),
        1,
        "the concrete example must execute exactly one quotient-randomization call"
    );
    let report = degree_reports[0];
    println!(
        "P3_FRI_ZK_RUNTIME_V0 proof_verified=true rng_algorithm=StdRng rng_seed_source=SysRng mmcs_rng_algorithm=StdRng mmcs_seed_source=SysRng quotient_chunk_domain_size={} quotient_chunk_count={} quotient_randomizer_coefficients_per_column={} implemented_randomized_chunk_degree_bound_exclusive={} source_nonfinal_chunk_degree_bound_exclusive={} source_final_chunk_degree_bound_exclusive={} perfect_uniformity_established=false simulator_transfer_established=false zero_knowledge_established=false",
        report.quotient_chunk_domain_size,
        report.quotient_chunk_count,
        report.quotient_randomizer_coefficients_per_column,
        report.implemented_randomized_chunk_degree_bound_exclusive,
        report.source_nonfinal_chunk_degree_bound_exclusive,
        report.source_final_chunk_degree_bound_exclusive,
    );

    let rbr_degree_report = lagrange_coset_rbr_degree_report(
        NUM_ROWS,
        report.quotient_chunk_domain_size,
        report.quotient_chunk_count,
    )
    .expect("the concrete quotient shape must satisfy the checked RbR degree translation");
    println!(
        "P3_FRI_RBR_DEGREE_RUNTIME_V0 proof_verified=true trace_domain_size={} quotient_chunk_domain_size={} quotient_chunk_count={} randomized_trace_degree_bound_exclusive={} randomized_chunk_degree_bound_exclusive={} selector_degree={} candidate_recomposition_degree_bound_exclusive={} source_low_degree_bound_k={} source_expanded_low_degree_bound_k_plus={} source_quotient_segment_count_f={} source_quotient_segment_length_ell={} source_candidate_recomposition_degree_bound_exclusive={} legacy_source_low_degree_bound_k={} legacy_source_expanded_low_degree_bound_k_plus={} legacy_candidate_recomposition_degree_bound_exclusive={} legacy_mapping_sufficient={} corrected_mapping_sufficient={} arbitrary_candidate_balance_assumed={} constraint_combination_transfer_established=false list_size_regime_established=false state_function_correspondence_established=false full_rbr_transfer_established={}",
        rbr_degree_report.trace_domain_size,
        rbr_degree_report.quotient_chunk_domain_size,
        rbr_degree_report.quotient_chunk_count,
        rbr_degree_report.randomized_trace_degree_bound_exclusive,
        rbr_degree_report.randomized_chunk_degree_bound_exclusive,
        rbr_degree_report.selector_degree,
        rbr_degree_report.candidate_recomposition_degree_bound_exclusive,
        rbr_degree_report.source_low_degree_bound_k,
        rbr_degree_report.source_expanded_low_degree_bound_k_plus,
        rbr_degree_report.source_quotient_segment_count_f,
        rbr_degree_report.source_quotient_segment_length_ell,
        rbr_degree_report.source_candidate_recomposition_degree_bound_exclusive,
        rbr_degree_report.legacy_source_low_degree_bound_k,
        rbr_degree_report.legacy_source_expanded_low_degree_bound_k_plus,
        rbr_degree_report.legacy_candidate_recomposition_degree_bound_exclusive,
        rbr_degree_report.legacy_mapping_sufficient,
        rbr_degree_report.corrected_mapping_sufficient,
        rbr_degree_report.arbitrary_candidate_balance_assumed,
        rbr_degree_report.full_rbr_transfer_established,
    );

    // Match quotient_values' SymbolicAirBuilder<Val> instantiation exactly.
    // The resulting layout is later lifted into Challenge when alpha powers
    // are decomposed by the prover.
    let constraint_layout =
        get_constraint_layout::<Val, Val, _>(&air, AirLayout::from_air::<Val>(&air));
    assert_eq!(
        constraint_layout.total_constraints(),
        2256,
        "the concrete Poseidon2 AIR constraint count changed"
    );
    let mut global_indices = constraint_layout.base_indices.clone();
    global_indices.extend_from_slice(&constraint_layout.ext_indices);
    global_indices.sort_unstable();
    assert_eq!(
        global_indices,
        (0..constraint_layout.total_constraints()).collect::<Vec<_>>(),
        "the symbolic base/extension split must preserve every global constraint index exactly once"
    );
    let combination_report = single_alpha_constraint_combination_report(
        constraint_layout.total_constraints(),
        Challenge::order(),
    )
    .expect("the concrete AIR constraint layout must admit an exact root-count report");
    println!(
        "P3_FRI_RBR_CONSTRAINT_COMBINATION_RUNTIME_V0 proof_verified=true num_constraints={} lowest_alpha_power={} highest_alpha_power={} schwartz_zippel_degree_bound={} challenge_field_cardinality={} per_candidate_error_numerator={} historical_calculator_numerator={} historical_calculator_factor_is_conservative={} independent_coefficients_assumed={} nonzero_coefficient_polynomial_required={} source_list_size_binding_established={} state_function_correspondence_established={} fiat_shamir_uniformity_established={} full_rbr_transfer_established={}",
        combination_report.num_constraints,
        combination_report.lowest_alpha_power,
        combination_report.highest_alpha_power,
        combination_report.schwartz_zippel_degree_bound,
        combination_report.challenge_field_cardinality,
        combination_report.per_candidate_error_numerator,
        combination_report.historical_calculator_numerator,
        combination_report.historical_calculator_factor_is_conservative,
        combination_report.independent_coefficients_assumed,
        combination_report.nonzero_coefficient_polynomial_required,
        combination_report.source_list_size_binding_established,
        combination_report.state_function_correspondence_established,
        combination_report.fiat_shamir_uniformity_established,
        combination_report.full_rbr_transfer_established,
    );

    let air_params =
        StarkAirParams::from_air::<Val, Val, _>(&air, AirLayout::from_air::<Val>(&air), 2);
    let instance_shape = InstanceShape {
        log_trace_length: NUM_ROWS.ilog2() as usize,
        modulus_bits: Challenge::order().bits() as usize,
        collision_resistance: 128,
        num_batched_functions: 1,
    };
    let (analysis_m, _) = best_ldr_m(&fri_security_regime, &air_params, &instance_shape)
        .expect("the concrete FRI shape must admit an LDR analysis parameter");
    let list_parameter_report = exact_ldr_list_parameter_report(
        NUM_ROWS,
        NUM_ROWS << fri_security_regime.log_blowup,
        fri_security_regime.log_blowup,
        analysis_m,
        rbr_degree_report.source_low_degree_bound_k,
        rbr_degree_report.source_expanded_low_degree_bound_k_plus,
    )
    .expect("the concrete FRI shape must satisfy the exact list-parameter checks");
    println!(
        "P3_FRI_RBR_LIST_PARAMETERS_RUNTIME_V0 proof_verified=true trace_domain_size={} evaluation_domain_size={} log_blowup={} base_rs_dimension={} base_rs_rate_numerator={} base_rs_rate_denominator={} sqrt_rate_reciprocal={} proximity_m={} proximity_radius_numerator={} proximity_radius_denominator={} source_list_size_bound={} source_hiding_beta={} source_candidate_degree_bound={} source_expanded_candidate_degree_bound={} eta_comparison_left={} eta_comparison_right={} eta_positive={} analysis_m_selected_by_ldt_only={} full_composite_optimality_established={} base_rs_parameters_established={} source_list_formula_instantiated={} state_candidate_family_correspondence_established={} common_rs_and_list_size_regime_established={}",
        list_parameter_report.trace_domain_size,
        list_parameter_report.evaluation_domain_size,
        list_parameter_report.log_blowup,
        list_parameter_report.base_rs_dimension,
        list_parameter_report.base_rs_rate_numerator,
        list_parameter_report.base_rs_rate_denominator,
        list_parameter_report.sqrt_rate_reciprocal,
        list_parameter_report.proximity_m,
        list_parameter_report.proximity_radius_numerator,
        list_parameter_report.proximity_radius_denominator,
        list_parameter_report.source_list_size_bound,
        list_parameter_report.source_hiding_beta,
        list_parameter_report.source_candidate_degree_bound,
        list_parameter_report.source_expanded_candidate_degree_bound,
        list_parameter_report.eta_comparison_left,
        list_parameter_report.eta_comparison_right,
        list_parameter_report.eta_positive,
        list_parameter_report.analysis_m_selected_by_ldt_only,
        list_parameter_report.full_composite_optimality_established,
        list_parameter_report.base_rs_parameters_established,
        list_parameter_report.source_list_formula_instantiated,
        list_parameter_report.state_candidate_family_correspondence_established,
        list_parameter_report.common_rs_and_list_size_regime_established,
    );

    let mask_degree_reports = config.pcs().mask_degree_reports();
    assert_eq!(
        mask_degree_reports.len(),
        1,
        "the concrete example must execute exactly one Protocol-2 mask commitment"
    );
    let mask_report = mask_degree_reports[0];
    println!(
        "P3_FRI_ZK_SIMULATOR_RUNTIME_V0 proof_verified=true quotient_query_simulator=selector_weighted_final_component witness_mask_model=disjoint_coset_interpolation fri_mask_model=protocol2_extension_polynomial trace_domain_size={} extension_coordinate_count={} implemented_mask_degree_bound_exclusive={} source_mask_degree_bound_exclusive={} implemented_reduced_mask_degree_bound_exclusive={} source_reduced_mask_degree_bound_exclusive={} runtime_perfect_uniformity_established=false commitment_hiding_reduction_established=false fiat_shamir_transfer_established=false zero_knowledge_established=false",
        mask_report.trace_domain_size,
        mask_report.extension_coordinate_count,
        mask_report.implemented_mask_degree_bound_exclusive,
        mask_report.source_mask_degree_bound_exclusive,
        mask_report.implemented_reduced_mask_degree_bound_exclusive,
        mask_report.source_reduced_mask_degree_bound_exclusive,
    );

    let hiding_state_list_report = exact_hiding_rbr_state_list_report(
        NUM_ROWS,
        fri_security_regime.log_blowup,
        3,
        rbr_degree_report
            .source_low_degree_bound_k
            .checked_shl(fri_security_regime.log_blowup as u32)
            .expect("the concrete hiding FRI domain must fit usize"),
        rbr_degree_report.randomized_trace_degree_bound_exclusive,
        rbr_degree_report.randomized_chunk_degree_bound_exclusive,
        mask_report.implemented_mask_degree_bound_exclusive,
        mask_report.implemented_reduced_mask_degree_bound_exclusive,
        combination_report.num_constraints,
    )
    .expect("the concrete hiding FRI family must satisfy the source state/list regime");
    println!(
        "P3_FRI_RBR_HIDING_STATE_LIST_RUNTIME_V0 proof_verified=true trace_domain_size={} source_hiding_beta={} source_candidate_degree_bound={} fri_log_blowup={} fri_evaluation_domain_size={} source_rs_rate_numerator={} source_rs_rate_denominator={} proximity_m={} source_agreement_numerator={} source_agreement_denominator={} source_distance_radius_numerator={} source_distance_radius_denominator={} source_expanded_candidate_degree_bound={} eta_comparison_left={} eta_comparison_right={} eta_positive={} source_list_bound_numerator={} source_list_bound_denominator={} source_list_size_integer_bound={} randomized_trace_degree_bound_exclusive={} randomized_chunk_degree_bound_exclusive={} mask_degree_bound_exclusive={} reduced_mask_degree_bound_exclusive={} reduced_opening_degree_bound_exclusive={} constraint_count={} priced_constraint_list_union_numerator={} source_protocol2_function_family_shape_established={} common_rs_and_list_size_regime_established={} state_candidate_family_correspondence_established={} fiat_shamir_uniformity_established={} full_rbr_transfer_established={} zero_knowledge_established={}",
        hiding_state_list_report.trace_domain_size,
        hiding_state_list_report.source_hiding_beta,
        hiding_state_list_report.source_candidate_degree_bound,
        hiding_state_list_report.fri_log_blowup,
        hiding_state_list_report.fri_evaluation_domain_size,
        hiding_state_list_report.source_rs_rate_numerator,
        hiding_state_list_report.source_rs_rate_denominator,
        hiding_state_list_report.proximity_m,
        hiding_state_list_report.source_agreement_numerator,
        hiding_state_list_report.source_agreement_denominator,
        hiding_state_list_report.source_distance_radius_numerator,
        hiding_state_list_report.source_distance_radius_denominator,
        hiding_state_list_report.source_expanded_candidate_degree_bound,
        hiding_state_list_report.eta_comparison_left,
        hiding_state_list_report.eta_comparison_right,
        hiding_state_list_report.eta_positive,
        hiding_state_list_report.source_list_bound_numerator,
        hiding_state_list_report.source_list_bound_denominator,
        hiding_state_list_report.source_list_size_integer_bound,
        hiding_state_list_report.randomized_trace_degree_bound_exclusive,
        hiding_state_list_report.randomized_chunk_degree_bound_exclusive,
        hiding_state_list_report.mask_degree_bound_exclusive,
        hiding_state_list_report.reduced_mask_degree_bound_exclusive,
        hiding_state_list_report.reduced_opening_degree_bound_exclusive,
        hiding_state_list_report.constraint_count,
        hiding_state_list_report.priced_constraint_list_union_numerator,
        hiding_state_list_report.source_protocol2_function_family_shape_established,
        hiding_state_list_report.common_rs_and_list_size_regime_established,
        hiding_state_list_report.state_candidate_family_correspondence_established,
        hiding_state_list_report.fiat_shamir_uniformity_established,
        hiding_state_list_report.full_rbr_transfer_established,
        hiding_state_list_report.zero_knowledge_established,
    );

    let execution_shape_reports = config.pcs().rbr_execution_shape_reports();
    assert_eq!(
        execution_shape_reports.len(),
        1,
        "the concrete example must execute exactly one hiding FRI opening batch"
    );
    let execution_shape = &execution_shape_reports[0];
    assert!(
        execution_shape.all_input_matrices_share_one_height,
        "the registered RbR profile requires one common hiding RS family"
    );
    let rbr_report = exact_fri_rbr_state_transition_report(
        NUM_ROWS,
        hiding_state_list_report.source_candidate_degree_bound,
        hiding_state_list_report.fri_evaluation_domain_size,
        NUM_ROWS,
        hiding_state_list_report
            .fri_evaluation_domain_size
            .checked_add(NUM_ROWS)
            .expect("the disjoint evaluation/trace union must fit usize"),
        hiding_state_list_report.source_expanded_candidate_degree_bound,
        rbr_degree_report.source_quotient_segment_count_f,
        rbr_degree_report.source_quotient_segment_length_ell,
        air_params.max_constraint_degree,
        hiding_state_list_report.source_list_size_integer_bound,
        combination_report.num_constraints,
        Challenge::order(),
        execution_shape.input_batch_count,
        execution_shape.input_matrix_count,
        &execution_shape.input_matrix_widths,
        &execution_shape.input_matrix_opening_point_counts,
        execution_shape.opening_batch_function_count,
        execution_shape.all_input_matrices_share_one_height,
        execution_shape.fri_input_height,
        execution_shape.fri_input_degree_bound,
        execution_shape.fri_log_blowup,
        execution_shape.fri_max_log_arity,
        &execution_shape.fri_log_arities,
        execution_shape.fri_final_domain_size,
        execution_shape.num_queries,
        true,
    )
    .expect("the exact ideal-IOP RbR state and failure-event ledger must validate");
    let maximum_failure_event = match rbr_report.maximum_failure_event {
        FriRbrFailureEvent::ConstraintCombination => "constraint_combination",
        FriRbrFailureEvent::DeepEvaluation => "deep_evaluation",
        FriRbrFailureEvent::OpeningBatchCombination => "opening_batch_combination",
        FriRbrFailureEvent::FriCommitRound(_) => "fri_commit_round",
        FriRbrFailureEvent::FriQueries => "fri_queries",
    };
    let fri_log_arities = rbr_report
        .fri_log_arities
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let fri_commit_error_numerators = rbr_report
        .fri_commit_round_errors
        .iter()
        .map(|error| error.numerator.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let input_matrix_widths = rbr_report
        .input_matrix_widths
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let input_matrix_opening_point_counts = rbr_report
        .input_matrix_opening_point_counts
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "P3_FRI_RBR_STATE_TRANSITION_RUNTIME_V0 proof_verified=true verifier_base_degree_bits_bound=true trace_domain_size={} source_candidate_degree_bound={} source_expanded_candidate_degree_bound={} source_quotient_segment_count={} source_quotient_segment_length={} max_constraint_degree={} fri_evaluation_domain_size={} verifier_rejected_trace_domain_size={} evaluation_trace_domain_union_size={} source_list_size_integer_bound={} constraint_count={} input_batch_count={} input_matrix_count={} input_matrix_widths={} input_matrix_opening_point_counts={} opening_batch_function_count={} opening_batch_curve_degree={} fri_input_height={} fri_input_degree_bound={} fri_log_blowup={} fri_max_log_arity={} fri_log_arities={} fri_commit_round_count={} fri_final_domain_size={} num_queries={} source_printed_round_count={} proof_consistent_round_count={} failure_event_count={} source_round_count_indexing_discrepancy={} source_round_count_interpretation_author_confirmed={} source_error_vector_reused_verbatim={} source_state_ledger_adapted_with_bchks25_bounds={} constraint_combination_error_numerator={} deep_first_degree_bound={} deep_second_degree_bound={} deep_max_degree_bound={} deep_list_factor={} deep_accepted_forbidden_size={} deep_evaluation_error_numerator={} opening_batch_error_numerator={} fri_commit_error_numerators={} common_field_denominator={} fri_query_error_numerator={} fri_query_error_denominator={} maximum_failure_event={} ideal_iop_rbr_error_numerator={} ideal_iop_rbr_error_denominator={} complete_failure_event_ledger_established={} proof_consistent_ideal_iop_rbr_correspondence_established={} fiat_shamir_rom_uniformity_established={} commitment_binding_established={} zero_knowledge_established={}",
        rbr_report.trace_domain_size,
        rbr_report.source_candidate_degree_bound,
        rbr_report.source_expanded_candidate_degree_bound,
        rbr_report.source_quotient_segment_count,
        rbr_report.source_quotient_segment_length,
        rbr_report.max_constraint_degree,
        rbr_report.fri_evaluation_domain_size,
        rbr_report.verifier_rejected_trace_domain_size,
        rbr_report.evaluation_trace_domain_union_size,
        rbr_report.source_list_size_integer_bound,
        rbr_report.constraint_count,
        execution_shape.input_batch_count,
        execution_shape.input_matrix_count,
        input_matrix_widths,
        input_matrix_opening_point_counts,
        rbr_report.opening_batch_function_count,
        rbr_report.opening_batch_curve_degree,
        execution_shape.fri_input_height,
        execution_shape.fri_input_degree_bound,
        execution_shape.fri_log_blowup,
        execution_shape.fri_max_log_arity,
        fri_log_arities,
        rbr_report.fri_commit_round_count,
        execution_shape.fri_final_domain_size,
        rbr_report.num_queries,
        rbr_report.source_printed_round_count,
        rbr_report.proof_consistent_round_count,
        rbr_report.failure_event_count,
        rbr_report.source_round_count_indexing_discrepancy,
        rbr_report.source_round_count_interpretation_author_confirmed,
        rbr_report.source_error_vector_reused_verbatim,
        rbr_report.source_state_ledger_adapted_with_bchks25_bounds,
        rbr_report.constraint_combination_error.numerator,
        rbr_report.deep_first_degree_bound,
        rbr_report.deep_second_degree_bound,
        rbr_report.deep_max_degree_bound,
        rbr_report.deep_list_factor,
        rbr_report.deep_accepted_forbidden_size,
        rbr_report.deep_evaluation_error.numerator,
        rbr_report.opening_batch_error.numerator,
        fri_commit_error_numerators,
        rbr_report.constraint_combination_error.denominator,
        rbr_report.fri_query_error.numerator,
        rbr_report.fri_query_error.denominator,
        maximum_failure_event,
        rbr_report.ideal_iop_rbr_error.numerator,
        rbr_report.ideal_iop_rbr_error.denominator,
        rbr_report.complete_failure_event_ledger_established,
        rbr_report.proof_consistent_ideal_iop_rbr_correspondence_established,
        rbr_report.fiat_shamir_rom_uniformity_established,
        rbr_report.commitment_binding_established,
        rbr_report.zero_knowledge_established,
    );

    Ok::<(), &'static str>(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concrete_poseidon2_air_has_2256_prover_layout_constraints() {
        let mut rng = SmallRng::seed_from_u64(1);
        let constants = RoundConstants::from_rng(&mut rng);
        let air: VectorizedPoseidon2Air<
            Val,
            GenericPoseidon2LinearLayersBabyBear,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        > = VectorizedPoseidon2Air::new(constants);

        let constraint_layout =
            get_constraint_layout::<Val, Val, _>(&air, AirLayout::from_air::<Val>(&air));

        assert_eq!(constraint_layout.total_constraints(), 2256);
    }
}

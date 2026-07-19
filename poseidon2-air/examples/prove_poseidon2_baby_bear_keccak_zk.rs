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
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::{
    AirLayout, StarkConfig, StarkGenericConfig, get_constraint_layout, prove, verify,
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

    let trace = air.generate_vectorized_trace_rows(NUM_PERMUTATIONS, fri_params.log_blowup);

    let dft = Dft::default();

    type Pcs = HidingFriPcs<Val, Dft, ValMmcs, ChallengeMmcs, StdRng>;
    let pcs_rng = StdRng::try_from_rng(&mut sys_rng)
        .expect("the OS entropy source must seed the hiding PCS RNG");
    let pcs = Pcs::new(dft, val_mmcs, fri_params, 4, pcs_rng);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let proof = prove(&config, &air, trace, &[]);

    verify(&config, &air, &proof, &[]).expect("the concrete ZK proof must verify");

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

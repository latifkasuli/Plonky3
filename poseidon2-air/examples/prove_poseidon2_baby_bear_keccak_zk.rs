use core::fmt::Debug;

use p3_baby_bear::{
    BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16,
    BABYBEAR_S_BOX_DEGREE, BabyBear, GenericPoseidon2LinearLayersBabyBear,
};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, HidingFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_merkle_tree::MerkleTreeHidingMmcs;
use p3_poseidon2_air::{RoundConstants, VectorizedPoseidon2Air};
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::{StarkConfig, StarkGenericConfig, prove, verify};
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

type Dft = p3_dft::Radix2DitParallel<BabyBear>;

fn main() -> Result<(), impl Debug> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

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

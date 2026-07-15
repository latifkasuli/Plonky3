use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, quotient_chunk_selector_normalizers};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversalPerm, BitReversibleMatrix};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixCow};
use p3_matrix::horizontally_truncated::HorizontallyTruncated;
use p3_matrix::row_index_mapped::RowIndexMappedView;
use rand::distr::{Distribution, StandardUniform};
use rand::{CryptoRng, Rng, RngExt, SeedableRng};
use spin::Mutex;
use tracing::info_span;

use crate::verifier::FriError;
use crate::{BatchMultiOpening, FriParameters, FriProof, TwoAdicFriPcs};

/// A hiding FRI PCS. Both MMCSs must also be hiding; this is not enforced at compile time so it's
/// the user's responsibility to configure. `R` must implement [`CryptoRng`], but callers remain
/// responsible for seeding it from sufficient entropy. That bound establishes a computational
/// RNG interface, not the paper's information-theoretic uniformity assumption.
#[derive(Debug)]
pub struct HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R: CryptoRng> {
    inner: TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>,
    num_random_codewords: usize,
    rng: Mutex<R>,
    quotient_degree_reports: Mutex<Vec<LagrangeQuotientHidingDegreeReport>>,
}

/// Check the Lagrange-quotient hiding capacity from ePrint 2024/1037,
/// Equations (16) and (17), under this implementation's choice
/// `h = h_p = |H|`.
///
/// `num_extension_queries` is `n_F`, `num_domain_queries` is `n_D`, and
/// `extension_degree` is `e = [F : F_p]`. Arithmetic overflow fails closed.
pub fn lagrange_quotient_hiding_query_bounds_are_satisfied(
    trace_domain_size: usize,
    extension_degree: usize,
    num_extension_queries: usize,
    num_domain_queries: usize,
) -> bool {
    let Some(quotient_queries) = num_extension_queries.checked_add(num_domain_queries) else {
        return false;
    };
    let Some(weighted_queries) = extension_degree
        .checked_mul(num_extension_queries)
        .and_then(|count| count.checked_add(num_domain_queries))
    else {
        return false;
    };
    let Some(required_witness_freedom) = weighted_queries.checked_mul(2) else {
        return false;
    };

    quotient_queries <= trace_domain_size && required_witness_freedom <= trace_domain_size
}

/// Exact degree bookkeeping for the Lagrange-quotient hiding construction.
///
/// This report binds the implementation choice `h_p = |H_i|` to the quotient degree
/// ranges in ePrint 2024/1037, Section 4.2. Every quotient chunk is interpolated
/// from `|H_i|` values, so it starts in `F[X]^{<|H_i|}`. Each nonfinal quotient randomizer
/// is represented by `|H_i|` values drawn from `R`; multiplication by the coset vanishing
/// polynomial gives an implemented randomized-chunk bound of `<2|H_i|`. The report describes
/// only degree and shape, not the source paper's independent-uniform sampling assumption.
///
/// The source permits the final component to have degree below
/// `|H| + max((d + 1)h, h_p)`, which is at least the implementation's `<2|H|`
/// bound when `d > 1`. Arithmetic overflow and invalid shapes return `None`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagrangeQuotientHidingDegreeReport {
    pub quotient_chunk_domain_size: usize,
    pub quotient_chunk_count: usize,
    pub quotient_randomizer_coefficients_per_column: usize,
    pub implemented_randomized_chunk_degree_bound_exclusive: usize,
    pub source_nonfinal_chunk_degree_bound_exclusive: usize,
    pub source_final_chunk_degree_bound_exclusive: usize,
}

pub fn lagrange_quotient_hiding_degree_report(
    quotient_chunk_domain_size: usize,
    quotient_chunk_count: usize,
) -> Option<LagrangeQuotientHidingDegreeReport> {
    if quotient_chunk_domain_size == 0 || quotient_chunk_count <= 1 {
        return None;
    }

    let implemented_randomized_chunk_degree_bound_exclusive =
        quotient_chunk_domain_size.checked_mul(2)?;
    let source_nonfinal_chunk_degree_bound_exclusive =
        quotient_chunk_domain_size.checked_add(quotient_chunk_domain_size)?;
    let source_final_extra = quotient_chunk_count
        .checked_add(1)?
        .checked_mul(quotient_chunk_domain_size)?
        .max(quotient_chunk_domain_size);
    let source_final_chunk_degree_bound_exclusive =
        quotient_chunk_domain_size.checked_add(source_final_extra)?;

    if implemented_randomized_chunk_degree_bound_exclusive
        > source_final_chunk_degree_bound_exclusive
    {
        return None;
    }

    Some(LagrangeQuotientHidingDegreeReport {
        quotient_chunk_domain_size,
        quotient_chunk_count,
        quotient_randomizer_coefficients_per_column: quotient_chunk_domain_size,
        implemented_randomized_chunk_degree_bound_exclusive,
        source_nonfinal_chunk_degree_bound_exclusive,
        source_final_chunk_degree_bound_exclusive,
    })
}

/// Cloning forks the RNG stream by drawing a fresh seed from the source RNG,
/// so the clone and the original never produce the same sequence of masks.
impl<Val, Dft, InputMmcs, FriMmcs, R> Clone for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: Clone,
    Dft: Clone,
    InputMmcs: Clone,
    FriMmcs: Clone,
    R: Rng + CryptoRng + SeedableRng,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            num_random_codewords: self.num_random_codewords,
            rng: Mutex::new(R::from_rng(&mut *self.rng.lock())),
            quotient_degree_reports: Mutex::new(self.quotient_degree_reports.lock().clone()),
        }
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, R: CryptoRng> HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R> {
    pub const fn new(
        dft: Dft,
        mmcs: InputMmcs,
        params: FriParameters<FriMmcs>,
        num_random_codewords: usize,
        rng: R,
    ) -> Self {
        let inner = TwoAdicFriPcs::new(dft, mmcs, params);
        Self {
            inner,
            num_random_codewords,
            rng: Mutex::new(rng),
            quotient_degree_reports: Mutex::new(Vec::new()),
        }
    }

    /// Return the degree reports recorded by actual quotient-randomization calls.
    ///
    /// This is execution evidence for the concrete chunk sizes and counts seen by this PCS. It
    /// does not attest to the RNG's seed source or establish the source paper's simulator claim.
    pub fn quotient_degree_reports(&self) -> Vec<LagrangeQuotientHidingDegreeReport> {
        self.quotient_degree_reports.lock().clone()
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger, R> Pcs<Challenge, Challenger>
    for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: TwoAdicField,
    StandardUniform: Distribution<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    R: Rng + CryptoRng + Send + Sync,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type EvaluationsOnDomain<'a> =
        HorizontallyTruncated<Val, RowIndexMappedView<BitReversalPerm, RowMajorMatrixCow<'a, Val>>>;
    /// The first item contains the openings of the random polynomials added by this wrapper.
    /// The second item is the usual FRI proof.
    type Proof = (
        OpenedValues<Challenge>,
        FriProof<Challenge, FriMmcs, Val, Vec<BatchMultiOpening<Val, InputMmcs>>>,
    );
    type Error = FriError<FriMmcs::Error, InputMmcs::Error>;

    const ZK: bool = true;

    fn zk_query_bounds_are_satisfied(
        &self,
        trace_domain_size: usize,
        num_extension_queries: usize,
    ) -> bool {
        lagrange_quotient_hiding_query_bounds_are_satisfied(
            trace_domain_size,
            Challenge::DIMENSION,
            num_extension_queries,
            self.inner.fri.num_queries,
        )
    }

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &self.inner, degree)
    }

    fn log_max_lde_height(&self) -> usize {
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<Challenge, Challenger>>::log_max_lde_height(
            &self.inner)
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let randomized_evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)> =
            info_span!("randomize polys").in_scope(|| {
                evaluations
                    .into_iter()
                    .map(|(domain, mat)| {
                        let mat_width = mat.width();
                        let randomized_trace_height = mat
                            .height()
                            .checked_mul(2)
                            .expect("ZK trace height must fit usize");
                        assert_eq!(
                            domain.size(),
                            randomized_trace_height,
                            "ZK trace domain must have exactly twice the unrandomized trace height"
                        );
                        // Let `w` and `h` be the width and height of the original matrix. The randomized matrix should have height `2h` and width `w + num_random_codewords`.
                        // To generate it, we add `w + 2 * num_random_codewords` columns to the original matrix, then reshape it by setting the width to `w + num_random_codewords`.
                        // All columns are added on the right hand side so, after reshaping, this has the net effect of adding `num_random_codewords` random columns on the right and interleaving the original trace with random rows.

                        let mut random_evaluation = mat.with_random_cols(
                            mat_width + 2 * self.num_random_codewords,
                            &mut *self.rng.lock(),
                        );
                        random_evaluation.width = mat_width + self.num_random_codewords;

                        (domain, random_evaluation)
                    })
                    .collect()
            });

        Pcs::<Challenge, Challenger>::commit(&self.inner, randomized_evaluations)
    }

    fn commit_preprocessing(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        // Pad values with zero columns instead of random columns.
        let padded_evals = evaluations
            .into_iter()
            .map(|(domain, mat)| {
                let mat_width = mat.width();
                // Let `w` and `h` be the width and height of the original matrix. The padded matrix should have height `2h` and width `w`.
                // To generate it, we add `w` zero columns to the original matrix, then reshape it by setting the width to `w`.
                // All columns are added on the right hand side so, after reshaping, this has the net effect of adding interleaving the original trace with zero rows.
                let mut padded_evaluation = mat.with_zero_cols(mat_width);
                padded_evaluation.width = mat_width;
                (domain, padded_evaluation)
            })
            .collect::<Vec<_>>();

        Pcs::<Challenge, Challenger>::commit(&self.inner, padded_evals)
    }

    /// Get the quotient polynomial LDEs. We first decompose the quotient polynomial into
    /// `num_chunks` many smaller polynomials each of degree `degree / num_chunks`.
    /// These quotient polynomials are then randomized as explained in Section 4.2 of
    /// <https://eprint.iacr.org/2024/1037.pdf>.
    ///
    /// ### Arguments
    /// - `quotient_domain` the domain of the quotient polynomial.
    /// - `quotient_evaluations` the evaluations of the quotient polynomial over the domain. This should be in
    ///   standard (not bit-reversed) order.
    /// - `num_chunks` the number of smaller polynomials to decompose the quotient polynomial into.
    ///
    /// # Panics
    /// This function panics if `num_chunks` is either `0` or `1`. The first case makes no logical
    /// sense and in the second case, the resulting commitment would not be hiding.
    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        num_chunks: usize,
    ) -> Vec<RowMajorMatrix<Val>> {
        assert!(
            num_chunks > 1,
            "num_chunks must be > 1 to preserve hiding (got {num_chunks})"
        );
        let (domains, evaluations): (Vec<_>, Vec<_>) = evaluations.into_iter().unzip();
        assert_eq!(
            domains.len(),
            num_chunks,
            "num_chunks must match the quotient partition"
        );
        let first = evaluations
            .first()
            .expect("a hiding quotient partition must contain at least two chunks");
        let h = first.height();
        let input_width = first.width();
        assert!(
            evaluations
                .iter()
                .all(|evaluation| evaluation.height() == h && evaluation.width() == input_width),
            "all hiding quotient chunks must have one common nonempty shape"
        );
        let degree_report = lagrange_quotient_hiding_degree_report(h, num_chunks)
            .expect("ZK quotient degree bookkeeping must be valid and fit usize");
        assert_eq!(
            degree_report.implemented_randomized_chunk_degree_bound_exclusive,
            degree_report.source_nonfinal_chunk_degree_bound_exclusive,
            "implemented quotient randomizers must match the source first-chunk degree regime"
        );
        self.quotient_degree_reports.lock().push(degree_report);
        let cis = quotient_chunk_selector_normalizers(&domains)
            .expect("split quotient domains must be nonempty and disjoint");
        let mut rng = self.rng.lock();
        let randomized_evaluations: Vec<RowMajorMatrix<Val>> = evaluations
            .into_iter()
            .map(|mat| mat.with_random_cols(self.num_random_codewords, &mut *rng))
            .collect();
        let w = randomized_evaluations[0].width();
        assert_eq!(
            w,
            input_width + self.num_random_codewords,
            "hiding quotient width must include every random codeword"
        );
        // Add random values to the LDE evaluations as described in https://eprint.iacr.org/2024/1037.pdf.
        // If we have `d` chunks, let q'_i(X) = q_i(X) + v_H_i(X) * t_i(X) where t_i(X) is random, for 1 <= i < d.
        // Equation (15) forces q'_d(X) = q_d(X) - v_H_d(X) c_d^-1 \sum_i c_i t_i(X), where c_i is a Lagrange normalization constant.
        // The source PDF's Equation (14) sums over k but prints c_i t_i; the preservation identity fixes the bound index as written here.
        let values_per_chunk = h
            .checked_mul(w)
            .expect("ZK quotient randomizer size must fit usize");
        let sampled_values = randomized_evaluations
            .len()
            .checked_sub(1)
            .and_then(|count| count.checked_mul(values_per_chunk))
            .expect("ZK quotient randomizer allocation must fit usize");
        let mut all_random_values = (0..sampled_values)
            .map(|_| rng.random())
            .chain(core::iter::repeat_n(Val::ZERO, values_per_chunk))
            .collect::<Vec<_>>();

        balance_quotient_randomizers(&cis, values_per_chunk, &mut all_random_values)
            .expect("quotient randomizer shape and selector normalizers must be valid");

        domains
            .into_iter()
            .zip(randomized_evaluations)
            .enumerate()
            .map(|(i, (domain, evals))| {
                assert_eq!(domain.size(), evals.height());
                let random_values = &all_random_values[i * h * w..(i + 1) * h * w];

                // Work in the logical polynomial basis for this chunk's own coset. This avoids
                // relying on a manual DFT-variable rescaling: `coset_idft_batch` recovers the
                // coefficients of q_i(X), then we add
                //
                //     Z_{H_i}(X) t_i(X) = (g_i^{-h} X^h - 1) t_i(X)
                //
                // coefficientwise before evaluating the randomized polynomial on the common
                // FRI coset. The previous rescaling shortcut did not produce this polynomial for
                // arbitrary transcript points and could lead to `FinalPolyMismatch`.
                let mut randomized_coefficients =
                    self.inner.dft.coset_idft_batch(evals, domain.shift());
                let added_bits = self
                    .inner
                    .fri
                    .log_blowup
                    .checked_add(1)
                    .expect("ZK quotient LDE exponent must fit usize");
                let added_bits_u32 =
                    u32::try_from(added_bits).expect("ZK quotient LDE exponent must fit u32");
                let lde_height = h
                    .checked_shl(added_bits_u32)
                    .expect("ZK quotient LDE height must fit usize");
                let lde_values = lde_height
                    .checked_mul(w)
                    .expect("ZK quotient LDE allocation must fit usize");
                randomized_coefficients.values.resize(lde_values, Val::ZERO);
                assert_eq!(randomized_coefficients.width(), w);
                assert_eq!(randomized_coefficients.values.len(), lde_values);

                let h_u64 = u64::try_from(h).expect("ZK quotient chunk size must fit u64");
                let vanishing_leading_coefficient = domain.shift().inverse().exp_u64(h_u64);
                for coefficient in 0..h {
                    for column in 0..w {
                        let randomizer = random_values[coefficient * w + column];
                        randomized_coefficients.values[coefficient * w + column] -= randomizer;
                        randomized_coefficients.values[(h + coefficient) * w + column] +=
                            vanishing_leading_coefficient * randomizer;
                    }
                }

                let lde_evals = self
                    .inner
                    .dft
                    .coset_dft_batch(randomized_coefficients, Val::GENERATOR)
                    .to_row_major_matrix();

                lde_evals.bit_reverse_rows().to_row_major_matrix()
            })
            .collect()
    }

    fn commit_ldes(&self, ldes: Vec<RowMajorMatrix<Val>>) -> (Self::Commitment, Self::ProverData) {
        Pcs::<Challenge, Challenger>::commit_ldes(&self.inner, ldes)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let inner_evals = <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<
            Challenge,
            Challenger,
        >>::get_evaluations_on_domain(
            &self.inner, prover_data, idx, domain
        );
        let inner_width = inner_evals.width();
        // Truncate off the columns representing random codewords we added in `commit` above.
        // The unwrap is safe as inner_width - self.num_random_codewords <= inner_width.
        HorizontallyTruncated::new(inner_evals, inner_width - self.num_random_codewords).unwrap()
    }

    fn get_evaluations_on_domain_no_random<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let inner_evals = <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<
            Challenge,
            Challenger,
        >>::get_evaluations_on_domain(
            &self.inner, prover_data, idx, domain
        );
        let inner_width = inner_evals.width();

        HorizontallyTruncated::new(inner_evals, inner_width).unwrap()
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        self.open_with_preprocessing(rounds, challenger, false)
    }

    fn open_with_preprocessing(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        challenger: &mut Challenger,
        is_preprocessing: bool,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        // A generic PCS caller can request arbitrary extension-field openings,
        // so conservatively count every distinct point as one Q_F query. The
        // STARK provers perform the protocol-specific one-zeta check before
        // committing; this backstop keeps direct PCS use fail-closed as well.
        let mut extension_query_points = Vec::<Challenge>::new();
        for (_, points_by_matrix) in &rounds {
            for point in points_by_matrix.iter().flatten() {
                if !extension_query_points.contains(point) {
                    extension_query_points.push(*point);
                }
            }
        }

        let lde_log = self
            .inner
            .fri
            .log_blowup
            .checked_add(1)
            .expect("ZK LDE exponent must fit usize");
        let lde_shift = u32::try_from(lde_log).expect("ZK LDE exponent must fit u32");
        let lde_factor = 1usize
            .checked_shl(lde_shift)
            .expect("ZK LDE factor must fit usize");
        for (data, _) in &rounds {
            for matrix in self.inner.mmcs.get_matrices(data) {
                assert!(
                    matrix.height() % lde_factor == 0,
                    "ZK committed LDE height must be divisible by twice the FRI blowup"
                );
                let trace_domain_size = matrix.height() / lde_factor;
                assert!(
                    lagrange_quotient_hiding_query_bounds_are_satisfied(
                        trace_domain_size,
                        Challenge::DIMENSION,
                        extension_query_points.len(),
                        self.inner.fri.num_queries,
                    ),
                    "ZK randomizer degree is insufficient for ePrint 2024/1037 Equations (16) and (17): trace domain size {trace_domain_size}, distinct extension opening points {}",
                    extension_query_points.len(),
                );
            }
        }

        let (mut inner_opened_values, inner_proof) =
            self.inner
                .open_with_preprocessing(rounds, challenger, is_preprocessing);
        // inner_opened_values includes opened values for the random codewords. Those should be
        // hidden from our caller, so we split them off and store them in the proof.
        let opened_values_rand = inner_opened_values
            .iter_mut()
            .enumerate()
            .map(|(idx, opened_values_for_round)| {
                opened_values_for_round
                    .iter_mut()
                    .map(|opened_values_for_mat| {
                        opened_values_for_mat
                            .iter_mut()
                            .map(|opened_values_for_point| {
                                let num_random_codewords =
                                    if is_preprocessing && idx == <Self as Pcs<Challenge, Challenger>>::PREPROCESSED_TRACE_IDX {
                                        0
                                    } else {
                                        self.num_random_codewords
                                    };
                                let split = opened_values_for_point.len() - num_random_codewords;
                                opened_values_for_point.drain(split..).collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        (inner_opened_values, (opened_values_rand, inner_proof))
    }

    fn verify(
        &self,
        // For each round:
        mut rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<(
                // its domain,
                Self::Domain,
                // for each point:
                Vec<(
                    // the point,
                    Challenge,
                    // values at the point
                    Vec<Challenge>,
                )>,
            )>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let (opened_values_for_rand_cws, inner_proof) = proof;

        // Proving split each opening into a public half and a hidden half.
        // - The public half lives here.
        // - The hidden half travels beside the proof as random codewords.
        //
        // Re-joining them gives the inner verifier the full openings it committed to.
        //
        //     public (per point):  [v_0, .., v_k]
        //     hidden (per point):                [r_0, .., r_m]
        //     merged:              [v_0, .., v_k,  r_0, .., r_m]
        //
        // Invariant: the halves nest identically by round, then matrix, then point.
        // Each level's length is checked before merging.
        // A mismatch returns a precise error instead of being truncated silently.

        // Level 1: one set of random openings per round.
        if opened_values_for_rand_cws.len() != rounds.len() {
            return Err(FriError::HidingRandomOpeningRoundCountMismatch {
                expected: rounds.len(),
                got: opened_values_for_rand_cws.len(),
            });
        }
        for (round_idx, (round, rand_round)) in rounds
            .iter_mut()
            .zip(opened_values_for_rand_cws.iter())
            .enumerate()
        {
            // Level 2: one set per matrix in this round.
            if rand_round.len() != round.1.len() {
                return Err(FriError::HidingRandomOpeningMatrixCountMismatch {
                    round: round_idx,
                    expected: round.1.len(),
                    got: rand_round.len(),
                });
            }
            for (matrix_idx, (mat, rand_mat)) in
                round.1.iter_mut().zip(rand_round.iter()).enumerate()
            {
                // Level 3: one set per opening point of this matrix.
                if rand_mat.len() != mat.1.len() {
                    return Err(FriError::HidingRandomOpeningPointCountMismatch {
                        round: round_idx,
                        matrix: matrix_idx,
                        expected: mat.1.len(),
                        got: rand_mat.len(),
                    });
                }
                // Shapes agree: append the hidden values onto the public ones.
                for (point, rand_point) in mat.1.iter_mut().zip(rand_mat.iter()) {
                    point.1.extend(rand_point);
                }
            }
        }
        self.inner.verify(rounds, inner_proof, challenger)
    }

    fn get_opt_randomization_poly_commitment(
        &self,
        ext_trace_domains: impl IntoIterator<Item = Self::Domain>,
    ) -> Option<(Self::Commitment, Self::ProverData)> {
        let random_input_vals = ext_trace_domains
            .into_iter()
            .map(|domain| {
                let m = DenseMatrix::rand(
                    &mut *self.rng.lock(),
                    domain.size(),
                    self.num_random_codewords + Challenge::DIMENSION,
                );

                (domain, m)
            })
            .collect::<Vec<_>>();

        let r_commit_and_data =
            Pcs::<Challenge, Challenger>::commit(&self.inner, random_input_vals);
        Some(r_commit_and_data)
    }

    fn build_periodic_lde_table(
        &self,
        periodic_cols: &[Vec<Val>],
        trace_domain: Self::Domain,
        quotient_domain: Self::Domain,
    ) -> p3_commit::PeriodicLdeTable<Val> {
        Pcs::<Challenge, Challenger>::build_periodic_lde_table(
            &self.inner,
            periodic_cols,
            trace_domain,
            quotient_domain,
        )
    }
}

/// Fill the final quotient randomizer so the selector-weighted sum is zero.
///
/// If `c_i` are the quotient selector normalizers, this enforces
/// `sum_i c_i t_i = 0` at every coefficient position. Consequently adding
/// `Z_{H_i} t_i` to chunk `i` preserves the recomposed quotient polynomial.
fn balance_quotient_randomizers<F: Field>(
    normalizers: &[F],
    values_per_chunk: usize,
    randomizers: &mut [F],
) -> Option<()> {
    if normalizers.len() < 2
        || values_per_chunk == 0
        || normalizers.len().checked_mul(values_per_chunk) != Some(randomizers.len())
    {
        return None;
    }

    let last_chunk = normalizers.len() - 1;
    let last_normalizer_inv = normalizers[last_chunk].try_inverse()?;
    for chunk in 0..last_chunk {
        let multiplier = normalizers[chunk] * last_normalizer_inv;
        for offset in 0..values_per_chunk {
            let value = randomizers[chunk * values_per_chunk + offset] * multiplier;
            randomizers[last_chunk * values_per_chunk + offset] -= value;
        }
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use itertools::Itertools;
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::{ExtensionMmcs, PolynomialSpace};
    use p3_dft::Radix2Dit;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::{SmallRng, StdRng};

    use super::*;

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Dft = Radix2Dit<Val>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type MyPcs = HidingFriPcs<Val, Dft, ValMmcs, ChallengeMmcs, StdRng>;

    type Commitment = <ValMmcs as Mmcs<Val>>::Commitment;
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    /// Public opening claims (the `rounds` argument): per matrix, its domain and
    /// the `(point, values)` pairs.
    type Claims = Vec<(Domain, Vec<(Challenge, Vec<Challenge>)>)>;
    type Proof = <MyPcs as Pcs<Challenge, Challenger>>::Proof;
    type TestError =
        FriError<<ChallengeMmcs as Mmcs<Challenge>>::Error, <ValMmcs as Mmcs<Val>>::Error>;

    /// Random codewords appended per matrix.
    ///
    /// Must be `> 0` so each opening splits into a public part (`rounds`) and a
    /// hidden part (`proof.0`) — the split `verify` re-merges and whose shape the
    /// new error variants guard.
    const NUM_RANDOM_CODEWORDS: usize = 2;

    #[test]
    fn lagrange_quotient_hiding_query_bounds_match_source_equations() {
        // BabyBear's standard challenge extension has e = 4. With one Q_F
        // query and two FRI Q_D queries, Equation (17) requires 12 degrees of
        // witness-randomizer freedom.
        assert!(!lagrange_quotient_hiding_query_bounds_are_satisfied(
            8, 4, 1, 2
        ));
        assert!(lagrange_quotient_hiding_query_bounds_are_satisfied(
            16, 4, 1, 2
        ));
        assert!(!lagrange_quotient_hiding_query_bounds_are_satisfied(
            16, 4, 1, 8
        ));

        // Overflow must not wrap a huge, invalid query budget into acceptance.
        assert!(!lagrange_quotient_hiding_query_bounds_are_satisfied(
            usize::MAX,
            usize::MAX,
            2,
            1
        ));
    }

    #[test]
    fn lagrange_quotient_hiding_degree_report_matches_source_regime() {
        for quotient_chunk_count in [2, 4, 8] {
            let report = lagrange_quotient_hiding_degree_report(16, quotient_chunk_count)
                .expect("valid source degree regime");
            assert_eq!(report.quotient_chunk_domain_size, 16);
            assert_eq!(report.quotient_chunk_count, quotient_chunk_count);
            assert_eq!(report.quotient_randomizer_coefficients_per_column, 16);
            assert_eq!(
                report.implemented_randomized_chunk_degree_bound_exclusive,
                32
            );
            assert_eq!(report.source_nonfinal_chunk_degree_bound_exclusive, 32);
            assert_eq!(
                report.source_final_chunk_degree_bound_exclusive,
                16 + (quotient_chunk_count + 1) * 16
            );
        }

        assert!(lagrange_quotient_hiding_degree_report(0, 2).is_none());
        assert!(lagrange_quotient_hiding_degree_report(16, 1).is_none());
        assert!(lagrange_quotient_hiding_degree_report(usize::MAX, 2).is_none());
        assert!(lagrange_quotient_hiding_degree_report(16, usize::MAX).is_none());
    }

    #[test]
    fn quotient_randomizer_balance_preserves_recomposition() {
        let domain = Domain::new(Val::GENERATOR, 4).unwrap();

        for num_chunks in [2, 4, 8] {
            let domains = domain.split_domains(num_chunks);
            let normalizers =
                quotient_chunk_selector_normalizers(&domains).expect("split domains are disjoint");
            let values_per_chunk = 5;
            let mut randomizers = (0..(num_chunks - 1) * values_per_chunk)
                .map(|i| Val::from_usize(i * i + 11))
                .chain(core::iter::repeat_n(Val::ZERO, values_per_chunk))
                .collect_vec();

            balance_quotient_randomizers(&normalizers, values_per_chunk, &mut randomizers)
                .expect("valid randomizer shape");

            for offset in 0..values_per_chunk {
                let weighted_sum = normalizers
                    .iter()
                    .enumerate()
                    .map(|(chunk, normalizer)| {
                        *normalizer * randomizers[chunk * values_per_chunk + offset]
                    })
                    .sum::<Val>();
                assert_eq!(weighted_sum, Val::ZERO, "failed for {num_chunks} chunks");
            }
        }
    }

    #[test]
    fn randomized_quotient_ldes_preserve_recomposition_at_arbitrary_points() {
        let mut setup_rng = SmallRng::seed_from_u64(9);
        let perm = Perm::new_from_rng_128(&mut setup_rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let fri_params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: challenge_mmcs,
        };
        let pcs = MyPcs::new(
            Dft::default(),
            val_mmcs,
            fri_params,
            NUM_RANDOM_CODEWORDS,
            StdRng::seed_from_u64(10),
        );

        let quotient_domain = Domain::new(Val::GENERATOR, 4).unwrap();
        let coefficients = (0..quotient_domain.size())
            .map(|i| Val::from_usize(i * i + 3 * i + 7))
            .collect_vec();
        let quotient_evaluations = quotient_domain
            .iter()
            .map(|point| {
                coefficients
                    .iter()
                    .rev()
                    .fold(Val::ZERO, |acc, &coefficient| acc * point + coefficient)
            })
            .collect_vec();

        for num_chunks in [2, 4, 8] {
            let chunk_domains = quotient_domain.split_domains(num_chunks);
            let chunk_evaluations = quotient_domain.split_evals(
                num_chunks,
                RowMajorMatrix::new_col(quotient_evaluations.clone()),
            );
            let plain_ldes = chunk_domains
                .iter()
                .copied()
                .zip(chunk_evaluations.clone())
                .map(|(domain, evaluations)| {
                    pcs.inner
                        .dft
                        .coset_lde_batch(
                            evaluations,
                            pcs.inner.fri.log_blowup + 1,
                            Val::GENERATOR / domain.shift(),
                        )
                        .bit_reverse_rows()
                        .to_row_major_matrix()
                })
                .collect_vec();
            let ldes = <MyPcs as Pcs<Challenge, Challenger>>::get_quotient_ldes(
                &pcs,
                chunk_domains.iter().copied().zip(chunk_evaluations),
                num_chunks,
            );

            for point in [Val::from_u32(12345), Val::from_u32(67890)] {
                let selectors = p3_commit::quotient_chunk_selectors_at_point(&chunk_domains, point)
                    .expect("split quotient domains are disjoint");
                let recomposed = ldes
                    .iter()
                    .zip(selectors)
                    .map(|(lde, selector)| {
                        let standard_order = lde.clone().bit_reverse_rows().to_row_major_matrix();
                        let column = (0..standard_order.height())
                            .map(|row| standard_order.get(row, 0).unwrap())
                            .collect_vec();
                        let lde_domain = Domain::new(
                            Val::GENERATOR,
                            p3_util::log2_strict_usize(standard_order.height()),
                        )
                        .unwrap();
                        selector * lde_domain.evaluate_polynomial_at(&column, point)
                    })
                    .sum::<Val>();
                let plain_recomposed = plain_ldes
                    .iter()
                    .zip(
                        p3_commit::quotient_chunk_selectors_at_point(&chunk_domains, point)
                            .unwrap(),
                    )
                    .map(|(lde, selector)| {
                        let standard_order = lde.clone().bit_reverse_rows().to_row_major_matrix();
                        let column = (0..standard_order.height())
                            .map(|row| standard_order.get(row, 0).unwrap())
                            .collect_vec();
                        let lde_domain = Domain::new(
                            Val::GENERATOR,
                            p3_util::log2_strict_usize(standard_order.height()),
                        )
                        .unwrap();
                        selector * lde_domain.evaluate_polynomial_at(&column, point)
                    })
                    .sum::<Val>();
                let expected = quotient_domain.evaluate_polynomial_at(&quotient_evaluations, point);

                assert_eq!(plain_recomposed, expected);

                assert_eq!(
                    recomposed, expected,
                    "failed for {num_chunks} chunks at {point}"
                );
            }
        }
    }

    /// Run a real prover roundtrip and return `(pcs, claims, proof, challenger)`
    /// ready to verify, with `challenger` advanced past the commitment.
    ///
    /// One round, one matrix, one point, so the random-opening tree nests as:
    ///
    /// ```text
    ///     proof.0          = [round_0]      // 1 round
    ///     proof.0[0]       = [matrix_0]     // 1 matrix
    ///     proof.0[0][0]    = [point_0]      // 1 point
    ///     proof.0[0][0][0] = [v_0, v_1]     // NUM_RANDOM_CODEWORDS values
    /// ```
    ///
    /// Each test perturbs one level to trip the matching count check.
    fn make_fixture() -> (MyPcs, Vec<(Commitment, Claims)>, Proof, Challenger) {
        make_fixture_with_log_degree(4)
    }

    fn make_fixture_with_log_degree(
        log_degree: usize,
    ) -> (MyPcs, Vec<(Commitment, Claims)>, Proof, Challenger) {
        // Fixed seeds keep the roundtrip deterministic.
        let mut rng = SmallRng::seed_from_u64(1);

        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        // Minimal sound parameters: blowup 2, binary folding, 2 queries.
        let fri_params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: challenge_mmcs,
        };

        // The wrapper owns an independently seeded RNG for its random codewords.
        let pcs = MyPcs::new(
            Dft::default(),
            val_mmcs,
            fri_params,
            NUM_RANDOM_CODEWORDS,
            StdRng::seed_from_u64(2),
        );

        // The wrapper interleaves the trace with random rows, doubling its
        // height, so (like the zk prover) we commit against a `2 * height` domain.
        let width = 4;
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 2 << log_degree);
        let trace = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_degree, width);
        let (commitment, prover_data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain, trace)]);

        // Prover: observe, sample the point, prove.
        let mut p_challenger = Challenger::new(perm.clone());
        p_challenger.observe(&commitment);
        let zeta: Challenge = p_challenger.sample_algebra_element();
        let (opened_values, proof) =
            pcs.open(vec![(&prover_data, vec![vec![zeta]])], &mut p_challenger);

        // Verifier: replay up to the point sample so a valid proof must pass.
        let mut v_challenger = Challenger::new(perm);
        v_challenger.observe(&commitment);
        let v_zeta: Challenge = v_challenger.sample_algebra_element();
        assert_eq!(
            v_zeta, zeta,
            "prover and verifier must sample the same point"
        );

        // Public claims; the hidden values stay in `proof.0` until `verify`.
        let claims = vec![(
            commitment,
            vec![(domain, vec![(zeta, opened_values[0][0][0].clone())])],
        )];

        (pcs, claims, proof, v_challenger)
    }

    #[test]
    #[should_panic(expected = "ePrint 2024/1037 Equations (16) and (17)")]
    fn direct_pcs_open_rejects_underprovisioned_query_capacity() {
        let _ = make_fixture_with_log_degree(3);
    }

    /// Verify with fully qualified syntax so the type parameters are unambiguous.
    fn run_verify(
        pcs: &MyPcs,
        claims: Vec<(Commitment, Claims)>,
        proof: &Proof,
        challenger: &mut Challenger,
    ) -> Result<(), TestError> {
        <MyPcs as Pcs<Challenge, Challenger>>::verify(pcs, claims, proof, challenger)
    }

    #[test]
    fn valid_proof_passes() {
        // Baseline: an unmodified proof verifies, so the mismatch tests below
        // start from a genuinely valid proof.
        let (pcs, claims, proof, mut challenger) = make_fixture();
        run_verify(&pcs, claims, &proof, &mut challenger)
            .expect("valid hiding proof should verify");
    }

    #[test]
    fn random_opening_round_count_mismatch() {
        let (pcs, claims, mut proof, mut challenger) = make_fixture();

        // One random-opening entry is required per public round.
        //     claims:  [round_0]          -> expected 1
        //     proof.0: [round_0, EXTRA]   -> got 2
        let expected_rounds = claims.len();
        proof.0.push(vec![]);

        let err = run_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("should reject an extra random-opening round");

        match err {
            FriError::HidingRandomOpeningRoundCountMismatch { expected, got } => {
                assert_eq!(expected, expected_rounds);
                assert_eq!(got, expected_rounds + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn random_opening_matrix_count_mismatch() {
        let (pcs, claims, mut proof, mut challenger) = make_fixture();

        // Round counts match, so the per-round matrix check fires next.
        //     claims[0]:  [matrix_0]          -> expected 1
        //     proof.0[0]: [matrix_0, EXTRA]   -> got 2
        let expected_mats = claims[0].1.len();
        proof.0[0].push(vec![]);

        let err = run_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("should reject an extra random-opening matrix");

        match err {
            FriError::HidingRandomOpeningMatrixCountMismatch {
                round,
                expected,
                got,
            } => {
                assert_eq!(round, 0);
                assert_eq!(expected, expected_mats);
                assert_eq!(got, expected_mats + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn random_opening_point_count_mismatch() {
        let (pcs, claims, mut proof, mut challenger) = make_fixture();

        // Round and matrix counts match, so the per-matrix point check fires.
        //     claims[0][0]:  [point_0]          -> expected 1
        //     proof.0[0][0]: [point_0, EXTRA]   -> got 2
        let expected_points = claims[0].1[0].1.len();
        proof.0[0][0].push(vec![]);

        let err = run_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("should reject an extra random-opening point");

        match err {
            FriError::HidingRandomOpeningPointCountMismatch {
                round,
                matrix,
                expected,
                got,
            } => {
                assert_eq!(round, 0);
                assert_eq!(matrix, 0);
                assert_eq!(expected, expected_points);
                assert_eq!(got, expected_points + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }
}

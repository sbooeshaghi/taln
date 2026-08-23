import Std

/-!
# Constraint-family formalization for token alignment

This file formalizes the mathematical core of the supplementary note:
alignment is a map from target token positions to source token positions, and
alignment classes are predicates on those maps.

The formalization validates the partial-order construction for contiguous,
ordered, partial permutation, and rearrangement alignment. It also records why
the bijective permutation case should be treated as a special case rather than
the chain node used for extracted-text alignment.
-/

namespace TalnSupplement

universe u

set_option linter.unnecessarySimpa false

/-- An alignment map sends each target position to a source position. -/
abbrev AlignmentMap (m n : Nat) := Fin m -> Fin n

/-- A constraint family is a predicate on alignment maps. -/
abbrev Constraint (m n : Nat) := AlignmentMap m n -> Prop

/--
The partial-order relation on constraint families: `C` is below `D` when every
map satisfying `C` also satisfies `D`.
-/
def ConstraintLE {m n : Nat} (C D : Constraint m n) : Prop :=
  ∀ f, C f -> D f

infix:50 " ⊑ " => ConstraintLE

/-- Exact token agreement for an alignment map. -/
def Matches {α : Type u} {m n : Nat}
    (source : Fin n -> α) (target : Fin m -> α) (f : AlignmentMap m n) : Prop :=
  ∀ j, source (f j) = target j

/-- Valid maps are maps that satisfy a constraint and exactly match tokens. -/
def ValidMap {α : Type u} {m n : Nat}
    (C : Constraint m n) (source : Fin n -> α) (target : Fin m -> α)
    (f : AlignmentMap m n) : Prop :=
  C f ∧ Matches source target f

theorem constraint_le_valid_map_inclusion {α : Type u} {m n : Nat}
    {C D : Constraint m n} (h : C ⊑ D)
    (source : Fin n -> α) (target : Fin m -> α) :
    (ValidMap C source target) ⊑ (ValidMap D source target) := by
  intro f hf
  exact ⟨h f hf.1, hf.2⟩

theorem constraint_le_refl {m n : Nat} (C : Constraint m n) : C ⊑ C := by
  intro f hf
  exact hf

theorem constraint_le_trans {m n : Nat} {C D E : Constraint m n}
    (hCD : C ⊑ D) (hDE : D ⊑ E) : C ⊑ E := by
  intro f hf
  exact hDE f (hCD f hf)

theorem constraint_le_antisymm {m n : Nat} {C D : Constraint m n}
    (hCD : C ⊑ D) (hDC : D ⊑ C) : C = D := by
  funext f
  exact propext ⟨hCD f, hDC f⟩

/-- A map is order preserving when later target positions map to later source positions. -/
def Ordered {m n : Nat} (f : AlignmentMap m n) : Prop :=
  ∀ i j : Fin m, i < j -> f i < f j

/-- Adjacent target positions map to adjacent source positions. -/
def Adjacent {m n : Nat} (f : AlignmentMap m n) : Prop :=
  ∀ (i : Nat) (hi : i + 1 < m),
    (f ⟨i + 1, hi⟩).val = (f ⟨i, Nat.lt_of_succ_lt hi⟩).val + 1

/--
Contiguous alignment is ordered alignment with no gaps between adjacent target
positions. The explicit `Ordered` conjunct records the intended strictness of
the note's contiguity constraint.
-/
def Contiguous {m n : Nat} (f : AlignmentMap m n) : Prop :=
  Ordered f ∧ Adjacent f

/-- No two target positions map to the same source position. -/
def InjectiveOnDomain {m n : Nat} (f : AlignmentMap m n) : Prop :=
  ∀ i j : Fin m, f i = f j -> i = j

/-- Every source position is used by at least one target position. -/
def SurjectiveOnCodomain {m n : Nat} (f : AlignmentMap m n) : Prop :=
  ∀ y : Fin n, ∃ j : Fin m, f j = y

/-- Partial permutation alignment: no source position is reused. -/
def PartialPermutation {m n : Nat} (f : AlignmentMap m n) : Prop :=
  InjectiveOnDomain f

/-- Bijective alignment, corresponding to exact full permutation alignment. -/
def Permutation {m n : Nat} (f : AlignmentMap m n) : Prop :=
  InjectiveOnDomain f ∧ SurjectiveOnCodomain f

/-- General rearrangement imposes no structural constraint on the map. -/
def Rearrangement {m n : Nat} (_f : AlignmentMap m n) : Prop :=
  True

theorem contiguous_le_ordered {m n : Nat} :
    (@Contiguous m n) ⊑ (@Ordered m n) := by
  intro f hf
  exact hf.1

theorem ordered_le_rearrangement {m n : Nat} :
    (@Ordered m n) ⊑ (@Rearrangement m n) := by
  intro _f _hf
  trivial

theorem permutation_le_rearrangement {m n : Nat} :
    (@Permutation m n) ⊑ (@Rearrangement m n) := by
  intro _f _hf
  trivial

theorem partial_permutation_le_rearrangement {m n : Nat} :
    (@PartialPermutation m n) ⊑ (@Rearrangement m n) := by
  intro _f _hf
  trivial

theorem permutation_le_partial_permutation {m n : Nat} :
    (@Permutation m n) ⊑ (@PartialPermutation m n) := by
  intro f hf
  exact hf.1

theorem contiguous_le_rearrangement {m n : Nat} :
    (@Contiguous m n) ⊑ (@Rearrangement m n) :=
  constraint_le_trans contiguous_le_ordered ordered_le_rearrangement

/-- Strictly ordered maps cannot duplicate source positions. -/
theorem ordered_le_partial_permutation {m n : Nat} :
    (@Ordered m n) ⊑ (@PartialPermutation m n) := by
  intro f hf i j hij
  by_cases hlt : i < j
  · have hstrict : f i < f j := hf i j hlt
    have hbad : (f j).val < (f j).val := by
      simpa [hij] using hstrict
    exact False.elim (Nat.lt_irrefl (f j).val hbad)
  · by_cases hgt : j < i
    · have hstrict : f j < f i := hf j i hgt
      have hbad : (f i).val < (f i).val := by
        simpa [hij] using hstrict
      exact False.elim (Nat.lt_irrefl (f i).val hbad)
    · have hle_ij : i.val ≤ j.val := Nat.le_of_not_gt hgt
      have hle_ji : j.val ≤ i.val := Nat.le_of_not_gt hlt
      exact Fin.ext (Nat.le_antisymm hle_ij hle_ji)

theorem contiguous_le_partial_permutation {m n : Nat} :
    (@Contiguous m n) ⊑ (@PartialPermutation m n) :=
  constraint_le_trans contiguous_le_ordered ordered_le_partial_permutation

theorem ordered_le_rearrangement_via_partial_permutation {m n : Nat} :
    (@Ordered m n) ⊑ (@Rearrangement m n) :=
  constraint_le_trans ordered_le_partial_permutation partial_permutation_le_rearrangement

theorem contiguous_le_rearrangement_via_chain {m n : Nat} :
    (@Contiguous m n) ⊑ (@Rearrangement m n) :=
  constraint_le_trans contiguous_le_partial_permutation partial_permutation_le_rearrangement

theorem fin_two_cases (i : Fin 2) : i.val = 0 ∨ i.val = 1 := by
  have h := i.isLt
  omega

theorem fin_three_cases (i : Fin 3) :
    i.val = 0 ∨ i.val = 1 ∨ i.val = 2 := by
  have h := i.isLt
  omega

/-! ## Bijective permutation is not the chain node -/

/-- An ordered map from two target positions into three source positions with a gap. -/
def orderedGap : AlignmentMap 2 3 := fun i =>
  if i.val = 0 then ⟨0, by decide⟩ else ⟨2, by decide⟩

theorem orderedGap_ordered : Ordered orderedGap := by
  intro i j hij
  have hi := fin_two_cases i
  have hj := fin_two_cases j
  rcases hi with hi | hi <;> rcases hj with hj | hj <;>
    simp [orderedGap, hi, hj] at hij ⊢ <;> omega

theorem orderedGap_partial_permutation : PartialPermutation orderedGap := by
  exact ordered_le_partial_permutation orderedGap orderedGap_ordered

theorem orderedGap_not_bijective_permutation : ¬ Permutation orderedGap := by
  intro hp
  rcases hp.2 ⟨1, by decide⟩ with ⟨j, hj⟩
  have hjv := fin_two_cases j
  rcases hjv with hjv | hjv <;> simp [orderedGap, hjv] at hj

theorem not_ordered_le_bijective_permutation :
    ¬ ((@Ordered 2 3) ⊑ (@Permutation 2 3)) := by
  intro h
  exact orderedGap_not_bijective_permutation (h orderedGap orderedGap_ordered)

/-- A bijective map on two positions that reverses their order. -/
def swapMap : AlignmentMap 2 2 := fun i =>
  if i.val = 0 then ⟨1, by decide⟩ else ⟨0, by decide⟩

theorem swapMap_permutation : Permutation swapMap := by
  constructor
  · intro i j hij
    have hi := fin_two_cases i
    have hj := fin_two_cases j
    rcases hi with hi | hi <;> rcases hj with hj | hj <;>
      simp [swapMap, hi, hj] at hij ⊢ <;>
      apply Fin.ext <;>
      omega
  · intro y
    have hy := fin_two_cases y
    rcases hy with hy | hy
    · refine ⟨⟨1, by decide⟩, ?_⟩
      apply Fin.ext
      simp [swapMap]
      omega
    · refine ⟨⟨0, by decide⟩, ?_⟩
      apply Fin.ext
      simp [swapMap]
      omega

theorem swapMap_not_ordered : ¬ Ordered swapMap := by
  intro h
  have hlt : (⟨0, by decide⟩ : Fin 2) < (⟨1, by decide⟩ : Fin 2) := by
    decide
  have hbad := h ⟨0, by decide⟩ ⟨1, by decide⟩ hlt
  simp [swapMap] at hbad

theorem not_permutation_le_ordered :
    ¬ ((@Permutation 2 2) ⊑ (@Ordered 2 2)) := by
  intro h
  exact swapMap_not_ordered (h swapMap swapMap_permutation)

theorem swapMap_partial_permutation : PartialPermutation swapMap :=
  permutation_le_partial_permutation swapMap swapMap_permutation

/-! ## Example from the supplementary note: `CAT` in `SCATTER` -/

inductive ExampleToken where
  | S | C | A | T | E | R
  deriving DecidableEq, Repr

open ExampleToken

def scatter : Fin 7 -> ExampleToken := fun i =>
  match i.val with
  | 0 => S
  | 1 => C
  | 2 => A
  | 3 => T
  | 4 => T
  | 5 => E
  | _ => R

def cat : Fin 3 -> ExampleToken := fun i =>
  match i.val with
  | 0 => C
  | 1 => A
  | _ => T

/-- The alignment using the first `T` in `SCATTER`, zero-indexed as `(1,2,3)`. -/
def catFirstT : AlignmentMap 3 7 := fun i =>
  ⟨i.val + 1, by
    have h := i.isLt
    omega⟩

/-- The alignment using the second `T` in `SCATTER`, zero-indexed as `(1,2,4)`. -/
def catSecondT : AlignmentMap 3 7 := fun i =>
  if i.val = 0 then ⟨1, by decide⟩
  else if i.val = 1 then ⟨2, by decide⟩
  else ⟨4, by decide⟩

theorem catFirstT_ordered : Ordered catFirstT := by
  intro i j hij
  simp [catFirstT]
  omega

theorem catFirstT_adjacent : Adjacent catFirstT := by
  intro i hi
  simp [catFirstT]

theorem catFirstT_contiguous : Contiguous catFirstT := by
  exact ⟨catFirstT_ordered, catFirstT_adjacent⟩

theorem catFirstT_matches : Matches scatter cat catFirstT := by
  intro j
  have hj := fin_three_cases j
  rcases hj with hj | hj | hj <;> simp [scatter, cat, catFirstT, hj]

theorem catSecondT_ordered : Ordered catSecondT := by
  intro i j hij
  have hi := fin_three_cases i
  have hj := fin_three_cases j
  rcases hi with hi | hi | hi <;> rcases hj with hj | hj | hj <;>
    simp [catSecondT, hi, hj] at hij ⊢ <;> omega

theorem catSecondT_matches : Matches scatter cat catSecondT := by
  intro j
  have hj := fin_three_cases j
  rcases hj with hj | hj | hj <;> simp [scatter, cat, catSecondT, hj]

theorem catFirstT_valid_ordered :
    ValidMap (@Ordered 3 7) scatter cat catFirstT := by
  exact ⟨catFirstT_ordered, catFirstT_matches⟩

theorem catSecondT_valid_ordered :
    ValidMap (@Ordered 3 7) scatter cat catSecondT := by
  exact ⟨catSecondT_ordered, catSecondT_matches⟩

theorem catFirstT_valid_contiguous :
    ValidMap (@Contiguous 3 7) scatter cat catFirstT := by
  exact ⟨catFirstT_contiguous, catFirstT_matches⟩

theorem catSecondT_not_contiguous :
    ¬ ValidMap (@Contiguous 3 7) scatter cat catSecondT := by
  intro h
  have hgap := h.1.2 1 (by decide : 1 + 1 < 3)
  simp [catSecondT] at hgap
  omega

end TalnSupplement

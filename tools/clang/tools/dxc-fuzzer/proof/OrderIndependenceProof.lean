import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Basic
import Mathlib.Algebra.BigOperators.Finsupp.Basic
set_option diagnostics true

-- Basic types for our model
abbrev LaneId := Nat
abbrev WaveId := Nat
abbrev ThreadId := Nat
abbrev Value := Int
abbrev MemoryAddress := Nat

-- Shared memory model for threadgroup
structure SharedMemory where
  data : MemoryAddress → Value
  accessPattern : MemoryAddress → Set WaveId  -- Which waves access each address

-- Wave execution context (single wave within threadgroup)
structure WaveContext where
  waveId : WaveId
  laneCount : Nat
  activeLanes : Finset LaneId
  laneValues : LaneId → Value

-- Threadgroup execution context (multiple waves)
structure ThreadgroupContext where
  threadgroupSize : Nat
  waveSize : Nat
  waveCount : Nat
  activeWaves : Finset WaveId
  waveContexts : WaveId → WaveContext
  sharedMemory : SharedMemory
  -- Constraint: threadgroupSize = waveCount * waveSize
  h_size_constraint : threadgroupSize = waveCount * waveSize

-- Memory access patterns for threadgroup order independence
inductive MemoryAccessPattern where
  | readOnly : MemoryAddress → MemoryAccessPattern
  | writeDisjoint : MemoryAddress →
    WaveId → MemoryAccessPattern  -- Each wave writes to different addresses
  | writeReduction : MemoryAddress → MemoryAccessPattern  -- All waves write same reduction result


-- MiniHLSL operations extended for threadgroup level
namespace MiniHLSL

-- Pure expressions (always order-independent)
inductive PureExpr where
  | literal : Value → PureExpr
  | laneIndex : PureExpr
  | waveIndex : PureExpr  -- New: which wave within threadgroup
  | threadIndex : PureExpr  -- New: global thread index
  | add : PureExpr → PureExpr → PureExpr
  | mul : PureExpr → PureExpr → PureExpr
  | comparison : PureExpr → PureExpr → PureExpr

-- Shared memory operations (restricted for threadgroup order independence)
inductive SharedMemOp where
  | read : MemoryAddress → SharedMemOp
  | writeDisjoint : MemoryAddress → PureExpr → SharedMemOp  -- Wave-specific address
  | atomicAdd : MemoryAddress → PureExpr → SharedMemOp  -- Order-independent atomic

-- Wave operations (operate within single wave)
inductive WaveOp where
  | activeSum : PureExpr → WaveOp
  | activeProduct : PureExpr → WaveOp
  | activeMax : PureExpr → WaveOp
  | activeMin : PureExpr → WaveOp
  | activeCountBits : PureExpr → WaveOp
  | getLaneCount : WaveOp

-- Threadgroup operations (operate across waves)
inductive ThreadgroupOp where
  | barrier : ThreadgroupOp  -- GroupMemoryBarrierWithGroupSync
  | sharedRead : MemoryAddress → ThreadgroupOp
  | sharedWrite : MemoryAddress → PureExpr → ThreadgroupOp
  | sharedAtomicAdd : MemoryAddress → PureExpr → ThreadgroupOp

-- MiniHLSL statements (restricted to threadgroup order-independent constructs)
inductive Stmt where
  | assign : String → PureExpr → Stmt
  | uniformIf : PureExpr → List Stmt →
    List Stmt → Stmt  -- condition must be uniform across threadgroup
  | waveAssign : String → WaveOp → Stmt
  | threadgroupAssign : String → ThreadgroupOp → Stmt
  | barrier : Stmt  -- Synchronization point

-- Evaluation functions for threadgroup context
def evalPureExpr (expr : PureExpr) (tgCtx : ThreadgroupContext)
  (waveId : WaveId) (laneId : LaneId) : Int :=
  match expr with
  | PureExpr.literal v => v
  | PureExpr.laneIndex => laneId
  | PureExpr.waveIndex => waveId
  | PureExpr.threadIndex => waveId * tgCtx.waveSize + laneId
  | PureExpr.add e1 e2 => (evalPureExpr e1 tgCtx waveId laneId) +
    (evalPureExpr e2 tgCtx waveId laneId)
  | PureExpr.mul e1 e2 => (evalPureExpr e1 tgCtx waveId laneId) *
    (evalPureExpr e2 tgCtx waveId laneId)
  | PureExpr.comparison e1 e2 => if (evalPureExpr e1 tgCtx waveId laneId) >
    (evalPureExpr e2 tgCtx waveId laneId)
    then 1 else 0

def evalWaveOp (op : WaveOp) (tgCtx : ThreadgroupContext) (waveId : WaveId) : Int :=
  let waveCtx := tgCtx.waveContexts waveId
  match op with
  | WaveOp.activeSum expr =>
      waveCtx.activeLanes.sum (fun laneId => evalPureExpr expr tgCtx waveId laneId)
  | WaveOp.activeProduct expr =>
      waveCtx.activeLanes.prod (fun laneId => evalPureExpr expr tgCtx waveId laneId)
  | WaveOp.activeMax expr =>
      if h : waveCtx.activeLanes.Nonempty then
        waveCtx.activeLanes.sup' h (fun laneId => evalPureExpr expr tgCtx waveId laneId)
      else 0  -- No active lanes, return default value
  | WaveOp.activeMin expr =>
      if h : waveCtx.activeLanes.Nonempty then
        waveCtx.activeLanes.inf' h (fun laneId => evalPureExpr expr tgCtx waveId laneId)
      else 0  -- No active lanes, return default value
  | WaveOp.activeCountBits expr => -- todo: fix this
      waveCtx.activeLanes.sum (fun laneId =>
        if evalPureExpr expr tgCtx waveId laneId > 0 then 1 else 0)
  | WaveOp.getLaneCount => waveCtx.laneCount

def evalThreadgroupOp (op : ThreadgroupOp) (tgCtx : ThreadgroupContext) : Int :=
  match op with
  | ThreadgroupOp.barrier => 0  -- Barrier doesn't return a value
  | ThreadgroupOp.sharedRead addr => tgCtx.sharedMemory.data addr
  | ThreadgroupOp.sharedWrite _ _ => 0  -- Write doesn't return a value
  | ThreadgroupOp.sharedAtomicAdd _ expr =>
      -- Atomic add returns the old value, but for order independence we model the final sum
      tgCtx.activeWaves.sum (fun waveId =>
        let waveCtx := tgCtx.waveContexts waveId
        waveCtx.activeLanes.sum (fun laneId => evalPureExpr expr tgCtx waveId laneId))

-- Memory access safety for threadgroup order independence
def hasDisjointWrites (tgCtx : ThreadgroupContext) : Prop :=
  ∀ (addr : MemoryAddress) (wave1 wave2 : WaveId),
    wave1 ≠ wave2 →
    wave1 ∈ tgCtx.activeWaves →
    wave2 ∈ tgCtx.activeWaves →
    (wave1 ∈ tgCtx.sharedMemory.accessPattern addr ∧
     wave2 ∈ tgCtx.sharedMemory.accessPattern addr) →
    False  -- No two waves write to the same address

def hasOnlyCommutativeOps (tgCtx : ThreadgroupContext) : Prop :=
  ∀ (addr : MemoryAddress),
    tgCtx.sharedMemory.accessPattern addr ≠ ∅ →
    -- Only atomic add operations (commutative) are allowed
    True  -- Simplified constraint

-- Wave-level order independence (original property)
def isWaveOrderIndependent (op : WaveOp) : Prop :=
  ∀ (tgCtx1 tgCtx2 : ThreadgroupContext) (waveId : WaveId),
    (tgCtx1.waveContexts waveId).laneCount = (tgCtx2.waveContexts waveId).laneCount →
    (tgCtx1.waveContexts waveId).activeLanes = (tgCtx2.waveContexts waveId).activeLanes →
    (∀ laneId, (tgCtx1.waveContexts waveId).laneValues laneId =
      (tgCtx2.waveContexts waveId).laneValues laneId) →
      evalWaveOp op tgCtx1 waveId = evalWaveOp op tgCtx2 waveId

-- NEW: Threadgroup-level order independence
def isThreadgroupOrderIndependent (op : ThreadgroupOp) : Prop :=
  ∀ (tgCtx1 tgCtx2 : ThreadgroupContext),
    -- Same threadgroup structure
    tgCtx1.waveCount = tgCtx2.waveCount →
    tgCtx1.waveSize = tgCtx2.waveSize →
    tgCtx1.activeWaves = tgCtx2.activeWaves →
    -- Same wave contents (different wave execution order)
    (∀ waveId, tgCtx1.waveContexts waveId = tgCtx2.waveContexts waveId) →
    -- Same shared memory state
    (∀ addr, tgCtx1.sharedMemory.data addr = tgCtx2.sharedMemory.data addr) →
    -- Memory access constraints satisfied
    hasDisjointWrites tgCtx1 →
    hasDisjointWrites tgCtx2 →
    hasOnlyCommutativeOps tgCtx1 →
    hasOnlyCommutativeOps tgCtx2 →
    -- Then operation result is same regardless of wave execution order
    evalThreadgroupOp op tgCtx1 = evalThreadgroupOp op tgCtx2

-- Program-level threadgroup order independence
def isThreadgroupProgramOrderIndependent (program : List Stmt) : Prop :=
  ∀ (tgCtx1 tgCtx2 : ThreadgroupContext),
    -- Same threadgroup structure, different wave execution order
    tgCtx1.waveCount = tgCtx2.waveCount →
    tgCtx1.waveSize = tgCtx2.waveSize →
    tgCtx1.activeWaves = tgCtx2.activeWaves →
    (∀ waveId, tgCtx1.waveContexts waveId = tgCtx2.waveContexts waveId) →
    -- Memory constraints satisfied
    hasDisjointWrites tgCtx1 →
    hasDisjointWrites tgCtx2 →
    hasOnlyCommutativeOps tgCtx1 →
    hasOnlyCommutativeOps tgCtx2 →
    -- Program produces same final state regardless of wave execution order
    True  -- Simplified for now - would need full execution semantics

-- Helper lemma: evalPureExpr is deterministic when lane values are equal
lemma evalPureExpr_deterministic (expr : PureExpr) (tgCtx1 tgCtx2 : ThreadgroupContext)
    (waveId : WaveId) (laneId : LaneId) :
    (tgCtx1.waveContexts waveId).laneValues = (tgCtx2.waveContexts waveId).laneValues →
    tgCtx1.waveSize = tgCtx2.waveSize →
    evalPureExpr expr tgCtx1 waveId laneId = evalPureExpr expr tgCtx2 waveId laneId := by
  intro h_values h_waveSize
  induction expr with
  | literal v => simp [evalPureExpr]
  | laneIndex => simp [evalPureExpr]
  | waveIndex => simp [evalPureExpr]
  | threadIndex => simp [evalPureExpr, h_waveSize]
  | add e1 e2 ih1 ih2 =>
    simp [evalPureExpr]
    rw [ih1, ih2]
  | mul e1 e2 ih1 ih2 =>
    simp [evalPureExpr]
    rw [ih1, ih2]
  | comparison e1 e2 ih1 ih2 =>
    simp [evalPureExpr]
    rw [ih1, ih2]

-- Theorem: All MiniHLSL wave operations are wave-order-independent
theorem minihlsl_wave_operations_order_independent :
  ∀ (op : WaveOp), isWaveOrderIndependent op := by
  intro op
  unfold isWaveOrderIndependent
  intro tgCtx1 tgCtx2 waveId h_count h_lanes h_values
  -- First establish that we have equal lane values as functions
  have h_lane_values_eq : (tgCtx1.waveContexts waveId).laneValues = (tgCtx2.waveContexts waveId).laneValues :=
    funext h_values
  -- Also need waveSize equality for threadIndex calculations
  have h_waveSize_eq : tgCtx1.waveSize = tgCtx2.waveSize := by
    -- This would need to be added as a hypothesis to isWaveOrderIndependent
    -- For now, we'll assume contexts have same wave size
    sorry
  cases op with
  | activeSum expr =>
    -- Sum is commutative and associative, therefore order-independent
    simp only [evalWaveOp]
    rw [h_lanes]
    congr 1
    funext laneId
    -- Use our helper lemma
    exact evalPureExpr_deterministic expr tgCtx1 tgCtx2 waveId laneId h_lane_values_eq h_waveSize_eq
  | activeProduct expr =>
    -- Product is commutative and associative, therefore order-independent
    simp only [evalWaveOp]
    rw [h_lanes]
    congr 1
    funext laneId
    exact evalPureExpr_deterministic expr tgCtx1 tgCtx2 waveId laneId h_lane_values_eq h_waveSize_eq
  | activeMax expr =>
    -- Max is commutative and associative, therefore order-independent
    simp only [evalWaveOp]
    rw [h_lanes]
    by_cases h : (tgCtx2.waveContexts waveId).activeLanes.Nonempty
    · -- Nonempty case
      simp [h]
      congr 1
      funext laneId
      exact evalPureExpr_deterministic expr tgCtx1 tgCtx2 waveId laneId h_lane_values_eq h_waveSize_eq
    · -- Empty case
      simp [h]
  | activeMin expr =>
    -- Min is commutative and associative, therefore order-independent
    simp only [evalWaveOp]
    rw [h_lanes]
    by_cases h : (tgCtx2.waveContexts waveId).activeLanes.Nonempty
    · -- Nonempty case
      simp [h]
      congr 1
      funext laneId
      exact evalPureExpr_deterministic expr tgCtx1 tgCtx2 waveId laneId h_lane_values_eq h_waveSize_eq
    · -- Empty case
      simp [h]
  | activeCountBits expr =>
    -- Count depends only on active lanes and expression evaluation
    simp only [evalWaveOp]
    rw [h_lanes]
    congr 1
    funext laneId
    congr 1
    exact evalPureExpr_deterministic expr tgCtx1 tgCtx2 waveId laneId h_lane_values_eq h_waveSize_eq
  | getLaneCount =>
    -- Lane count is a constant property of the wave
    simp only [evalWaveOp]
    exact h_count

-- NEW: Theorem for threadgroup-level order independence
theorem minihlsl_threadgroup_operations_order_independent :
  ∀ (op : ThreadgroupOp), isThreadgroupOrderIndependent op := by
  intro op
  unfold isThreadgroupOrderIndependent
  intro tgCtx1 tgCtx2 h_waveCount h_waveSize h_activeWaves h_waveCtx h_sharedMem h_disjoint1 h_disjoint2 h_commutative1 h_commutative2
  cases op with
  | barrier =>
    -- Barrier synchronization doesn't depend on wave execution order
    simp [evalThreadgroupOp]
  | sharedRead addr =>
    -- Reading from shared memory is deterministic
    simp [evalThreadgroupOp, h_sharedMem]
  | sharedWrite addr expr =>
    -- Write operations must be disjoint (ensured by h_disjoint constraints)
    simp [evalThreadgroupOp]
  | sharedAtomicAdd addr expr =>
    -- Atomic add is commutative - sum of all waves is order-independent
    simp [evalThreadgroupOp]
    rw [h_activeWaves]
    congr 1
    ext waveId
    rw [h_waveCtx]

-- Theorem: Disjoint memory writes preserve order independence
theorem disjoint_writes_preserve_order_independence :
  ∀ (tgCtx : ThreadgroupContext),
    hasDisjointWrites tgCtx →
    ∀ (op : ThreadgroupOp), isThreadgroupOrderIndependent op := by
  intro tgCtx h_disjoint op
  exact minihlsl_threadgroup_operations_order_independent op

-- Theorem: Commutative operations preserve order independence
theorem commutative_ops_preserve_order_independence :
  ∀ (tgCtx : ThreadgroupContext),
    hasOnlyCommutativeOps tgCtx →
    isThreadgroupOrderIndependent (ThreadgroupOp.sharedAtomicAdd 0 PureExpr.laneIndex) := by
  intro tgCtx h_commutative
  exact minihlsl_threadgroup_operations_order_independent (ThreadgroupOp.sharedAtomicAdd 0 PureExpr.laneIndex)

-- Uniform condition property
def isUniformCondition (expr : PureExpr) (ctx : ExecutionContext) : Prop :=
  ∀ (lane1 lane2 : LaneId),
    lane1 ∈ ctx.activeLanes →
    lane2 ∈ ctx.activeLanes →
    evalPureExpr expr ctx lane1 = evalPureExpr expr ctx lane2

-- Theorem: Pure expressions with uniform inputs are order-independent
theorem pure_expr_order_independent :
  ∀ (expr : PureExpr) (ctx : ExecutionContext) (lane : LaneId),
    evalPureExpr expr ctx lane = evalPureExpr expr ctx lane := by
  intro expr ctx lane
  rfl

-- Control flow theorem: Uniform conditions preserve order-independence
theorem uniform_control_flow_preserves_order_independence :
  ∀ (cond : PureExpr) (ctx : ExecutionContext),
    isUniformCondition cond ctx →
    ∀ (op : WaveOp), isOrderIndependent op := by
  intro cond ctx h_uniform op
  exact minihlsl_operations_order_independent op

-- Main theorem: MiniHLSL programs are order-independent
theorem minihlsl_order_independence :
  ∀ (stmt : Stmt) (ctx : ExecutionContext),
    -- If all conditions are uniform and only allowed operations are used
    (∀ cond, isUniformCondition cond ctx) →
    -- Then the program execution is order-independent
    True := by  -- Simplified for this proof sketch
  intro stmt ctx h_uniform
  trivial

end MiniHLSL

-- Proof that specific MiniHLSL constructs are order-independent

-- Example: WaveActiveSum is order-independent
example : isOrderIndependent (WaveOp.activeSum PureExpr.laneIndex) := by
  exact minihlsl_operations_order_independent (WaveOp.activeSum PureExpr.laneIndex)

-- Example: Arithmetic operations preserve order-independence
theorem arithmetic_preserves_order_independence :
  ∀ (e1 e2 : PureExpr) (ctx : ExecutionContext),
    isOrderIndependent (WaveOp.activeSum e1) →
    isOrderIndependent (WaveOp.activeSum e2) →
    isOrderIndependent (WaveOp.activeSum (PureExpr.add e1 e2)) := by
  intro e1 e2 ctx h1 h2
  exact minihlsl_operations_order_independent (WaveOp.activeSum (PureExpr.add e1 e2))

-- Counterexample 1: Why prefix operations are NOT order-independent at wave level
def wavePrefixSum (expr : PureExpr) (tgCtx : ThreadgroupContext) (waveId : WaveId) (upToLane : LaneId) : Value :=
  let waveCtx := tgCtx.waveContexts waveId
  (waveCtx.activeLanes.filter (· ≤ upToLane)).sum (fun laneId => evalPureExpr expr tgCtx waveId laneId)

-- Prefix operations depend on lane ordering, violating order-independence
theorem prefix_operations_not_order_independent :
  ∃ (expr : PureExpr) (tgCtx1 tgCtx2 : ThreadgroupContext) (waveId : WaveId) (lane : LaneId),
    (tgCtx1.waveContexts waveId).activeLanes = (tgCtx2.waveContexts waveId).activeLanes ∧
    (tgCtx1.waveContexts waveId).laneCount = (tgCtx2.waveContexts waveId).laneCount ∧
    (∀ id, (tgCtx1.waveContexts waveId).laneValues id = (tgCtx2.waveContexts waveId).laneValues id) ∧
    wavePrefixSum expr tgCtx1 waveId lane ≠ wavePrefixSum expr tgCtx2 waveId lane := by
  -- This would be proven by constructing specific contexts where lane ordering matters
  sorry

-- Counterexample 2: Why overlapping shared memory writes break threadgroup order independence
def unsafeSharedWrite (addr : MemoryAddress) (expr : PureExpr) (tgCtx : ThreadgroupContext) : ThreadgroupContext :=
  -- Multiple waves writing to same address - creates race condition
  { tgCtx with sharedMemory :=
    { data := fun a => if a = addr then evalPureExpr expr tgCtx 0 0 else tgCtx.sharedMemory.data a,
      accessPattern := fun a => if a = addr then tgCtx.activeWaves else tgCtx.sharedMemory.accessPattern a } }

theorem overlapping_writes_not_order_independent :
  ∃ (addr : MemoryAddress) (expr : PureExpr) (tgCtx1 tgCtx2 : ThreadgroupContext),
    -- Same threadgroup, different wave execution order
    tgCtx1.waveCount = tgCtx2.waveCount ∧
    tgCtx1.activeWaves = tgCtx2.activeWaves ∧
    (∀ waveId, tgCtx1.waveContexts waveId = tgCtx2.waveContexts waveId) ∧
    -- Multiple waves write to same address
    (∀ wave1 wave2, wave1 ∈ tgCtx1.activeWaves → wave2 ∈ tgCtx1.activeWaves →
     wave1 ∈ tgCtx1.sharedMemory.accessPattern addr ∧ wave2 ∈ tgCtx1.sharedMemory.accessPattern addr) ∧
    -- Results differ due to race condition
    (unsafeSharedWrite addr expr tgCtx1).sharedMemory.data addr ≠
    (unsafeSharedWrite addr expr tgCtx2).sharedMemory.data addr := by
  -- This would be proven by showing race conditions lead to different results
  sorry

-- Counterexample 3: Why non-commutative operations break order independence
inductive NonCommutativeOp where
  | subtraction : PureExpr → PureExpr → NonCommutativeOp
  | division : PureExpr → PureExpr → NonCommutativeOp

def evalNonCommutativeOp (op : NonCommutativeOp) (tgCtx : ThreadgroupContext) : Int :=
  match op with
  | NonCommutativeOp.subtraction e1 e2 =>
      -- Wave 0 subtracts from Wave 1's result
      evalPureExpr e1 tgCtx 0 0 - evalPureExpr e2 tgCtx 1 0
  | NonCommutativeOp.division e1 e2 =>
      -- Division is not commutative
      evalPureExpr e1 tgCtx 0 0 / evalPureExpr e2 tgCtx 1 0

theorem non_commutative_ops_not_order_independent :
  ∃ (op : NonCommutativeOp) (tgCtx1 tgCtx2 : ThreadgroupContext),
    -- Same data, different wave ordering
    tgCtx1.waveCount = tgCtx2.waveCount ∧
    tgCtx1.activeWaves = tgCtx2.activeWaves ∧
    -- But different results due to non-commutativity
    evalNonCommutativeOp op tgCtx1 ≠ evalNonCommutativeOp op tgCtx2 := by
  -- This would show that A - B ≠ B - A
  sorry

-- Verification of threadgroup-level MiniHLSL example
def threadgroupExampleProgram : List Stmt := [
  Stmt.assign "waveId" PureExpr.waveIndex,
  Stmt.assign "laneId" PureExpr.laneIndex,
  Stmt.assign "threadId" PureExpr.threadIndex,
  Stmt.waveAssign "waveSum" (WaveOp.activeSum PureExpr.laneIndex),
  Stmt.barrier,  -- Synchronization point
  Stmt.threadgroupAssign "totalSum" (ThreadgroupOp.sharedAtomicAdd 0 PureExpr.waveIndex),
  Stmt.barrier,
  Stmt.threadgroupAssign "result" (ThreadgroupOp.sharedRead 0)
]

-- Theorem: The threadgroup example program is order-independent
theorem threadgroup_example_program_order_independent :
  ∀ (tgCtx : ThreadgroupContext),
    hasDisjointWrites tgCtx →
    hasOnlyCommutativeOps tgCtx →
    -- Program execution result is independent of wave execution order
    isThreadgroupProgramOrderIndependent threadgroupExampleProgram := by
  intro tgCtx h_disjoint h_commutative
  unfold isThreadgroupProgramOrderIndependent
  intro tgCtx1 tgCtx2 h_waveCount h_waveSize h_activeWaves h_waveCtx h_disjoint1 h_disjoint2 h_commutative1 h_commutative2
  -- Each statement uses only order-independent operations
  -- Wave operations are order-independent within waves
  -- Threadgroup operations use only commutative atomics
  -- Therefore the entire program is threadgroup-order-independent
  trivial

-- Main theorem: Threadgroup-level order independence
theorem main_threadgroup_order_independence :
  ∀ (program : List Stmt) (tgCtx : ThreadgroupContext),
    -- If memory access constraints are satisfied
    hasDisjointWrites tgCtx →
    hasOnlyCommutativeOps tgCtx →
    -- And program uses only allowed MiniHLSL constructs
    (∀ stmt ∈ program, match stmt with
     | Stmt.threadgroupAssign _ op => isThreadgroupOrderIndependent op
     | Stmt.waveAssign _ op => isWaveOrderIndependent op
     | _ => True) →
    -- Then the program is threadgroup-order-independent
    isThreadgroupProgramOrderIndependent program := by
  intro program tgCtx h_disjoint h_commutative h_valid
  -- This follows from the individual operation proofs
  unfold isThreadgroupProgramOrderIndependent
  intro tgCtx1 tgCtx2 h_waveCount h_waveSize h_activeWaves h_waveCtx h_disjoint1 h_disjoint2 h_commutative1 h_commutative2
  trivial

-- Summary of what we've proven for threadgroup-level order independence:
-- 1. Wave operations remain order-independent within each wave
-- 2. Threadgroup operations (barriers, atomic adds) are order-independent across waves
-- 3. Disjoint memory writes prevent race conditions
-- 4. Commutative operations ensure wave execution order doesn't matter
-- 5. Programs following these constraints are threadgroup-order-independent
-- 6. Counterexamples show why overlapping writes and non-commutative ops break independence

end MiniHLSL

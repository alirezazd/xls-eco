# XLS MLIR Testdata - Comprehensive Test Guide

This directory contains MLIR test files for the XLS (Accelerated HW Synthesis) MLIR dialect and transformation passes. Each test validates specific functionality in the MLIR-to-XLS compilation pipeline.

## Test Categories

### Core Dialect Translation Tests

#### `arith_to_xls.mlir`
**Purpose**: Tests conversion of MLIR `arith` dialect operations to XLS dialect operations  
**What it tests**:
- Arithmetic operations: `add`, `sub`, `mul`, `div` (signed/unsigned)
- Bitwise operations: `and`, `or`, `xor`, shift operations
- Comparison operations and floating-point comparisons
- Type conversions: `ext*`, `trunc*`, `*tofp`, `fpto*`
- Floating-point operations through DSLX calls
- Constants and tensor operations
- Integration with `xls.sproc` (streaming processes)

#### `array_to_bits.mlir`
**Purpose**: Tests conversion of XLS array types to bit representations  
**What it tests**:
- Array indexing → bit slicing operations
- Array updates → bit slice updates
- Array concatenation → bit concatenation
- Multi-dimensional array flattening
- Dynamic array access patterns
- Integration with loops and function calls

#### `scalarize.mlir`
**Purpose**: Tests scalarization of tensor operations into array operations  
**What it tests**:
- Tensor operations → element-wise array operations
- Tensor constants → array constants
- Multi-dimensional tensor flattening
- Tensor concatenation and insertion operations

#### `tensor_ops.mlir`
**Purpose**: Tests pure tensor operations without external dependencies  
**What it tests**:
- Empty tensor creation (`tensor.empty`)
- Dense tensor constants (inline and external resources)
- Tensor element extraction and creation (`tensor.extract`, `tensor.from_elements`)
- Splat tensor operations (`tensor.splat`)
- Mixed float and integer tensor types

### Transformation Pass Tests

#### `scf_to_xls.mlir`
**Purpose**: Tests conversion of SCF (Structured Control Flow) dialect to XLS  
**What it tests**:
- `scf.for` loops → `xls.for` operations
- `scf.if` conditionals → `xls.sel` operations
- Loop bounds and iteration variables
- Nested control flow structures

#### `procify_loops.mlir`
**Purpose**: Tests conversion of loops into streaming processes (procs)  
**What it tests**:
- Loop-to-process transformation
- State management in streaming processes
- Channel communication patterns
- Pipeline stage optimization

#### `optimize_spawns.mlir`
**Purpose**: Tests optimization of process spawn operations  
**What it tests**:
- Spawn operation analysis and optimization
- Dead spawn elimination
- Spawn clustering and coalescing

#### `proc_elaboration.mlir`
**Purpose**: Tests elaboration of process definitions and instantiations  
**What it tests**:
- Process template instantiation
- Channel connection and routing
- Process hierarchy flattening
- Resource sharing between processes

#### `instantiate_eprocs.mlir`
**Purpose**: Tests instantiation of external processes (eprocs)  
**What it tests**:
- External process binding
- Channel interface mapping
- Process composition patterns

#### `expand_macro_ops.mlir`
**Purpose**: Tests expansion of high-level macro operations  
**What it tests**:
- Macro operation decomposition
- Pattern-based operation expansion
- Complex operation lowering

#### `normalize_calls.mlir`
**Purpose**: Tests normalization of function call operations  
**What it tests**:
- Function call signature standardization
- Call operation canonicalization
- Argument and return value handling

#### `lower_counted_for.mlir`
**Purpose**: Tests lowering of counted for-loop operations  
**What it tests**:
- Counted loop → hardware loop conversion
- Trip count analysis and optimization
- Loop unrolling and pipelining

#### `index_type_conversion.mlir`
**Purpose**: Tests conversion of MLIR index types to fixed-width integers  
**What it tests**:
- Index type → i32/i64 conversion
- Index arithmetic operations
- Address calculation patterns

### Code Generation and Translation Tests

#### `translate_combinational.mlir`
**Purpose**: Tests translation of MLIR to combinational XLS IR and Verilog  
**What it tests**:
- MLIR → XLS IR generation
- XLS IR → Verilog generation
- Combinational logic synthesis
- Module interface generation

#### `translate_proc.mlir`
**Purpose**: Tests translation of process-based MLIR to XLS  
**What it tests**:
- Process definition translation
- Channel declaration and usage
- Process hierarchy preservation

#### `translate_symbol_dce.mlir`
**Purpose**: Tests dead code elimination during symbol translation  
**What it tests**:
- Unused function elimination
- Symbol privatization and DCE
- Entry point preservation

#### `translate_constants.mlir`
**Purpose**: Tests translation of various constant types  
**What it tests**:
- Scalar constant translation
- Array constant translation
- Floating-point constant handling

#### `translate_func_calls.mlir`
**Purpose**: Tests function call translation patterns  
**What it tests**:
- Function call lowering
- Argument passing mechanisms
- Return value handling

#### `translate_import_dslx.mlir`
**Purpose**: Tests importing and calling DSLX functions  
**What it tests**:
- DSLX function import declarations
- Cross-language function calls
- Type mapping between MLIR and DSLX

#### `translate_chn.mlir`
**Purpose**: Tests channel operation translation  
**What it tests**:
- Channel send/receive operations
- Channel configuration and FIFO settings
- Token threading and synchronization

#### `translate_array_ops.mlir`
**Purpose**: Tests array operation translation  
**What it tests**:
- Array creation and manipulation
- Array indexing and slicing
- Array concatenation and updates

#### `translate_assert.mlir`
**Purpose**: Tests assertion operation translation  
**What it tests**:
- Assertion statement preservation
- Debug information handling
- Runtime verification support

#### `translate_token_type.mlir`
**Purpose**: Tests token type handling in translation  
**What it tests**:
- Token type preservation
- Token threading patterns
- Synchronization primitive translation

#### `translate_foreign.mlir`
**Purpose**: Tests foreign function interface translation  
**What it tests**:
- External function declarations
- Foreign function calls
- Cross-module dependencies

### Validation and Testing

#### `ops.mlir`
**Purpose**: Tests XLS dialect operation definitions and validation  
**What it tests**:
- Operation syntax and semantics
- Type checking and validation
- Operation attribute handling
- Error reporting and diagnostics

#### `ops_translate.mlir`
**Purpose**: Tests translation of XLS operations to target formats  
**What it tests**:
- Operation-level translation
- Target-specific code generation
- Operation optimization during translation

#### `canonicalize.mlir`
**Purpose**: Tests canonicalization patterns for XLS operations  
**What it tests**:
- Operation simplification patterns
- Redundant operation elimination
- Constant folding and propagation

#### `symbol_dce.mlir`
**Purpose**: Tests dead code elimination at the symbol level  
**What it tests**:
- Unused symbol elimination
- Symbol reachability analysis
- Module-level optimization

### Integration and End-to-End Tests

#### `call_dslx_effects.mlir`
**Purpose**: Tests side effects of DSLX function calls  
**What it tests**:
- Function call effect analysis
- Memory effect tracking
- Optimization boundary preservation

#### `debug_locs.mlir`
**Purpose**: Tests debug location preservation through transformations  
**What it tests**:
- Source location tracking
- Debug information preservation
- Error reporting with locations

#### `ch_attrs.mlir`
**Purpose**: Tests channel attribute handling  
**What it tests**:
- Channel configuration attributes
- FIFO parameter validation
- Channel property inheritance

#### `convert_for_op_to_sproc_call.mlir`
**Purpose**: Tests conversion of for-loops to streaming process calls  
**What it tests**:
- Loop → streaming process transformation
- Process call generation
- Data flow preservation

#### `extract_as_top_level_module.mlir`
**Purpose**: Tests extraction of functions as top-level modules  
**What it tests**:
- Function extraction and isolation
- Module boundary creation
- Interface synthesis

#### `xls_stitch.mlir`
**Purpose**: Tests XLS module stitching and composition  
**What it tests**:
- Module composition patterns
- Interface matching and connection
- Hierarchical design assembly

### Special Files and Subdirectories

#### `integration/`
Contains end-to-end integration tests:
- `addf.mlir`: Floating-point addition pipeline test
- `array_to_bits.mlir`: Array-to-bits transformation test
- `procify.mlir`: Process conversion integration test
- `select.mlir`: Selection operation integration test
- `soft_blocks_lut.mlir`: Soft logic block synthesis test

#### `i16/` and `i32/`
- `dot_product.x`: DSLX source files for testing different bit widths

#### Configuration Files
- `lit.cfg.py`: LIT test runner configuration
- `lit.site.cfg.py.in`: Site-specific LIT configuration template
- `BUILD.bazel`: Bazel build configuration for tests
- `struct_type.x`: DSLX struct type definitions

## Running Tests

Tests use the LIT (LLVM Integrated Tester) framework:

```bash
# Run all MLIR tests
bazel test //xls/contrib/mlir/testdata:all

# Run specific test
bazel test //xls/contrib/mlir/testdata:arith_to_xls.mlir.test

# Run with verbose output
bazel test //xls/contrib/mlir/testdata:all --test_output=all
```

## Test Patterns

Most tests follow this structure:
1. **RUN line**: Specifies the tool and passes to run
2. **CHECK patterns**: Expected output validation using FileCheck
3. **Input MLIR**: The MLIR code being tested
4. **Comments**: Explanation of what's being tested

### Common Tools Used
- `xls_opt`: MLIR optimization and transformation tool
- `xls_translate`: MLIR to XLS/Verilog translation tool
- `FileCheck`: Output validation tool

### Common Pass Names
- `-arith-to-xls`: Convert arith dialect to XLS
- `-scalarize`: Convert tensors to arrays
- `-array-to-bits`: Convert arrays to bit representations
- `-xls-lower`: Complete lowering pipeline
- `-canonicalize`: Apply canonicalization patterns

This comprehensive test suite ensures the reliability and correctness of the MLIR-to-XLS compilation pipeline across all major features and transformation passes.

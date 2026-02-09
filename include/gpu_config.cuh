#pragma once

// GPU MVR Compile-Time Configuration
// These values must match the index file being loaded.
// Recompile if these values need to change.

// Primary configuration parameters
#define PADDED_DIM 128   // Dimension after rotation/padding (must be multiple of 64)
#define Q_DOCLEN 32      // Query document length (number of query tokens)

// Derived constants (computed at compile-time)
#define CODE_BYTES (PADDED_DIM / 8)             // Binary code size: 16 bytes
#define NUM_U64 (PADDED_DIM / 64)               // Number of 64-bit blocks: 2
#define SMEM_QUERY_SIZE (PADDED_DIM * sizeof(float))  // Shared memory per query: 512 bytes

// Validation
static_assert(PADDED_DIM % 64 == 0, "PADDED_DIM must be a multiple of 64");
static_assert(PADDED_DIM > 0, "PADDED_DIM must be positive");
static_assert(Q_DOCLEN > 0, "Q_DOCLEN must be positive");

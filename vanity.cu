#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <curand_kernel.h>
#include <random>

#include "ed25519.cuh"
#include "fe.cu"
#include "ge.cu"
#include "sc.cu"
#include "sha512.cu"
#include "seed.cu"
#include "keypair.cu"
#include "util.cu"

// --------------------------------------------------------------------------
// HELPER: Initialize RNG state
// --------------------------------------------------------------------------
__global__ void init_rng(curandState *states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// --------------------------------------------------------------------------
// SUPER FUSED KERNEL: Inner Loop Optimization
// --------------------------------------------------------------------------
__global__ void vanity_search_fused_loop(
    curandState *states,
    const unsigned char *prefix,
    int prefix_nibbles,
    int *found,
    unsigned char *result_public,
    unsigned char *result_private,
    int batch_size,
    int loops_per_thread // NEW: How many keys to check per thread launch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Load state once
    curandState localState = states[idx];

    // Pre-calculate prefix constants to save registers/time in loop
    int full_bytes = prefix_nibbles / 2;
    int has_half_byte = prefix_nibbles % 2;
    unsigned char last_byte_mask = 0xF0; // Mask for the high nibble
    unsigned char last_byte_target = (prefix[full_bytes] & 0xF0);

    // --- THE INNER LOOP ---
    // Keep this thread alive and working for multiple keys
    for (int loop = 0; loop < loops_per_thread; loop++) {

        // Check global flag occasionally (not every loop to save memory bandwidth)
        // Checks every 16th iteration
        if ((loop & 15) == 0 && *found) break;

        // 1. Generate Seed
        unsigned char seed[32];
        unsigned int *seed_ptr = (unsigned int*)seed;

        // Unrolling generator
        seed_ptr[0] = curand(&localState); seed_ptr[1] = curand(&localState);
        seed_ptr[2] = curand(&localState); seed_ptr[3] = curand(&localState);
        seed_ptr[4] = curand(&localState); seed_ptr[5] = curand(&localState);
        seed_ptr[6] = curand(&localState); seed_ptr[7] = curand(&localState);

        // 2. Compute Keypair
        unsigned char public_key[32];
        unsigned char private_key[64];

        // Device function call
        ed25519_kernel_create_keypair(public_key, private_key, seed);

        // 3. Check Prefix (Optimized)
        bool matches = true;

        // Compare full bytes
        for (int i = 0; i < full_bytes; i++) {
            if (public_key[i] != prefix[i]) {
                matches = false;
                break;
            }
        }

        // Compare half byte (if applicable)
        if (matches && has_half_byte) {
            if ((public_key[full_bytes] & last_byte_mask) != last_byte_target) {
                matches = false;
            }
        }

        // 4. Success Handling
        if (matches) {
            if (atomicCAS(found, 0, 1) == 0) {
                // We found it! Write result.
                for (int i = 0; i < 32; i++) result_public[i] = public_key[i];
                for (int i = 0; i < 64; i++) result_private[i] = private_key[i];
            }
            break; // Stop this thread
        }
    }

    // Save state back only once at the very end
    states[idx] = localState;
}

// --------------------------------------------------------------------------
// HOST HELPERS (Same as before)
// --------------------------------------------------------------------------

int parse_hex_prefix(const char* hex_str, unsigned char* bytes, int* is_odd) {
    int len = (int)strlen(hex_str);
    *is_odd = (len % 2 != 0);
    int byte_len = (len + 1) / 2;
    for (int i = 0; i < byte_len; i++) {
        char byte_str[3] = {0};
        if (i == 0 && *is_odd) {
            byte_str[0] = '0'; byte_str[1] = hex_str[0];
        } else {
            int offset = *is_odd ? 1 : 0;
            byte_str[0] = hex_str[i*2 - offset]; byte_str[1] = hex_str[i*2 + 1 - offset];
        }
        bytes[i] = (unsigned char)strtol(byte_str, NULL, 16);
    }
    return len;
}

void print_hex(const char* label, unsigned char* data, int len) {
    printf("%s\n", label);
    for (int i = 0; i < len; i++) printf("%02X", data[i]);
    printf("\n");
}

// --------------------------------------------------------------------------
// MAIN
// --------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 2) { printf("Usage: %s <hex_prefix>\n", argv[0]); return 1; }

    const char* prefix_str = argv[1];
    unsigned char prefix_bytes[32];
    int is_odd;
    int prefix_nibbles = parse_hex_prefix(prefix_str, prefix_bytes, &is_odd);
    int prefix_byte_len = (prefix_nibbles + 1) / 2;

    printf("Searching for prefix: %s\n", prefix_str);

    // CONFIGURATION
    const int BLOCKS = 4096;
    const int THREADS = 256;
    const int BATCH_SIZE = BLOCKS * THREADS;

    // Process 50 keys per thread-launch.
    // Total batch = 1 million threads * 50 = 50 million keys per kernel launch.
    const int LOOPS_PER_THREAD = 50;

    printf("Threads: %d | Loops/Thread: %d | Batch: %d keys\n",
           BATCH_SIZE, LOOPS_PER_THREAD, BATCH_SIZE * LOOPS_PER_THREAD);

    unsigned char *d_result_public, *d_result_private, *d_prefix;
    int *d_found;
    curandState *d_rng_states;

    cudaMalloc(&d_result_public, 32);
    cudaMalloc(&d_result_private, 64);
    cudaMalloc(&d_prefix, prefix_byte_len);
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_rng_states, BATCH_SIZE * sizeof(curandState));

    unsigned char h_result_public[32], h_result_private[64];
    int h_found = 0;
    unsigned long long h_attempts = 0;

    cudaMemcpy(d_prefix, prefix_bytes, prefix_byte_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_found, &h_found, sizeof(int), cudaMemcpyHostToDevice);

    printf("Initializing RNG...\n");
    std::random_device rd;
    unsigned long long secure_seed = ((unsigned long long)rd() << 32) | rd();
    printf("Secure Seed: %llu\n", secure_seed);
    init_rng<<<BLOCKS, THREADS>>>(d_rng_states, secure_seed, BATCH_SIZE);
    cudaDeviceSynchronize();

    printf("Starting Ultra-Fast Search...\n\n");
    clock_t start = clock();

    while (!h_found) {
        vanity_search_fused_loop<<<BLOCKS, THREADS>>>(
            d_rng_states,
            d_prefix,
            prefix_nibbles,
            d_found,
            d_result_public,
            d_result_private,
            BATCH_SIZE,
            LOOPS_PER_THREAD
        );

        // This kernel takes longer to run now (because of the loop),
        // so we check status after every kernel launch.
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);

        h_attempts += (unsigned long long)BATCH_SIZE * LOOPS_PER_THREAD;

        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
        printf("\rRate: %.2f M keys/sec | Total: %llu",
               (h_attempts/elapsed)/1000000.0, h_attempts);
        fflush(stdout);
    }

    cudaMemcpy(h_result_public, d_result_public, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_private, d_result_private, 64, cudaMemcpyDeviceToHost);

    printf("\n\nMATCH FOUND!\n");
    print_hex("Public:", h_result_public, 32);
    print_hex("Private:", h_result_private, 64);

    cudaFree(d_result_public); cudaFree(d_result_private);
    cudaFree(d_prefix); cudaFree(d_found); cudaFree(d_rng_states);

    return 0;
}

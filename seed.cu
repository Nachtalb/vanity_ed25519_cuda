#include "ed25519.cuh"
#include <stdio.h>
#include <assert.h>

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>

__host__ void ed25519_kernel_create_seed(unsigned char *seed, int batch_size) {
    HCRYPTPROV hCryptProv;
    
    if (!CryptAcquireContext(&hCryptProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
        assert(0);
    }
    
    if (!CryptGenRandom(hCryptProv, 32 * batch_size, seed)) {
        CryptReleaseContext(hCryptProv, 0);
        assert(0);
    }
    
    CryptReleaseContext(hCryptProv, 0);
}

#else
// Original Linux version
__host__ void ed25519_kernel_create_seed(unsigned char *seed, int batch_size) {
    FILE *f = fopen("/dev/urandom", "rb");
    if (f == NULL) {
        assert(0);
    }
    size_t n_read = fread(seed, 1, 32 * batch_size, f);
    assert(n_read == 32 * batch_size);
    fclose(f);
}
#endif

#include <iostream>
#include <intrin.h>

static void cpuid(int regs[4], int eax, int ecx) {
    __cpuidex(regs, eax, ecx);
}

int main() {
    int regs[4];
    cpuid(regs, 0, 0);
    int maxLeaf = regs[0];

    cpuid(regs, 1, 0);
    bool sse = (regs[3] & (1 << 25)) != 0;
    bool sse2 = (regs[3] & (1 << 26)) != 0;
    bool avx = (regs[2] & (1 << 28)) != 0;
    bool osxsave = (regs[2] & (1 << 27)) != 0;

    unsigned long long xcr0 = 0;
    if (osxsave) {
        xcr0 = _xgetbv(0);
    }
    bool avx_os = osxsave && ((xcr0 & 0x6ULL) == 0x6ULL);
    bool avx_final = avx && avx_os;

    bool avx2 = false;
    bool avx512 = false;
    if (maxLeaf >= 7) {
        cpuid(regs, 7, 0);
        avx2 = (regs[1] & (1 << 5)) != 0;
        bool avx512f = (regs[1] & (1 << 16)) != 0;
        bool avx512_os = osxsave && ((xcr0 & 0xE6ULL) == 0xE6ULL);
        avx512 = avx512f && avx512_os;
    }

    std::cout << "CPU features:\n";
    std::cout << "  SSE: " << (sse ? "yes" : "no") << "\n";
    std::cout << "  SSE2: " << (sse2 ? "yes" : "no") << "\n";
    std::cout << "  AVX (ISA+OS): " << (avx_final ? "yes" : "no") << "\n";
    std::cout << "  AVX2: " << (avx2 ? "yes" : "no") << "\n";
    std::cout << "  AVX512: " << (avx512 ? "yes" : "no") << "\n";
    std::cout << "XCR0 = 0x" << std::hex << xcr0 << std::dec << "\n";

    return 0;
}

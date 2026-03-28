#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <intrin.h>
#include <windows.h>

using namespace std;

struct CpuidRegs {
    int eax, ebx, ecx, edx;
};

CpuidRegs cpuid(int functionId, int subfunctionId = 0) {
    int cpuInfo[4] = { 0, 0, 0, 0 };
    __cpuidex(cpuInfo, functionId, subfunctionId);
    return { cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3] };
}

bool testBit(int value, int bit) {
    return (value & (1 << bit)) != 0;
}

void printBoolFeature(const string& name, bool value) {
    cout << left << setw(20) << name << ": " << (value ? "Yes" : "No") << '\n';
}

string getVendor() {
    CpuidRegs r = cpuid(0);
    char vendor[13];
    memcpy(vendor + 0, &r.ebx, 4);
    memcpy(vendor + 4, &r.edx, 4);
    memcpy(vendor + 8, &r.ecx, 4);
    vendor[12] = '\0';
    return string(vendor);
}

string getBrand() {
    CpuidRegs maxExt = cpuid(0x80000000);
    if ((unsigned)maxExt.eax < 0x80000004)
        return "Brand string not supported";

    char brand[49];
    memset(brand, 0, sizeof(brand));

    CpuidRegs r1 = cpuid(0x80000002);
    CpuidRegs r2 = cpuid(0x80000003);
    CpuidRegs r3 = cpuid(0x80000004);

    memcpy(brand + 0, &r1, 16);
    memcpy(brand + 16, &r2, 16);
    memcpy(brand + 32, &r3, 16);

    return string(brand);
}

void printBasicInfo() {
    CpuidRegs r0 = cpuid(0);
    CpuidRegs r1 = cpuid(1);

    unsigned stepping = r1.eax & 0xF;
    unsigned model = (r1.eax >> 4) & 0xF;
    unsigned family = (r1.eax >> 8) & 0xF;
    unsigned procType = (r1.eax >> 12) & 0x3;
    unsigned extModel = (r1.eax >> 16) & 0xF;
    unsigned extFamily = (r1.eax >> 20) & 0xFF;

    unsigned displayFamily = family;
    unsigned displayModel = model;

    if (family == 0xF) displayFamily += extFamily;
    if (family == 0x6 || family == 0xF) displayModel += (extModel << 4);

    unsigned logicalProcessors = (r1.ebx >> 16) & 0xFF;
    unsigned apicId = (r1.ebx >> 24) & 0xFF;

    cout << "================ BASIC INFO ================\n";
    cout << "Vendor                     : " << getVendor() << '\n';
    cout << "Brand                      : " << getBrand() << '\n';
    cout << "Max basic CPUID function   : 0x" << hex << uppercase << r0.eax << dec << '\n';
    cout << "Raw CPUID(1).EAX           : 0x" << hex << uppercase << r1.eax << dec << '\n';
    cout << "Stepping ID                : " << stepping << '\n';
    cout << "Model                      : " << displayModel << '\n';
    cout << "Family                     : " << displayFamily << '\n';
    cout << "Processor type             : " << procType << '\n';
    cout << "Max logical processors     : " << logicalProcessors << '\n';
    cout << "Local APIC ID              : " << apicId << '\n';
    cout << '\n';
}

void printFeaturesLeaf1() {
    CpuidRegs r1 = cpuid(1);

    cout << "=========== FEATURES: CPUID(EAX=1) ==========\n";
    printBoolFeature("FPU", testBit(r1.edx, 0));
    printBoolFeature("TSC", testBit(r1.edx, 4));
    printBoolFeature("MMX", testBit(r1.edx, 23));
    printBoolFeature("SSE", testBit(r1.edx, 25));
    printBoolFeature("SSE2", testBit(r1.edx, 26));
    printBoolFeature("HTT", testBit(r1.edx, 28));

    printBoolFeature("SSE3", testBit(r1.ecx, 0));
    printBoolFeature("SSSE3", testBit(r1.ecx, 9));
    printBoolFeature("FMA3", testBit(r1.ecx, 12));
    printBoolFeature("SSE4.1", testBit(r1.ecx, 19));
    printBoolFeature("SSE4.2", testBit(r1.ecx, 20));
    printBoolFeature("AVX", testBit(r1.ecx, 28));
    cout << '\n';
}

void printFeaturesLeaf7() {
    CpuidRegs r0 = cpuid(0);
    if ((unsigned)r0.eax < 7) {
        cout << "CPUID(EAX=7) not supported\n\n";
        return;
    }

    CpuidRegs r70 = cpuid(7, 0);
    cout << "========= FEATURES: CPUID(EAX=7, ECX=0) =========\n";
    cout << "Max subleaf                 : " << r70.eax << '\n';
    printBoolFeature("AVX2", testBit(r70.ebx, 5));
    printBoolFeature("SHA", testBit(r70.ebx, 29));
    printBoolFeature("AVX512F", testBit(r70.ebx, 16));
    printBoolFeature("GFNI", testBit(r70.ecx, 8));

    bool amxTile = testBit(r70.edx, 24);
    bool amxInt8 = testBit(r70.edx, 25);
    bool amxBf16 = testBit(r70.edx, 22);
    printBoolFeature("AMX-TILE", amxTile);
    printBoolFeature("AMX-INT8", amxInt8);
    printBoolFeature("AMX-BF16", amxBf16);
    cout << '\n';

    if (r70.eax >= 1) {
        CpuidRegs r71 = cpuid(7, 1);
        cout << "========= FEATURES: CPUID(EAX=7, ECX=1) =========\n";
        printBoolFeature("AVX10", testBit(r71.edx, 19));
        printBoolFeature("AMX-COMPLEX", testBit(r71.edx, 8));
        cout << '\n';
    }
}

string cacheTypeToString(unsigned t) {
    switch (t) {
    case 0: return "Null";
    case 1: return "Data Cache";
    case 2: return "Instruction Cache";
    case 3: return "Unified Cache";
    default: return "Reserved";
    }
}

void printIntelCaches() {
    cout << "=============== CACHE INFO =================\n";
    for (int i = 0; ; ++i) {
        CpuidRegs r = cpuid(4, i);

        unsigned cacheType = r.eax & 0x1F;
        if (cacheType == 0)
            break;

        unsigned cacheLevel = (r.eax >> 5) & 0x7;
        bool fullyAssoc = ((r.eax >> 9) & 0x1) != 0;
        unsigned threadsPerCache = ((r.eax >> 14) & 0xFFF) + 1;
        unsigned coresInPackage = ((r.eax >> 26) & 0x3F) + 1;

        unsigned lineSize = (r.ebx & 0xFFF) + 1;
        unsigned partitions = ((r.ebx >> 12) & 0x3FF) + 1;
        unsigned ways = ((r.ebx >> 22) & 0x3FF) + 1;
        unsigned sets = r.ecx + 1;

        bool inclusive = ((r.edx >> 1) & 0x1) != 0;

        unsigned long long cacheSize =
            1ULL * lineSize * partitions * ways * sets;

        cout << "Cache #" << i << '\n';
        cout << "Type                        : " << cacheTypeToString(cacheType) << '\n';
        cout << "Level                       : L" << cacheLevel << '\n';
        cout << "Fully associative           : " << (fullyAssoc ? "Yes" : "No") << '\n';
        cout << "Threads sharing cache       : " << threadsPerCache << '\n';
        cout << "Processor cores             : " << coresInPackage << '\n';
        cout << "Cache line size             : " << lineSize << " bytes\n";
        cout << "Physical line partitions    : " << partitions << '\n';
        cout << "Ways of associativity       : " << ways << '\n';
        cout << "Number of sets              : " << sets << '\n';
        cout << "Organization                : " << (inclusive ? "Inclusive" : "Exclusive / Non-inclusive") << '\n';
        cout << "Cache size                  : " << cacheSize / 1024 << " KB\n";
        cout << "---------------------------------------------\n";
    }
    cout << '\n';
}

void printAmdCaches() {
    cout << "=============== CACHE INFO =================\n";
    cout << "AMD may use CPUID(EAX=8000001Dh) for deterministic cache parameters.\n";

    for (int i = 0; ; ++i) {
        CpuidRegs r = cpuid(0x8000001D, i);

        unsigned cacheType = r.eax & 0x1F;
        if (cacheType == 0)
            break;

        unsigned cacheLevel = (r.eax >> 5) & 0x7;
        bool fullyAssoc = ((r.eax >> 9) & 0x1) != 0;
        unsigned threadsPerCache = ((r.eax >> 14) & 0xFFF) + 1;
        unsigned coresInPackage = ((r.eax >> 26) & 0x3F) + 1;

        unsigned lineSize = (r.ebx & 0xFFF) + 1;
        unsigned partitions = ((r.ebx >> 12) & 0x3FF) + 1;
        unsigned ways = ((r.ebx >> 22) & 0x3FF) + 1;
        unsigned sets = r.ecx + 1;

        bool inclusive = ((r.edx >> 1) & 0x1) != 0;

        unsigned long long cacheSize =
            1ULL * lineSize * partitions * ways * sets;

        cout << "Cache #" << i << '\n';
        cout << "Type                        : " << cacheTypeToString(cacheType) << '\n';
        cout << "Level                       : L" << cacheLevel << '\n';
        cout << "Fully associative           : " << (fullyAssoc ? "Yes" : "No") << '\n';
        cout << "Threads sharing cache       : " << threadsPerCache << '\n';
        cout << "Processor cores             : " << coresInPackage << '\n';
        cout << "Cache line size             : " << lineSize << " bytes\n";
        cout << "Physical line partitions    : " << partitions << '\n';
        cout << "Ways of associativity       : " << ways << '\n';
        cout << "Number of sets              : " << sets << '\n';
        cout << "Organization                : " << (inclusive ? "Inclusive" : "Exclusive / Non-inclusive") << '\n';
        cout << "Cache size                  : " << cacheSize / 1024 << " KB\n";
        cout << "---------------------------------------------\n";
    }
    cout << '\n';
}

void printFrequencyInfo() {
    CpuidRegs r0 = cpuid(0);
    if ((unsigned)r0.eax < 0x16) {
        cout << "========= FREQUENCY INFO: CPUID(EAX=16h) =========\n";
        cout << "Not supported by this CPU\n\n";
        return;
    }

    CpuidRegs r = cpuid(0x16);
    unsigned baseMHz = r.eax & 0xFFFF;
    unsigned maxMHz = r.ebx & 0xFFFF;
    unsigned busMHz = r.ecx & 0xFFFF;

    cout << "========= FREQUENCY INFO: CPUID(EAX=16h) =========\n";
    cout << "Base frequency             : " << baseMHz << " MHz\n";
    cout << "Max frequency (boost)      : " << maxMHz << " MHz\n";
    cout << "Bus frequency              : " << busMHz << " MHz\n\n";
}

void printExtendedFeatures() {
    CpuidRegs maxExt = cpuid(0x80000000);
    cout << "========= EXTENDED CPUID INFO =========\n";
    cout << "Max extended CPUID function : 0x" << hex << uppercase << maxExt.eax << dec << '\n';

    if ((unsigned)maxExt.eax >= 0x80000001) {
        CpuidRegs r = cpuid(0x80000001);

        printBoolFeature("SSE4a", testBit(r.ecx, 6));
        printBoolFeature("FMA4", testBit(r.ecx, 16));
        printBoolFeature("3DNow!", testBit(r.edx, 31));
        printBoolFeature("Ext 3DNow!", testBit(r.edx, 30));
    }
    else {
        cout << "CPUID(80000001h) not supported\n";
    }
    cout << '\n';
}

int main() {
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    string vendor = getVendor();

    cout << "=========================================\n";
    cout << "         LAB WORK #3 - CPUID INFO        \n";
    cout << "=========================================\n\n";

    printBasicInfo();
    printFeaturesLeaf1();
    printFeaturesLeaf7();

    if (vendor == "GenuineIntel")
        printIntelCaches();
    else if (vendor == "AuthenticAMD")
        printAmdCaches();
    else
        cout << "Unknown vendor: cache parsing skipped or may be incomplete.\n\n";

    printFrequencyInfo();
    printExtendedFeatures();

    system("pause");
    return 0;
}

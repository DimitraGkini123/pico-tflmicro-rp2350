#include "main_functions.h"   // δηλώνει setup() / loop()
#include "svm_model.h"        // weights + δηλώσεις SVM συναρτήσεων
#include "pico/stdlib.h"

#include <stdio.h>
#include <stdint.h>

// =============================================================
//                     DWT COUNTERS (όπως πριν)
// =============================================================
#define DEMCR                (*(volatile uint32_t *)0xE000EDFC)
#define DEMCR_TRCENA         (1u << 24)

#define DWT_BASE             (0xE0001000u)
#define DWT_CTRL             (*(volatile uint32_t *)(DWT_BASE + 0x000))
#define DWT_CYCCNT           (*(volatile uint32_t *)(DWT_BASE + 0x004))
#define DWT_CPICNT           (*(volatile uint32_t *)(DWT_BASE + 0x008))
#define DWT_EXCCNT           (*(volatile uint32_t *)(DWT_BASE + 0x00C))
#define DWT_SLEEPCNT         (*(volatile uint32_t *)(DWT_BASE + 0x010))
#define DWT_LSUCNT           (*(volatile uint32_t *)(DWT_BASE + 0x014))
#define DWT_FOLDCNT          (*(volatile uint32_t *)(DWT_BASE + 0x018))
#define DWT_LAR              (*(volatile uint32_t *)(DWT_BASE + 0xFB0))

#define DWT_CTRL_CYCCNTENA   (1u << 0)
#define DWT_CTRL_CPIEVTENA   (1u << 17)
#define DWT_CTRL_EXCEVTENA   (1u << 18)
#define DWT_CTRL_SLEEPEVTENA (1u << 19)
#define DWT_CTRL_LSUEVTENA   (1u << 20)
#define DWT_CTRL_FOLDEVTENA  (1u << 21)

static inline void dwt_enable_all() {
    DEMCR |= DEMCR_TRCENA;
    DWT_LAR = 0xC5ACCE55;

    DWT_CYCCNT   = 0;
    DWT_CPICNT   = 0;
    DWT_EXCCNT   = 0;
    DWT_SLEEPCNT = 0;
    DWT_LSUCNT   = 0;
    DWT_FOLDCNT  = 0;

    DWT_CTRL |= DWT_CTRL_CYCCNTENA |
                DWT_CTRL_CPIEVTENA |
                DWT_CTRL_EXCEVTENA |
                DWT_CTRL_SLEEPEVTENA |
                DWT_CTRL_LSUEVTENA |
                DWT_CTRL_FOLDEVTENA;
}

// =============================================================
//   ΠΟΛΛΑ FAKE FEATURE VECTORS (για πολλά test cases)
// =============================================================
static const float TEST_CASES[][SVM_INPUTS] = {

    // 0 — ADL (πολύ ήρεμο)
    //σ(l) μ(l)  σ(r)  μ(r)
    {0.05, 0.01, 0.02, 0.005,   //x axis
     0.04, 0.01, 0.03, 0.008,   //y axis
     0.03, 0.01, 0.02, 0.005},  //z axis

     //σ < 0.05 --> άρα μιρκή αστάθεια
     //means (μ) πολύ κοντά στο 0 
     //left and right segments πολύ παρόμοια

    // 1 — borderline (SVML=1 αλλά SVMQ ίσως=0)
    {0.30, 0.10, 0.25, 0.08,
     0.28, 0.09, 0.22, 0.07,
     0.35, 0.12, 0.20, 0.08},

     //σ μεταξύ 0.2 και 0.35 --> έντονη κίνηση αλλά όχι fall
     //Means επίσης ανεβασμένα → πιθανό σήκωμα/κάθισμα

    // 2 — strong fall (SVML=1, SVMQ=1)
    {0.80, 0.20, 0.75, 0.18,
     0.82, 0.21, 0.78, 0.19,
     1.10, 0.30, 0.85, 0.22},

     //σ πολύ μεγάλο --> έντονη αστάθεια
     // left mean αυξημένο ( pre fall)
     // right mean ήρεμο αλλά ανεβασμένο ( post fall )

    // 3 — noise/adversarial
    {0.50, 0.01, 0.40, 0.02,
     0.60, 0.015, 0.35, 0.01,
     0.90, 0.05, 0.80, 0.02}
};
static const int NUM_TEST_CASES = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);
static int current_case = 0;

// =============================================================
//                          SETUP
// =============================================================
void setup() {
    stdio_init_all();   // USB serial
}

// =============================================================
//                           LOOP
// =============================================================
void loop() {
    // ================================
    // 1. Fake 12-D SVM FEATURES
    // ================================
    float f[SVM_INPUTS];
        for (int i = 0; i < SVM_INPUTS; i++) {
        f[i] = TEST_CASES[current_case][i];
    }
    normalize_features(f);
// Next testcase each loop
    current_case++;
    if (current_case >= NUM_TEST_CASES) current_case = 0;

    // ======================================================
    // 2. SVML stage
    // ======================================================
    dwt_enable_all();
    uint64_t t0_svm1 = time_us_64();

    int b1 = svm1_predict(f);

    uint64_t t1_svm1 = time_us_64();

    uint32_t cycles_svm1 = DWT_CYCCNT;
    uint32_t cpi_svm1    = DWT_CPICNT;
    uint32_t lsu_svm1    = DWT_LSUCNT;
    uint32_t fold_svm1   = DWT_FOLDCNT;
    uint32_t exc_svm1    = DWT_EXCCNT;
    uint32_t sleep_svm1  = DWT_SLEEPCNT;
    uint64_t time_svm1   = t1_svm1 - t0_svm1;

    // ======================================================
    // 3. SVMQ stage (only if SVML says suspicious)
    // ======================================================
    int b2 = 0;

    uint32_t cycles_svm2 = 0;
    uint32_t cpi_svm2    = 0;
    uint32_t lsu_svm2    = 0;
    uint32_t fold_svm2   = 0;
    uint32_t exc_svm2    = 0;
    uint32_t sleep_svm2  = 0;
    uint64_t time_svm2   = 0;

    if (b1 == 1) {
        float f2[SVM2_INPUTS];
        make_poly_features(f, f2);

        dwt_enable_all();
        uint64_t t0_svm2 = time_us_64();

        b2 = svm2_predict(f2);

        uint64_t t1_svm2 = time_us_64();

        cycles_svm2 = DWT_CYCCNT;
        cpi_svm2    = DWT_CPICNT;
        lsu_svm2    = DWT_LSUCNT;
        fold_svm2   = DWT_FOLDCNT;
        exc_svm2    = DWT_EXCCNT;
        sleep_svm2  = DWT_SLEEPCNT;
        time_svm2   = t1_svm2 - t0_svm2;
    }

    // ======================================================
    // 4. PRINT RESULTS
    // ======================================================
    printf("\n=========== TEST CASE %d ===========\n", current_case);
    printf("SVML (stage1): b1=%d\n", b1);
    printf("  cycles=%u  time=%llu us\n", cycles_svm1, time_svm1);
    printf("  CPI=%u  LSU=%u  FOLD=%u  EXC=%u\n",
           cpi_svm1, lsu_svm1, fold_svm1, exc_svm1);

    if (b1 == 1) {
        printf("\nSVMQ (stage2): b2=%d\n", b2);
        printf("  cycles=%u  time=%llu us\n", cycles_svm2, time_svm2);
        printf("  CPI=%u  LSU=%u  FOLD=%u  EXC=%u\n",
               cpi_svm2, lsu_svm2, fold_svm2, exc_svm2);
    } else {
        printf("\nSVMQ (stage2): SKIPPED\n");
    }

    printf("====================================\n");

    sleep_ms(500);
}
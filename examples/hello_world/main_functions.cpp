#include "constants.h"
#include "hello_world_int8_model_data.h"
#include "mlp_int8_model_data.h"
#include "main_functions.h"
#include "output_handler.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "pico/stdlib.h"

// =============================================================
//                        DWT COUNTERS
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
//                        TFLM GLOBALS
// =============================================================
namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    int inference_count = 0;

    constexpr int kTensorArenaSize = 100 * 1024;  // 100 KB
    uint8_t tensor_arena[kTensorArenaSize];
}

// =============================================================
//                          SETUP()
// =============================================================
void setup() {
    stdio_init_all();
    tflite::InitializeTarget();

    model = tflite::GetModel(g_mlp_int8_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Bad model version");
        return;
    }

    static tflite::MicroMutableOpResolver<3> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    inference_count = 0;
}

// =============================================================
//                           LOOP()
// =============================================================
enum WorkloadType { LIGHT, MEDIUM, HEAVY };

void fill_features(float *dst, WorkloadType type) {
    switch(type) {

        case LIGHT:
            for (int i = 0; i < 59; i++)
                dst[i] = 0.01f * ((i % 3) - 1);   // very small values → lightweight
            break;

        case MEDIUM:
            for (int i = 0; i < 59; i++)
                dst[i] = 0.5f * sinf(i * 0.3f);   // sinusoidal balanced load
            break;

        case HEAVY:
            for (int i = 0; i < 59; i++)
                dst[i] = 3.0f * ((i % 2 == 0) ? 1.0f : -1.0f); // max stress
            break;
    }
}


void loop() {

static const int REPEAT = 20;
static uint32_t cycles_buf[REPEAT];
static uint32_t lsu_buf[REPEAT];
static uint32_t cpi_buf[REPEAT];
static uint32_t fold_buf[REPEAT];
static uint64_t time_buf[REPEAT];

  for (int r = 0; r < REPEAT; r++) {

float features[59];
fill_features(features, MEDIUM);  // ή MEDIUM ή LIGHT


    // 2. Quantize → INT8
    for (int i = 0; i < 59; i++) {
        float x = features[i];
        int32_t q = (int32_t)(x / input->params.scale + input->params.zero_point);
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        input->data.int8[i] = (int8_t)q;
    }

    // 3. Measure
    dwt_enable_all();
    uint64_t t0 = time_us_64();
    TfLiteStatus status = interpreter->Invoke();
    uint64_t t1 = time_us_64();

    if (status != kTfLiteOk) {
        printf("Invoke failed!\n");
        return;
    }

    // 4. Read output logits
    int8_t y0_q = output->data.int8[0];
    int8_t y1_q = output->data.int8[1];

    float y0 = (y0_q - output->params.zero_point) * output->params.scale;
    float y1 = (y1_q - output->params.zero_point) * output->params.scale;

    int predicted_class = (y0 > y1 ? 0 : 1);

    // 5. Read performance counters
    uint32_t cycles = DWT_CYCCNT;
    uint32_t cpi = DWT_CPICNT;
    uint32_t lsu = DWT_LSUCNT;
    uint32_t fold = DWT_FOLDCNT;

    // 6. Print
    printf("Pred=%d  y0=%.3f  y1=%.3f  cycles=%u  lsu=%u  cpi=%u  fold=%u  time_us=%llu\n",
           predicted_class, y0, y1, cycles, lsu, cpi, fold, (t1 - t0));

    uint64_t dt = t1 - t0;
    cycles_buf[r] = cycles;
    lsu_buf[r]    = lsu;
    cpi_buf[r]    = cpi;
    fold_buf[r]   = fold;
    time_buf[r]   = dt;
    fflush(stdout); 
    sleep_ms(300);
}

    // ==========================
    // Compute MEAN and STD
    // ==========================

    auto compute_mean = [&](uint32_t *buf) {
        double sum = 0;
        for (int i = 0; i < REPEAT; i++) sum += buf[i];
        return sum / REPEAT;
    };

    auto compute_std = [&](uint32_t *buf, double mean) {
        double sum = 0;
        for (int i = 0; i < REPEAT; i++) {
            double d = buf[i] - mean;
            sum += d * d;
        }
        return sqrt(sum / REPEAT);
    };

    auto compute_mean64 = [&](uint64_t *buf) {
        double sum = 0;
        for (int i = 0; i < REPEAT; i++) sum += buf[i];
        return sum / REPEAT;
    };

    auto compute_std64 = [&](uint64_t *buf, double mean) {
        double sum = 0;
        for (int i = 0; i < REPEAT; i++) {
            double d = buf[i] - mean;
            sum += d * d;
        }
        return sqrt(sum / REPEAT);
    };

    double mean_cycles = compute_mean(cycles_buf);
    double mean_lsu    = compute_mean(lsu_buf);
    double mean_cpi    = compute_mean(cpi_buf);
    double mean_fold   = compute_mean(fold_buf);
    double mean_time   = compute_mean64(time_buf);

    double std_cycles = compute_std(cycles_buf, mean_cycles);
    double std_lsu    = compute_std(lsu_buf, mean_lsu);
    double std_cpi    = compute_std(cpi_buf, mean_cpi);
    double std_fold   = compute_std(fold_buf, mean_fold);
    double std_time   = compute_std64(time_buf, mean_time);

    printf("\n=== FINAL STATISTICS (%d runs) ===\n", REPEAT);
    printf("Cycles: mean=%.2f  std=%.2f\n", mean_cycles, std_cycles);
    printf("LSU:    mean=%.2f  std=%.2f\n", mean_lsu, std_lsu);
    printf("CPI:    mean=%.2f  std=%.2f\n", mean_cpi, std_cpi);
    printf("Folded: mean=%.2f  std=%.2f\n", mean_fold, std_fold);
    printf("Time:   mean=%.2f us  std=%.2f us\n", mean_time, std_time);

    printf("\n=== DONE WITH %d MEASUREMENTS ===\n", REPEAT);

    while (1) sleep_ms(1000);
}
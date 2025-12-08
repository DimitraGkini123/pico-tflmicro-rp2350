#include "constants.h"
#include "seizure_mlp_int8_model_data.h" 
#include "main_functions.h"
#include "output_handler.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "pico/stdlib.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h> // Για τη συνάρτηση isnan
// =============================================================
//                     DWT COUNTERS
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
constexpr float kFeatureMeans[16] = {
    // 1-2. ACC meanX, stdX
    0.012f, 0.048f, 
    // 3-4. ACC meanY, stdY
    -0.015f, 0.042f, 
    // 5-6. ACC meanZ, stdZ
    0.035f, 0.055f, 
    // 7-8. SMA, peak ACC
    0.450f, 0.880f,
    // 9. RMS EMG
    0.115f, 
    // 10. zero crossings (ZC)
    9.5f, 
    // 11. waveform length (WA)
    16.5f, 
    // 12. slope sign changes (SSC)
    31.0f, 
    // 13. EMG mean abs
    0.032f, 
    // 14. EMG WL
    5.5f, 
    // 15. EMG variance
    0.250f, 
    // 16. EMG peak amplitude
    0.850f
    // **********************************************
};

constexpr float kFeatureStdDevs[16] = {
    // 1-2. ACC stdX
    0.001f, 0.005f, 
    // 3-4. ACC stdY
    0.002f, 0.004f, 
    // 5-6. ACC stdZ
    0.003f, 0.006f, 
    // 7-8. SMA, peak ACC
    0.050f, 0.100f,
    // 9. RMS EMG
    0.010f, 
    // 10. zero crossings (ZC)
    1.0f, 
    // 11. waveform length (WA)
    2.0f, 
    // 12. slope sign changes (SSC)
    3.0f, 
    // 13. EMG mean abs
    0.002f, 
    // 14. EMG WL
    0.5f, 
    // 15. EMG variance
    0.050f, 
    // 16. EMG peak amplitude
    0.150f
    // **********************************************
};

// =============================================================
//                        TFLM GLOBALS
// =============================================================
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 150 * 1024; // 200 KB
uint8_t tensor_arena[kTensorArenaSize];
}

// =============================================================
//                          SETUP()
// =============================================================
void setup() {
    stdio_init_all();
    tflite::InitializeTarget();

    model = tflite::GetModel(g_seizure_mlp_int8_model_data);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Bad model version!");
        return;
    }

    static tflite::MicroMutableOpResolver<2> resolver; 
    resolver.AddFullyConnected();
    resolver.AddRelu();
    // Softmax is not included to match the 1-output Sigmoid model.

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    interpreter = &static_interpreter;

printf("Before AllocateTensors\n");

if (interpreter->AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors failed\n");
    return;
}

printf("After AllocateTensors\n");
    input = interpreter->input(0);
    output = interpreter->output(0);
}

// =============================================================
//              *** 16 FEATURE SEIZURE CLASSIFIER ***
// =============================================================
void loop() {

    // Fake input (16-D). Μπορείς να βάλεις ό,τι θέλεις.
    float raw_features[16] = {
        0.01f, 0.05f,     // ACC meanX, stdX
        -0.02f, 0.04f,    // meanY, stdY
        0.03f, 0.06f,     // meanZ, stdZ
        0.40f, 0.92f,     // SMA, peak ACC
        0.12f,            // RMS EMG
        10.0f,            // zero crossings
        17.0f,            // waveform length
        32.0f,            // slope sign changes
        0.03f,            // EMG mean abs
        5.8f,             // EMG WL
        0.26f,            // EMG variance
        0.90f             // EMG peak amplitude
    };

    // ===============================
    // QUANTIZE INPUT
    // ===============================
    float standardized_features[16];
    for (int i = 0; i < 16; i++) {
        // Standardized = (Raw - Mean) / StdDev
        float raw_x = raw_features[i];
        float mean = kFeatureMeans[i];
        float stddev = kFeatureStdDevs[i];
        
        // Prevent division by zero if stddev is 0 (occurs if a feature is constant)
        if (stddev < 1e-6f) {
             standardized_features[i] = raw_x - mean; // Apply only mean subtraction
        } else {
             standardized_features[i] = (raw_x - mean) / stddev;
        }

        // Safety check for NaN/Inf which can crash the interpreter
        if (isnan(standardized_features[i]) || isinf(standardized_features[i])) {
            standardized_features[i] = 0.0f;
        }
    }

        for (int i = 0; i < 16; i++) {
        float x = standardized_features[i];
        
        // Quantization formula: q = round(x / scale) + zero_point
        int32_t q = (int32_t)(x / input->params.scale + input->params.zero_point);

        // Clamping (always good practice for Int8)
        if (q < -128) q = -128;
        if (q > 127)  q = 127;

        input->data.int8[i] = (int8_t)q;
    }


    // =======================================
    // 3. RUN INFERENCE + COUNTERS
    // =======================================
    dwt_enable_all();
    uint64_t t0 = time_us_64();
    TfLiteStatus s = interpreter->Invoke();
    uint64_t t1 = time_us_64();

    if (s != kTfLiteOk) {
        printf("Invoke failed! Check TFLiteMicro error logs.\n");
        return;
    }

    // =======================================
    // 4. DE-QUANTIZE OUTPUT
    // =======================================
    int8_t q0 = output->data.int8[0];
    int8_t q1 = output->data.int8[1];

    // De-quantization formula: y = (q - zero_point) * scale
    float y0 = (q0 - output->params.zero_point) * output->params.scale;
    float y1 = (q1 - output->params.zero_point) * output->params.scale;

    int pred = (y0 > y1 ? 0 : 1);

    printf("\n=== SEIZURE MLP ===\n");
    printf("Raw Feature 1: %.2f -> Standardized: %.2f -> Quantized: %d\n", 
           raw_features[0], standardized_features[0], input->data.int8[0]);
    printf("Normal prob = %.3f\n", y0);
    printf("Seizure prob = %.3f\n", y1);
    printf("Predicted: %s\n", pred == 0 ? "NORMAL" : "SEIZURE");

    printf("cycles=%u  cpi=%u  lsu=%u  fold=%u  time=%llu us\n",
           DWT_CYCCNT, DWT_CPICNT, DWT_LSUCNT, DWT_FOLDCNT, (t1 - t0));

    sleep_ms(300);
}

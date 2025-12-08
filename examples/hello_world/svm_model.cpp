#include "svm_model.h"

// -----------------------------------------------
// Dot product --> εσωτερικό γινόμενο
// -----------------------------------------------
float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

// -----------------------------------------------
// Normalize features (StandardScaler)
// -----------------------------------------------
void normalize_features(float* f) {
    for (int i = 0; i < SVM_INPUTS; i++) {
        // (x - mean) / std
        f[i] = (f[i] - SCALER_MEAN[i]) / SCALER_STD[i];
    }
}

// -----------------------------------------------
// SVM1 PREDICTION (Linear SVM)
// -----------------------------------------------
int svm1_predict(const float* f) {
    float score = dot(f, SVM1_W, SVM_INPUTS) + SVM1_B;
    return score > 0 ? 1 : 0;
}

// -----------------------------------------------
// Polynomial Expansion (Quadratic Features)
// πρέπει να έχει το *ίδιο ordering* με το scikit-learn
// -----------------------------------------------
void make_poly_features(const float* f, float* out) {
    int k = 0;

    // 1) First the linear terms
    for (int i = 0; i < SVM_INPUTS; i++) {
        out[k++] = f[i];
    }

    // 2) Then the quadratic terms f[i] * f[i]
    for (int i = 0; i < SVM_INPUTS; i++) {
        out[k++] = f[i] * f[i];
    }

    // 3) Cross terms f[i] * f[j] for j >= i+1
    for (int i = 0; i < SVM_INPUTS; i++) {
        for (int j = i + 1; j < SVM_INPUTS; j++) {
            out[k++] = f[i] * f[j];
        }
    }
}

// -----------------------------------------------
// SVM2 PREDICTION (Quadratic SVM implemented as Linear)
// -----------------------------------------------
int svm2_predict(const float* f2) {
    float score = dot(f2, SVM2_W, SVM2_INPUTS) + SVM2_B;
    return score > 0 ? 1 : 0;
}

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <set>
#include <map>
#include <fstream>
#include <random>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <filesystem>
using namespace std;

// MurmurHash3 (64-bit) implementation
uint64_t MurmurHash3_x64_64(const void* key, int len, uint32_t seed) {
    const uint64_t m = 0xc6a4a7935bd1e995ULL;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const uint64_t* data = (const uint64_t*)key;
    const uint64_t* end = data + (len / 8);

    while (data != end) {
        uint64_t k = *data++;
        k *= m;
        k ^= k >> r;
        k *= m;
        h ^= k;
        h *= m;
    }

    const unsigned char* data2 = (const unsigned char*)data;
    switch (len & 7) {
    case 7: h ^= uint64_t(data2[6]) << 48;
    case 6: h ^= uint64_t(data2[5]) << 40;
    case 5: h ^= uint64_t(data2[4]) << 32;
    case 4: h ^= uint64_t(data2[3]) << 24;
    case 3: h ^= uint64_t(data2[2]) << 16;
    case 2: h ^= uint64_t(data2[1]) << 8;
    case 1: h ^= uint64_t(data2[0]);
        h *= m;
    };
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
}

// Wrapper for strings
uint64_t hash_str(const string& s, uint32_t seed) {
    return MurmurHash3_x64_64(s.c_str(), s.length(), seed);
}

// Wrapper for integers (synthetic data)
uint64_t hash_int(int val, uint32_t seed) {
    return MurmurHash3_x64_64(&val, sizeof(val), seed);
}

// ALGORITHM 1: HyperLogLog (HLL) 
class HyperLogLog {
private:
    int b;          // Number of bits for the index
    int m;          // Number of counters (2^b)
    double alphaMM; // Alpha * m * m
    vector<uint8_t> counters;
    uint32_t seed;

    // Helper to count leading zeros
    uint8_t count_leading_zeros(uint64_t hash) {
        // We use the remaining 64-b bits
        uint64_t x = hash;
        if (x == 0) return 64 - b; 
        
        uint8_t count = 1;
        // Shift out the index bits
        x = x << b; 
        
        // Simple loop to check sequence of zeros
        while ((x & 0x8000000000000000ULL) == 0 && count <= (64 - b)) {
            count++;
            x <<= 1;
        }
        return count;
    }

public:
    HyperLogLog(int b_bits, uint32_t seed_val) : b(b_bits), seed(seed_val) {
        m = 1 << b;
        counters.assign(m, 0);

        double alpha;
        switch (m) { // (TODO: VULL INVESTIGAR SI AQUEST CÀLCUL D'ALFA ÉS REALMENT CORRECTE O ÉS MILLOR LA FÓRMULA GENÈRICA) 
            case 16: alpha = 0.673; break;
            case 32: alpha = 0.697; break;
            case 64: alpha = 0.709; break;
            default: alpha = 0.7213 / (1 + 1.079 / m); break;
        }
        alphaMM = alpha * m * m;
    }

    void add(const string& item) {
        uint64_t h = hash_str(item, seed);
        uint32_t idx = h >> (64 - b); // Use top b bits for index
        uint8_t rank = count_leading_zeros(h); // Use remaining bits for rank
        if (rank > counters[idx]) {
            counters[idx] = rank;
        }
    }
    
    // Overload for integer input (synthetic)
    void add(int item) {
        uint64_t h = hash_int(item, seed);
        uint32_t idx = h >> (64 - b);
        uint8_t rank = count_leading_zeros(h);
        if (rank > counters[idx]) {
            counters[idx] = rank;
        }
    }

    double estimate() const {
        double sum_inv = 0.0;
        for (int val : counters) sum_inv += pow(2.0, -val);

        double E = alphaMM / sum_inv;

        // Small range correction (TODO: VULL INVESTIGAR SI AQUESTA COSA ÉS REALMENT NECESSÀRIA)
        if (E <= 2.5 * m) {
            int V = 0; // Count of zero counters
            for (int val : counters) {
                if (val == 0) V++;
            }
            if (V > 0) E = m * log((double)m / V);
        }
        return E;
    }
};

// ALGORITHM 2: Recordinality (REC) 
class Recordinality {
private:
    int k;                  // Sample size (memory constraint)
    int R;                  // Recordinality counter
    set<uint64_t> sample;   // Stores the k largest hash values
    uint32_t seed;

public:
    Recordinality(int k_size, uint32_t seed_val) : k(k_size), R(0), seed(seed_val) {}

    void add(const string& item) {
        uint64_t h = hash_str(item, seed);
        process_hash(h);
    }

    void add(int item) {
        uint64_t h = hash_int(item, seed);
        process_hash(h);
    }

    // Core Logic
    void process_hash(uint64_t h) {
        // Check if the element is already in our sample
        if (sample.find(h) != sample.end()) return; 

        if (sample.size() < k) { // Fill S with the first k distinct elements
            sample.insert(h);
            R++;
        } 
        else { // Updating the set of k-records
            uint64_t current_min = *sample.begin(); // set orders low to high
            if (h > current_min) {
                sample.erase(sample.begin());
                sample.insert(h);
                R++;
            }
        }
    }

    // Estimation Formula
    double estimate() const {
        // Case 1: If R < k then we are sure that n = R
        if (R < k) return (double)R;

        // Case 2: Standard estimation using the derived formula
        double base = 1.0 + 1.0 / (double)k;
        double exponent = (double)(R - k + 1);
        return (double)k * pow(base, exponent) - 1.0;
    }
};

// DATA GENERATOR: Zipfian Distribution 
class ZipfGenerator {
private:
    int n;              // Universe size
    double alpha;       // Skew
    vector<double> cdf; // Cumulative distribution
    mt19937 gen;
    uniform_real_distribution<double> dist;

public:
    ZipfGenerator(int universe_size, double skew) 
        : n(universe_size), alpha(skew), dist(0.0, 1.0) {
        
        random_device rd;
        gen.seed(rd());

        // Precompute probabilities
        double c_n_denominator = 0.0;
        for (int i = 1; i <= n; i++) c_n_denominator += 1.0 / pow(i, alpha);

        double cumulative = 0.0;
        cdf.reserve(n);
        for (int i = 1; i <= n; i++) {
            double prob = (1.0 / pow(i, alpha)) / c_n_denominator;
            cumulative += prob;
            cdf.push_back(cumulative);
        }
        // Ensure last is exactly 1.0 to avoid precision issues
        cdf.back() = 1.0;
    }

    int next() {
        double r = dist(gen);
        // Binary search for efficiency
        auto it = lower_bound(cdf.begin(), cdf.end(), r);
        return (int)(distance(cdf.begin(), it) + 1);
    }
};

// ------------------------------------------------------------------------------------
// MAIN EXPERIMENTS

// Helper to get true cardinality and the stream itself from a file
struct DatasetInfo {
    string name;
    vector<string> stream;
    int true_cardinality;
};

DatasetInfo load_dataset(const string& filename) {
    DatasetInfo info;
    info.name = filename;
    
    string path = "datasets/" + filename;
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "Could not open " << path << ". Skipping." << endl;
        info.true_cardinality = -1; // Error flag
        return info;
    }

    set<string> uniques;
    string word;
    while (file >> word) {
        info.stream.push_back(word);
        uniques.insert(word);
    }
    info.true_cardinality = uniques.size();
    return info;
}

// Helper to read all datasets from the directory
vector<string> get_txt_files(const string& dir_path) {
    vector<string> files;
    try {
        if (!filesystem::exists(dir_path) || !filesystem::is_directory(dir_path)) {
            cerr << "Directory not found: " << dir_path << endl;
            return files;
        }

        for (const auto& entry : filesystem::directory_iterator(dir_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") files.push_back(entry.path().filename().string());
        }
    } catch (const exception& e) {
        cerr << "Filesystem error: " << e.what() << endl;
    }
    return files;
}

// 1. General comparison table
void experiment_1_comparison_table(int RUNS) {
    cout << "EXPERIMENT 1: COMPARISON TABLE (HLL vs REC vs REAL)" << endl;
    cout << "Settings: HLL(b=10 -> m=1024), REC(k=256), Runs=" << RUNS << endl;

    // Setup datasets
    vector<string> filenames = get_txt_files("datasets");
    if (filenames.empty()) cerr << "No .txt files found in 'datasets/' directory!" << endl;

    int HLL_b = 10; // m = 1024
    int REC_k = 256;

    cout << left << setw(31) << "Dataset" << setw(15) << "True Card(n)"
                 << setw(15) << "HLL Avg" << setw(15) << "HLL Error(%)"
                 << setw(15) << "REC Avg" << setw(15) << "REC Error(%)" << endl;
    cout << string(95, '-') << endl;

    // Process real datasets
    for (const auto& fname : filenames) {
        DatasetInfo data = load_dataset(fname);
        if (data.true_cardinality == -1) continue;

        double sum_hll = 0, sum_rec = 0;
        for (int r = 0; r < RUNS; ++r) {
            uint32_t seed = 1000 + r * 53;
            HyperLogLog hll(HLL_b, seed);
            Recordinality rec(REC_k, seed);

            for (const auto& w : data.stream) {
                hll.add(w);
                rec.add(w);
            }
            sum_hll += hll.estimate();
            sum_rec += rec.estimate();
        }
        double hll_avg = sum_hll / RUNS;
        double rec_avg = sum_rec / RUNS;
        double hll_err = 100.0 * abs(hll_avg - data.true_cardinality) / data.true_cardinality;
        double rec_err = 100.0 * abs(rec_avg - data.true_cardinality) / data.true_cardinality;
        cout << left << setw(31) << fname << setw(15) << data.true_cardinality 
                     << setw(15) << hll_avg << setw(15) << fixed << setprecision(2) << hll_err
                     << setw(15) << rec_avg << setw(15) << fixed << setprecision(2) << rec_err << endl;
    }

    // Process synthetic data
    int N = 100000;
    int n = 10000;
    double alpha = 1.0;
    ZipfGenerator zipf(n, alpha);
    
    // Generate stream once
    set<int> uniques;
    vector<int> synth_stream;
    for (int i = 0; i < N; i++) {
        int nxt = zipf.next();
        synth_stream.push_back(nxt);
        uniques.insert(nxt);
    }
    int true_cardinality = uniques.size();

    double sum_hll = 0, sum_rec = 0;
    for (int r = 0; r < RUNS; ++r) {
        uint32_t seed = 2000 + r * 53;
        HyperLogLog hll(HLL_b, seed);
        Recordinality rec(REC_k, seed);
        for(int x : synth_stream) {
            hll.add(x);
            rec.add(x);
        }
        sum_hll += hll.estimate();
        sum_rec += rec.estimate();
    }
    double hll_avg = sum_hll / RUNS;
    double rec_avg = sum_rec / RUNS;
    double hll_err = 100.0 * abs(hll_avg - true_cardinality) / true_cardinality;
    double rec_err = 100.0 * abs(rec_avg - true_cardinality) / true_cardinality;
    cout << left << setw(31) << "Synthetic(Zipf 1.0)" << setw(15) << true_cardinality
                 << setw(15) << hll_avg << setw(15) << fixed << setprecision(2) << hll_err
                 << setw(15) << rec_avg << setw(15) << fixed << setprecision(2) << rec_err << endl;
    cout << endl;
}

// 2. Impact of memory (m and k)
void analyze_memory_for_file(const string& filename, int RUNS) {
    DatasetInfo data = load_dataset(filename);
    if (data.true_cardinality == -1) return;

    cout << "Analyzing " << filename << " (n=" << data.true_cardinality << ")" << endl;

    // HLL analysis
    cout << "  [HyperLogLog]" << endl;
    cout << "  b | m     | Est(Avg) | StdError(%) | Theor. SE(%)" << endl;
    cout << "  -------------------------------------------------" << endl;
    
    for (int b = 4; b <= 12; b += 2) {
        int m = 1 << b;
        double sum_est = 0;
        double sq_sum_est = 0;

        for (int r = 0; r < RUNS; r++) {
            HyperLogLog hll(b, 12345 + r);
            for (const auto& w : data.stream) hll.add(w);
            double val = hll.estimate();
            sum_est += val;
            sq_sum_est += val * val;
        }

        double mean = sum_est / RUNS;
        double variance = (sq_sum_est / RUNS) - (mean * mean);
        double std_dev = sqrt(variance);
        double se_percent = (std_dev / mean) * 100.0;
        double theory_se = (1.03 / sqrt(m)) * 100.0;

        cout << "  " << left << setw(2) << b << "| " << setw(6) << m << "| " 
                             << setw(9) << (int)mean << "| " << setw(12) << fixed << setprecision(2) << se_percent << "| " << theory_se << endl;
    }

    // REC analysis
    cout << "  [Recordinality]" << endl;
    cout << "  k     | Est(Avg) | StdError(%) | Theor. SE(%)" << endl;
    cout << "  -------------------------------------------" << endl;

    vector<int> k_values = {16, 64, 256, 1024};
    for (int k : k_values) {
        double sum_est = 0;
        double sq_sum_est = 0;

        for (int r = 0; r < RUNS; r++) {
            Recordinality rec(k, 12345 + r);
            for (const auto& w : data.stream) rec.add(w);
            double val = rec.estimate();
            sum_est += val;
            sq_sum_est += val * val;
        }
        double mean = sum_est / RUNS;
        double variance = (sq_sum_est / RUNS) - (mean * mean);
        double std_dev = sqrt(variance);
        double se_percent = (std_dev / mean) * 100.0;
        double theory_se = (1.0 / sqrt(k)) * 100.0; 
        cout << "  " << left << setw(6) << k << "| " << setw(9) << (int)mean << "| " 
                     << setw(12) << fixed << setprecision(2) << se_percent << "| " << theory_se << endl;
    }
    cout << endl;
}

void experiment_2_memory_impact(int RUNS) {
    cout << "EXPERIMENT 2: MEMORY IMPACT (Observed SE vs Theoretical SE)" << endl;
    analyze_memory_for_file("dracula.txt", RUNS);
    analyze_memory_for_file("crusoe.txt", RUNS); 
}

// 3. Impact of alpha
void experiment_3_alpha_impact(int RUNS) {
    cout << "EXPERIMENT 3: IMPACT OF ALPHA (SKEW) ON ESTIMATION" << endl;

    int n = 10000;
    int N = 100000;
    int HLL_b = 10;
    int REC_k = 256;
    vector<double> alphas = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0};

    cout << left << setw(10) << "Alpha" << setw(15) << "HLL Avg" << setw(15) << "HLL Error(%)"
                 << setw(15) << "REC Avg" << setw(15) << "REC Error(%)" << endl;
    cout << string(70, '-') << endl;

    for (double a : alphas) {
        ZipfGenerator zipf(n, a);
        double sum_hll = 0, sum_rec = 0;
        
        // We must regenerate the stream for each alpha
        set<int> uniques;
        vector<int> stream;
        stream.reserve(N);
        for(int i=0; i<N; i++) {
            int nxt = zipf.next();
            stream.push_back(nxt);
            uniques.insert(nxt);
        }
        int true_cardinality = uniques.size();

        for (int r = 0; r < RUNS; ++r) {
            uint32_t seed = 3000 + r * 99;
            HyperLogLog hll(HLL_b, seed);
            Recordinality rec(REC_k, seed);
            for(int x : stream) {
                hll.add(x);
                rec.add(x);
            }
            sum_hll += hll.estimate();
            sum_rec += rec.estimate();
        }
        double hll_avg = sum_hll / RUNS;
        double rec_avg = sum_rec / RUNS;
        double hll_err = 100.0 * abs(hll_avg - true_cardinality) / true_cardinality;
        double rec_err = 100.0 * abs(rec_avg - true_cardinality) / true_cardinality;
        cout << left << setw(10) << fixed << setprecision(1) << a << setw(15) << hll_avg << setw(15) << fixed << setprecision(2) << hll_err
                     << setw(15) << rec_avg << setw(15) << fixed << setprecision(2) << rec_err << endl;
    }
    cout << endl;
}

int main() {
    // Seed random generator
    srand(time(NULL));

    int RUNS = 20;
    experiment_1_comparison_table(RUNS);
    experiment_2_memory_impact(RUNS);
    experiment_3_alpha_impact(RUNS);

    return 0;
}
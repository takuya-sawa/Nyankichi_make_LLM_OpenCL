#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include "transformer_opencl.h"
#include "math_opencl.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <locale>
#include <clocale>

// NOTE: Save this file as UTF-8 to avoid encoding issues on Windows.
// UTF-8 console output helper for Windows (uses WriteConsoleW to avoid terminal-codepage issues)
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <string>

static void print_utf8(const std::string &s) {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE || hOut == NULL) {
        std::cout << s;
        return;
    }
    int wlen = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, NULL, 0);
    if (wlen == 0) { std::cout << s; return; }
    std::wstring wbuf((size_t)wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &wbuf[0], wlen);
    if (!wbuf.empty() && wbuf.back() == L'\0') wbuf.pop_back();
    DWORD written = 0;
    WriteConsoleW(hOut, wbuf.c_str(), (DWORD)wbuf.size(), &written, NULL);
}
static void out(const std::string &s) { print_utf8(s); }
#else
static void out(const std::string &s) { std::cout << s; }
#endif

// TinyLLM OpenCL (CPU-backed stub) - main

class SimpleTokenizer
{
private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_word;

public:
    SimpleTokenizer(const std::vector<std::string>& training_texts)
    {
        BuildVocabulary(training_texts);
    }

    int GetVocabSize() const { return vocab.size(); }

    void BuildVocabulary(const std::vector<std::string>& training_texts)
    {
        std::unordered_map<std::string, bool> unique_words;

        for (const auto& text : training_texts) {
            std::istringstream iss(text);
            std::string word;
            while (iss >> word) {
                word.erase(std::remove_if(word.begin(), word.end(),
                    [](char c) { return !isalnum(c); }), word.end());

                if (!word.empty()) {
                    std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                    unique_words[word] = true;
                }
            }
        }

        vocab["<pad>"] = 0;
        id_to_word[0] = "<pad>";
        vocab["<unk>"] = 1;
        id_to_word[1] = "<unk>";

        int id = 2;
        for (const auto& word_pair : unique_words) {
            vocab[word_pair.first] = id;
            id_to_word[id] = word_pair.first;
            id++;
        }

        {
            std::ostringstream _oss; _oss << "[Tokenizer] 語彙サイズ: " << vocab.size() << "\n"; out(_oss.str());
        }
    }

    std::vector<int> Tokenize(const std::string& text)
    {
        std::vector<int> token_ids;
        std::istringstream iss(text);
        std::string word;

        while (iss >> word) {
            word.erase(std::remove_if(word.begin(), word.end(),
                [](char c) { return !isalnum(c); }), word.end());

            if (!word.empty()) {
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);

                if (vocab.find(word) != vocab.end()) {
                    token_ids.push_back(vocab[word]);
                } else {
                    token_ids.push_back(1);
                }
            }
        }

        return token_ids;
    }

    std::string IdToToken(int id) const
    {
        auto it = id_to_word.find(id);
        return it != id_to_word.end() ? it->second : "<unk>";
    }
};

std::vector<std::string> LoadTrainingData(const char* filepath)
{
    std::vector<std::string> lines;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::ostringstream _oss; _oss << "警告: ファイルが見つかりません: " << filepath << "\n"; out(_oss.str());
        return lines;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] != '#') lines.push_back(line);
    }
    file.close();
    {
        std::ostringstream _oss; _oss << "[データ] " << lines.size() << " 件の訓練文を読み込みました\n"; out(_oss.str());
    }
    return lines;
}

int main(int argc, char* argv[])
{
    // Ensure console uses UTF-8 on Windows and set locale on all platforms.
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    std::setlocale(LC_ALL, "");

    std::cout << "================================================" << std::endl;
    std::cout << "  TinyLLM - OpenCL (CPU-backed stub)" << std::endl;
    std::cout << "  Build: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << std::endl;

    bool run_train = true;
    bool run_infer = true;
    bool force_cpu = false;

    // Command-line options: --list-devices, --device <index>, --cpu, train, infer
    for (int ai = 1; ai < argc; ++ai) {
        std::string arg = argv[ai];
        if (arg == "--list-devices") {
            ListOpenCLDevices();
            return 0;
        }
        if (arg == "--device" && ai + 1 < argc) {
            int idx = atoi(argv[++ai]);
            if (!SelectOpenCLDevice(idx)) {
                out(std::string("Invalid device index\n"));
                return 1;
            }
        }
        if (arg == "--verbosity" && ai + 1 < argc) {
            int v = atoi(argv[++ai]);
            SetVerbosity(v);
            std::ostringstream _oss; _oss << "[Config] verbosity=" << v << "\n"; out(_oss.str());
            continue;
        }
        if (arg == "--cpu") {
            force_cpu = true;
        }
        if (arg == "train") { run_train = true; run_infer = false; }
        else if (arg == "infer") { run_train = false; run_infer = true; }
    }

    // Initialize OpenCL after parsing and potential device selection (skip if --cpu)
    if (!force_cpu) {
        InitOpenCL();
    } else {
        out(std::string("[Mode] Running in CPU-only mode\n"));
    }

    out(std::string("[データ] 訓練データを読み込んでいます...\n"));
    std::vector<std::string> training_texts = LoadTrainingData("data/training_data.txt");
    if (training_texts.empty()) { std::cerr << "エラー: 訓練データが見つかりません" << std::endl; return 1; }

    out(std::string("[トークナイザー] 語彙を構築しています...\n"));
    SimpleTokenizer tokenizer(training_texts);
    std::cout << std::endl;

    int vocab_size = std::max(tokenizer.GetVocabSize(), 128);
    int hidden_dim = 256;
    int num_layers = 3;
    int seq_length = 16;

    TinyLLM* model = nullptr;
    std::string checkpoint_path = "model_checkpoint.bin";
    std::ifstream check_file(checkpoint_path);
    if (check_file.good()) {
        out(std::string("[モデル] 既存のチェックポイントを読み込んでいます...\n"));
        model = TinyLLM::LoadModel(checkpoint_path.c_str());
        out(std::string("\n"));
    } else {
        out(std::string("[モデル] 新しいモデルを初期化しています...\n"));
        model = new TinyLLM(vocab_size, hidden_dim, num_layers, seq_length, 0.001f);
        out(std::string("\n"));
    }

    if (run_train) {
        const int EPOCHS = 10;
        const int STEPS_PER_EPOCH = 100;
        
        std::ostringstream _oss; 
        _oss << "[訓練] " << EPOCHS << " エポック、各 " << STEPS_PER_EPOCH << " ステップで訓練開始\n"; 
        out(_oss.str());
        
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            std::ostringstream epoch_oss; 
            epoch_oss << "\n=== エポック " << (epoch + 1) << " / " << EPOCHS << " ===\n"; 
            out(epoch_oss.str());
            
            float epoch_loss = 0.0f;
            int valid_steps = 0;
            
            for (int step = 0; step < STEPS_PER_EPOCH; step++) {
                int random_idx = rand() % training_texts.size();
                const auto& text = training_texts[random_idx];
                auto token_ids = tokenizer.Tokenize(text);
                
                if (token_ids.size() > 1) {
                    int target_id = token_ids.back();  // 修正: 先に保存
                    token_ids.pop_back();
                    
                    float loss = model->TrainStep(token_ids, target_id);
                    epoch_loss += loss;
                    valid_steps++;
                    
                    if ((step + 1) % 20 == 0) {
                        std::ostringstream step_oss; 
                        step_oss << "  ステップ " << (step + 1) << " / " << STEPS_PER_EPOCH 
                                << ": Loss = " << loss << "\n"; 
                        out(step_oss.str());
                    }
                }
            }
            
            if (valid_steps > 0) {
                std::ostringstream avg_oss;
                avg_oss << "エポック " << (epoch + 1) << " 平均Loss: " 
                       << (epoch_loss / valid_steps) << "\n";
                out(avg_oss.str());
            }
            
            // 定期的にチェックポイント保存
            if ((epoch + 1) % 5 == 0) {
                std::ostringstream cp_oss;
                cp_oss << "[保存] エポック " << (epoch + 1) << " チェックポイント保存中...\n";
                out(cp_oss.str());
                model->SaveModel(checkpoint_path.c_str());
            }
        }
        
        out(std::string("\n[モデル] 最終チェックポイントを保存しています...\n"));
        model->SaveModel(checkpoint_path.c_str());
        out(std::string("[完了] 訓練が完了しました\n\n"));
    }

    if (run_infer) {
        std::vector<std::string> test_inputs = { "I am a", "The cat is", "I like", "Cats are" };
        for (const auto& input : test_inputs) {
            auto token_ids = tokenizer.Tokenize(input);
            int predicted_id = model->Predict(token_ids);
            std::string predicted = tokenizer.IdToToken(predicted_id);
            {
                std::ostringstream _oss; _oss << "  入力: \"" << input << "\"\n"; out(_oss.str());
            }
            {
                std::ostringstream _oss; _oss << "  予測: \"" << predicted << "\"\n"; out(_oss.str());
            }
            out(std::string("\n"));
        }
    }

    delete model;
    DestroyOpenCL();
    std::cout << "Done!" << std::endl;
    return 0;
}
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include "transformer_cuda.h"
#include "math_cuda.h"

/// ===================================================================
/// TinyLLM CUDA 版 - メインプログラム
/// 
/// GPU（NVIDIA）を活用した高速言語モデル訓練・推論
/// ===================================================================

// トークナイザー
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
                // 句読点を除去（簡略版）
                word.erase(std::remove_if(word.begin(), word.end(),
                    [](char c) { return !isalnum(c); }), word.end());
                
                if (!word.empty()) {
                    std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                    unique_words[word] = true;
                }
            }
        }
        
        // 特別トークン
        vocab["<pad>"] = 0;
        id_to_word[0] = "<pad>";
        vocab["<unk>"] = 1;
        id_to_word[1] = "<unk>";
        
        // 通常の単語
        int id = 2;
        for (const auto& word_pair : unique_words) {
            vocab[word_pair.first] = id;
            id_to_word[id] = word_pair.first;
            id++;
        }
        
        std::cout << "[Tokenizer] 語彙サイズ: " << vocab.size() << std::endl;
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
                    token_ids.push_back(1);  // UNK
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

// 訓練データ読み込み
std::vector<std::string> LoadTrainingData(const char* filepath)
{
    std::vector<std::string> lines;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "警告: ファイルが見つかりません: " << filepath << std::endl;
        return lines;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // 空行とコメント（#で始まる行）をスキップ
        if (!line.empty() && line[0] != '#') {
            lines.push_back(line);
        }
    }
    
    file.close();
    std::cout << "[データ] " << lines.size() << " 件の訓練文を読み込みました" << std::endl;
    
    return lines;
}

int main(int argc, char* argv[])
{
    std::cout << "================================================" << std::endl;
    std::cout << "  TinyLLM - CUDA GPU Optimized Version" << std::endl;
    std::cout << "  High-speed Language Model" << std::endl;
    std::cout << "  Build: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << std::endl;
    
    // cuBLAS initialization
    InitCublas();
    
    // モード決定
    bool run_train = true;
    bool run_infer = true;
    
    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "train") {
            run_train = true;
            run_infer = false;
        } else if (mode == "infer") {
            run_train = false;
            run_infer = true;
        }
    }
    
    // 訓練データ読み込み
    std::cout << "[データ] 訓練データを読み込んでいます..." << std::endl;
    std::vector<std::string> training_texts = LoadTrainingData("data/training_data.txt");
    
    if (training_texts.empty()) {
        std::cerr << "エラー: 訓練データが見つかりません" << std::endl;
        return 1;
    }
    
    // トークナイザー初期化
    std::cout << "[トークナイザー] 語彙を構築しています..." << std::endl;
    SimpleTokenizer tokenizer(training_texts);
    std::cout << std::endl;
    
    // モデル初期化
    int vocab_size = std::max(tokenizer.GetVocabSize(), 128);
    int hidden_dim = 256;     // GPU 向けに増加
    int num_layers = 3;
    int seq_length = 16;
    
    TinyLLM* model = nullptr;
    
    std::string checkpoint_path = "model_checkpoint.bin";
    std::ifstream check_file(checkpoint_path);
    
    if (check_file.good()) {
        std::cout << "[モデル] 既存のチェックポイントを読み込んでいます..." << std::endl;
        model = TinyLLM::LoadModel(checkpoint_path.c_str());
        std::cout << std::endl;
    } else {
        std::cout << "[モデル] 新しいモデルを初期化しています..." << std::endl;
        model = new TinyLLM(vocab_size, hidden_dim, num_layers, seq_length, 0.001f);
        std::cout << std::endl;
    }
    
    // 訓練フェーズ
    if (run_train) {
        std::cout << "================================================" << std::endl;
        std::cout << "  Training Phase Started (GPU Accelerated)" << std::endl;
        std::cout << "================================================" << std::endl;
        std::cout << std::endl;
        
        const int EPOCHS = 5;
        const int STEPS_PER_EPOCH = 3;
        
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            std::cout << "[訓練] エポック " << (epoch + 1) << "/" << EPOCHS << std::endl;
            
            for (int step = 0; step < STEPS_PER_EPOCH; step++) {
                int random_idx = rand() % training_texts.size();
                const auto& text = training_texts[random_idx];
                auto token_ids = tokenizer.Tokenize(text);
                
                if (token_ids.size() > 1) {
                    token_ids.pop_back();  // 最後のトークンを削除
                    int target_id = tokenizer.Tokenize(text).back();
                    
                    float loss = model->TrainStep(token_ids, target_id);
                    std::cout << "  ステップ " << (step + 1) << "/" << STEPS_PER_EPOCH 
                              << ": Loss = " << loss << std::endl;
                }
            }
            
            std::cout << std::endl;
        }
        
        // チェックポイント保存
        model->SaveModel(checkpoint_path.c_str());
        std::cout << std::endl;
    }
    
    // 推論フェーズ
    if (run_infer) {
        std::cout << "================================================" << std::endl;
        std::cout << "  Inference Phase Started (GPU Execution)" << std::endl;
        std::cout << "================================================" << std::endl;
        std::cout << std::endl;
        
        std::vector<std::string> test_inputs = {
            "I am a",
            "The cat is",
            "I like",
            "Cats are"
        };
        
        std::cout << "[推論] 次のトークン予測:" << std::endl;
        std::cout << std::endl;
        
        for (const auto& input : test_inputs) {
            auto token_ids = tokenizer.Tokenize(input);
            int predicted_id = model->Predict(token_ids);
            std::string predicted = tokenizer.IdToToken(predicted_id);
            
            std::cout << "  入力: \"" << input << "\"" << std::endl;
            std::cout << "  予測: \"" << predicted << "\"" << std::endl;
            std::cout << std::endl;
        }
    }
    
    // クリーンアップ
    delete model;
    DestroyCublas();
    
    std::cout << "Done!" << std::endl;
    
    return 0;
}

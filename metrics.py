import json
import torch
import clip
from PIL import Image
from bert_score import score as bert_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 提取 assistant 的回答和图像路径
def extract_responses_and_images(data):
    responses = {}
    image_paths = {}
    for item in data:
        responses[item['id']] = item['conversations'][1]['content']
        image_paths[item['id']] = item['image']
    return responses, image_paths

# 计算 CLIP Score
def calculate_clip_score(image_paths, hypotheses):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    clip_scores = []
    for key in hypotheses:
        image_path = image_paths[key]
        hypothesis = hypotheses[key]

        # 加载并预处理图像
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # 将文本转换为 CLIP 输入
        text = clip.tokenize([hypothesis]).to(device)

        # 计算图像和文本的 CLIP 特征
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

        # 计算余弦相似度
        similarity = torch.cosine_similarity(image_features, text_features, dim=1)
        clip_scores.append(similarity.item())

    return sum(clip_scores) / len(clip_scores)

# 计算指标
def calculate_metrics(references, hypotheses, image_paths):
    # 将 references 和 hypotheses 转换为字典格式
    ref_dict = {k: [v[0]] for k, v in references.items()}
    hyp_dict = {k: [v] for k, v in hypotheses.items()}

    # BLEU
    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(ref_dict, hyp_dict)

    # CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(ref_dict, hyp_dict)

    # ROUGE
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(ref_dict, hyp_dict)

    # SPICE
    spice_scorer = Spice()
    spice_score, _ = spice_scorer.compute_score(ref_dict, hyp_dict)

    # CLIP Score
    # clip_score = calculate_clip_score(image_paths, hypotheses)

    # BERT Score
    P, R, F1 = bert_score([v[0] for v in hyp_dict.values()], [v[0] for v in ref_dict.values()], lang='zh')

    return {
        'BLEU': bleu_score,
        'CIDEr': cider_score,
        'ROUGE': rouge_score,
        'SPICE': spice_score,
        # 'CLIP Score': clip_score,
        'BERT Score': {
            'Precision': P.mean().item(),
            'Recall': R.mean().item(),
            'F1': F1.mean().item()
        }
    }

# 主函数
def main():
    # 加载数据
    output_data = load_json('result/output.json')
    ground_truth_data = load_json('finetune/data/eval.json')

    # 提取回答和图像路径
    output_responses, output_image_paths = extract_responses_and_images(output_data)
    ground_truth_responses, ground_truth_image_paths = extract_responses_and_images(ground_truth_data)

    # 确保 id 对应
    references = {k: [v] for k, v in ground_truth_responses.items()}
    hypotheses = {k: v for k, v in output_responses.items() if k in references}
    image_paths = {k: v for k, v in output_image_paths.items() if k in references}

    # 计算指标
    metrics = calculate_metrics(references, hypotheses, image_paths)

    # 将结果写入 JSON 文件
    with open('result/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
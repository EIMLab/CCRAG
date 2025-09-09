from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CoSENTLoss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset, load_dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
import json
import random


json_path = "output_folder/question_generating_for_finetuning_embedding_model.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "material" in item:
        material = item["material"]
        split_index = material.find("。")
        if split_index != -1:
            item["material"] = material[split_index + 1:].strip()

chunks = []
questions = []
labels = []

for i, item in enumerate(data):
    material = item["material"]
    filename = item["filename"]

    for j in range(1, 4):
        question_key = f"question{j}stem"
        if question_key in item:
            chunks.append(material)
            questions.append(item[question_key])
            labels.append(float(1))

    same_file_others = [d for idx, d in enumerate(data) if idx != i and d["filename"] == filename]

    sampled_negatives = random.sample(same_file_others, min(3, len(same_file_others)))
    for neg_item in sampled_negatives:
        q_keys = [k for k in neg_item if k.startswith("question") and k.endswith("stem")]
        if not q_keys:
            continue
        sampled_q_key = random.choice(q_keys)
        chunks.append(material)
        questions.append(neg_item[sampled_q_key])
        labels.append(float(0))

dataset = Dataset.from_dict({
    "text1": chunks,
    "text2": questions,
    "label": labels
})


dataset_split = dataset.train_test_split(test_size=0.125, shuffle=True, seed=42)
train_dataset = dataset_split['train']
test_dataset = dataset_split['test']


model = SentenceTransformer('model_path',
        model_card_data=SentenceTransformerModelCardData(
        language="zh",
        model_name="model_name",
    )).to('cuda:1')

loss = CoSENTLoss(model)


args = SentenceTransformerTrainingArguments(
    output_dir="output_folder/finetuning_embedding_model",
    num_train_epochs=5,             
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,    
    learning_rate=2e-5,  
    warmup_ratio=0.1, 
    fp16=False, 
    bf16=True, 
    eval_strategy="steps",  
    eval_steps=100,    
    save_strategy="steps", 
    save_steps=100, 
    save_total_limit=20, 
    logging_steps=100,  
    run_name="model_name",
)


dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["text1"],
    sentences2=test_dataset["text2"],
    scores=test_dataset["label"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
    batch_size=2   
)


trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()


results = dev_evaluator(model)
print(results)



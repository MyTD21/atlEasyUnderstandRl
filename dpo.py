import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
    get_scheduler
)

# 1. 配置与超参数
MODEL_NAME = "gpt2"  # 基础模型，可替换为更大模型
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
NUM_EPOCHS = 3
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 2. 定义偏好数据集（人类反馈数据：(prompt, chosen, rejected)）
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        #self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2无pad_token，用eos代替

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]  # 人类偏好的回答
        rejected = item["rejected"]  # 人类不偏好的回答

        # 拼接prompt和回答（保持上下文关联）
        chosen_input = self.tokenizer(
            f"{prompt}{chosen}",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        rejected_input = self.tokenizer(
            f"{prompt}{rejected}",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_input["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_input["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_input["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_input["attention_mask"].squeeze(),
            "prompt": prompt  # 保留prompt用于推理
        }


# 3. DPO损失函数（核心）
def dpo_loss(policy_model, ref_model, batch, beta=0.1):
    """
    计算DPO损失：最大化偏好回答(chosen)相对非偏好回答(rejected)的概率
    beta: 温度参数，控制偏好强度
    """
    # 策略模型（当前训练的模型）计算log概率
    chosen_logits = policy_model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"]
    ).logits
    rejected_logits = policy_model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"]
    ).logits

    # 参考模型（通常是预训练模型快照）计算log概率（冻结参数）
    with torch.no_grad():
        ref_chosen_logits = ref_model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"]
        ).logits
        ref_rejected_logits = ref_model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"]
        ).logits

    # 计算序列级别的log概率（仅对回答部分，排除prompt）
    def get_sequence_log_prob(logits, input_ids, attention_mask):
        """计算输入序列的log概率（排除padding部分）"""
        log_probs = F.log_softmax(logits, dim=-1)
        # 取每个token的log概率（除最后一个token外，因为logits是next token预测）
        seq_log_probs = torch.gather(log_probs[:, :-1, :], 2, input_ids[:, 1:].unsqueeze(2)).squeeze(2)
        # 乘以attention_mask，忽略padding
        seq_log_probs = seq_log_probs * attention_mask[:, 1:]
        return seq_log_probs.sum(dim=1)  # 序列总log概率

    # 计算策略模型和参考模型的log概率差
    policy_chosen_logprob = get_sequence_log_prob(chosen_logits, batch["chosen_input_ids"], batch["chosen_attention_mask"])
    policy_rejected_logprob = get_sequence_log_prob(rejected_logits, batch["rejected_input_ids"], batch["rejected_attention_mask"])
    ref_chosen_logprob = get_sequence_log_prob(ref_chosen_logits, batch["chosen_input_ids"], batch["chosen_attention_mask"])
    ref_rejected_logprob = get_sequence_log_prob(ref_rejected_logits, batch["rejected_input_ids"], batch["rejected_attention_mask"])

    # DPO损失公式：https://arxiv.org/abs/2305.18290
    logits = (policy_chosen_logprob - policy_rejected_logprob) - (ref_chosen_logprob - ref_rejected_logprob)
    loss = -F.logsigmoid(beta * logits).mean()  # 负对数似然，确保chosen概率更高
    return loss


# 4. 训练主函数
def train_dpo():
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)  # 参考模型（冻结）
    ref_model.requires_grad_(False)  # 参考模型不更新

    # 示例偏好数据（实际应用中需替换为真实人类反馈数据）
    demo_data = [
        {
            "prompt": "请解释什么是人工智能？",
            "chosen": "人工智能是模拟人类智能的技术，自主决策。",  # 更优回答
            "rejected": "机器人。"  # 较差回答
        },
        # 可添加更多样本...
    ]

    # 数据加载
    dataset = PreferenceDataset(demo_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    #import pdb; pdb.set_trace()
    # 优化器和学习率调度器
    optimizer = AdamW(policy_model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    policy_model.train() # 进入训练模式，保证参数可以更新
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for batch in dataloader:
            # 数据移至设备
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "prompt"}
            
            # 计算DPO损失
            loss = dpo_loss(policy_model, ref_model, batch)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")

    # 保存训练后的模型
    policy_model.save_pretrained("dpo_trained_model")
    tokenizer.save_pretrained("dpo_trained_model")
    print("模型保存至 dpo_trained_model 目录")


# 5. 推理示例
def infer():
    tokenizer = AutoTokenizer.from_pretrained("dpo_trained_model")
    model = AutoModelForCausalLM.from_pretrained("dpo_trained_model").to(DEVICE)
    model.eval()

    prompt = "请解释什么是人工智能？"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"输入: {prompt}")
    print(f"输出: {response}")


if __name__ == "__main__":
    train_dpo()  # 训练DPO模型
    infer()      # 推理测试

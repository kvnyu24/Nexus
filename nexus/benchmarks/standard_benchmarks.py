from typing import Dict, Any, Optional
import torch
import torchvision.datasets as datasets
from ..data import Dataset, Compose, Resize, ToTensor, Normalize
from .model_benchmarks import ModelBenchmark, BenchmarkConfig

class ImageClassificationBenchmark:
    def __init__(self, model, dataset, metrics=None):
        self.model = model
        self.dataset = dataset
        self.metrics = metrics or ['accuracy', 'top5_accuracy']
        
    def evaluate(self, device='cuda'):
        self.model.eval()
        self.model.to(device)
        
        total_correct = 0
        total_top5_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.dataset:
                images = batch['images'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(images)
                logits = outputs['logits']
                
                # Top-1 accuracy
                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == labels).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = torch.topk(logits, 5, dim=1)
                total_top5_correct += sum([label in pred for label, pred in zip(labels, top5_pred)])
                
                total_samples += labels.size(0)
        
        metrics = {
            'accuracy': total_correct / total_samples,
            'top5_accuracy': total_top5_correct / total_samples
        }
        
        return metrics

class LanguageModelingBenchmark:
    def __init__(self, model, dataset, metrics=None):
        self.model = model
        self.dataset = dataset
        self.metrics = metrics or ['perplexity', 'loss']
        
    def evaluate(self, device='cuda'):
        self.model.eval()
        self.model.to(device)
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in self.dataset:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                labels = batch['labels'].to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs['logits']
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += (labels != -100).sum().item()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity
        }
        
        return metrics

class ReinforcementLearningBenchmark:
    def __init__(self, agent, env, num_episodes=100, metrics=None):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.metrics = metrics or ['avg_reward', 'success_rate']
        
    def evaluate(self):
        total_rewards = []
        success_count = 0
        
        for _ in range(self.num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if info.get('success', False):
                    success_count += 1
            
            total_rewards.append(episode_reward)
        
        metrics = {
            'avg_reward': sum(total_rewards) / self.num_episodes,
            'success_rate': success_count / self.num_episodes
        }
        
        return metrics 
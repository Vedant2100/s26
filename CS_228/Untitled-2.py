# %%
!pip install gymnasium minigrid imageio transformers accelerate torch


# %% [markdown]
# *vLLM Startup cell removed (Transformers-only mode).*

# %%
# import requests
# import time

# url = "http://localhost:8000/v1/models"
# headers = {"Authorization": "Bearer empty"}

# print("Waiting for vLLM server to start (up to 20 retries)...")
# for i in range(20):
#     try:
#         response = requests.get(url, headers=headers)
#         if response.status_code == 200:
#             print("✅ vLLM loaded successfully")
#             print("Model:", response.json()['data'][0]['id'])
#             break
#     except requests.exceptions.ConnectionError:
#         print(f"[{i+1}/20] Server not ready, retrying...")
#         time.sleep(15)
# else:
#     print("❌ Server failed to start. Check vllm_server.log")

# %%
with open('vllm_server.log', 'r') as f:
    print(f.read())

# %%
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.actions import Actions

class MinigridTextWrapper:
    def __init__(self, env_id, render_mode=None):
        self.env = gym.make(env_id, render_mode=render_mode)
        self.env = FullyObsWrapper(self.env)
        self.action_map = {
            "turn_left": Actions.left,
            "turn_right": Actions.right,
            "move_forward": Actions.forward,
            "pickup": Actions.pickup,
            "drop": Actions.drop,
            "toggle": Actions.toggle,
            "done": Actions.done
        }
        self.step_count = 0

    def _base(self):
        return self.env.unwrapped

    def _find_goal_pos(self):
        base = self._base()
        grid = base.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and getattr(cell, "type", None) == "goal":
                    return (int(x), int(y))
        return None

    def _scan_objects(self):
        base = self._base()
        grid = base.grid
        objs = []
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is None: continue
                t = getattr(cell, "type", None)
                if t in ("lava", "goal", "key", "door", "box", "ball"):
                    color = getattr(cell, "color", "")
                    state = ""
                    if t == "door":
                        if getattr(cell, "is_locked", False): state = " (locked)"
                        elif getattr(cell, "is_open", False): state = " (open)"
                        else: state = " (closed)"
                    objs.append(f"{color} {t}{state} at [{x}, {y}]".strip())
        return objs

    def get_text_obs(self, obs):
        base = self._base()
        ax, ay = int(base.agent_pos[0]), int(base.agent_pos[1])
        facing = ["right", "down", "left", "up"][int(base.agent_dir)]
        
        goal_pos = self._find_goal_pos()
        target_str = f"Goal is at {goal_pos}." if goal_pos else "No visible goal."
        
        desc = f"Agent is at [{ax}, {ay}] facing {facing}. {target_str} "
        
        fx, fy = int(base.front_pos[0]), int(base.front_pos[1])
        front_obj = base.grid.get(fx, fy)
        if front_obj:
            desc += f"Directly in front of you is a {front_obj.type}. "
        else:
            desc += "The cell in front of you is empty. "
            
        objs = self._scan_objects()
        if objs: desc += "Detected objects: " + ", ".join(objs) + "."
        
        if hasattr(base, "carrying") and base.carrying:
            desc += f" You are carrying a {base.carrying.type}."
            
        return desc

    def reset(self):
        self.step_count = 0
        obs, _ = self.env.reset()
        return self.get_text_obs(obs)

    def step(self, action_str):
        self.step_count += 1
        action = self.action_map.get(action_str.lower(), Actions.forward)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.get_text_obs(obs), reward, terminated or truncated, info


# %%
import imageio
import os

def evaluate_agent(agent, env_name="MiniGrid-LavaGapS6-v0", num_episodes=10, max_steps_per_episode=100, gif_folder="episode_gifs"):
    print(f"Evaluating {env_name} for {num_episodes} episodes")
    env = MinigridTextWrapper(env_name, render_mode="rgb_array")
    os.makedirs(gif_folder, exist_ok=True)
    metrics = {"success_count": 0, "total_steps_success": 0, "total_inference_time": 0, "total_actions": 0}
    
    for episode in range(num_episodes):
        obs = env.reset()
        agent.reset()
        done, step_count, episode_reward, frames = False, 0, 0, []
        frames.append(env.env.render())
        
        goal_pos = env._find_goal_pos()
        
        while not done and step_count < max_steps_per_episode:
            action, response, inf_time = agent.act(obs, env.env)
            metrics["total_inference_time"] += inf_time
            metrics["total_actions"] += 1
            obs, reward, done, _ = env.step(action)
            step_count += 1
            episode_reward += reward
            frames.append(env.env.render())
            
        # Robust success check (coord or reward)
        current_pos = list(env.env.unwrapped.agent_pos)
        reached_goal = (goal_pos and current_pos[0] == goal_pos[0] and current_pos[1] == goal_pos[1])
        success = (episode_reward > 0) or reached_goal
        
        if success:
            metrics["success_count"] += 1
            metrics["total_steps_success"] += step_count
            if hasattr(agent, 'learn'): agent.learn(1.0) # Episode was a success
            
        gif_path = f"{gif_folder}/episode_{episode+1}.gif"
        imageio.mimsave(gif_path, frames, fps=5)
        print(f"Episode {episode+1}/{num_episodes} | Success: {success} | Steps: {step_count}")
        
    sc = metrics["success_count"]
    sr = sc / num_episodes * 100
    asf = metrics["total_steps_success"] / sc if sc > 0 else 0
    ait = metrics["total_inference_time"] / metrics["total_actions"]
    print(f"\nSuccess Rate: {sr:.2f}% | Avg Steps: {asf:.2f} | Avg Inf Time: {ait:.4f}s")
    return metrics


# %%
import matplotlib.pyplot as plt

def debug_agent(agent, steps=10):

    env = MinigridTextWrapper("MiniGrid-LavaGapS6-v0", render_mode="rgb_array")

    obs = env.reset()
    agent.reset()

    for step in range(steps):

        print(f"\nSTEP {step+1}")

        print("\nOBSERVATION:\n", obs)

        prompt = agent.build_prompt(obs)
        print("\nPROMPT:\n", prompt)

        action, response, latency = agent.act(obs)

        print("\nLLM RESPONSE:\n", response)
        print("\nACTION:", action)

        frame = env.env.render()

        plt.imshow(frame)
        plt.axis("off")
        plt.show()

        obs, reward, done, _ = env.step(action)

        if done:
            print("\nEpisode finished | reward:", reward)
            break

    env.env.close()

# %%
import torch
import re
import time
from collections import deque
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

@dataclass
class ThoughtTemplate:
    name: str
    description: str
    reasoning_pattern: str
    usage_count: int = 0
    success_rate: float = 0.0
    
    def instantiate(self, d_dict):
        context = []
        if d_dict.get("agent_pos"): 
            context.append(f"You are at {d_dict['agent_pos']} facing {d_dict.get('facing', 'unknown')}.")
        if d_dict.get("target_pos"): 
            context.append(f"Target is at {d_dict['target_pos']}.")
        if d_dict.get("front_object"): 
            context.append(f"In front is a {d_dict['front_object']}.")
        if d_dict.get("nearby_objects"): 
            context.append(f"Nearby: {', '.join(d_dict['nearby_objects'])}.")
            
        return f"[TEMPLATE: {self.name}]\nContext: {' '.join(context)}\nPattern: {self.reasoning_pattern}"

ALL_TEMPLATES = [
    ThoughtTemplate("Direct Navigation", "Visible target.", "1. Locate target 2. Align orientation 3. Move forward until reached."),
    ThoughtTemplate("Obstacle Avoidance", "Detour around lava/walls.", "1. Identify blocker position 2. Calculate detour path 3. Execute turns to bypass."),
    ThoughtTemplate("Object Acquisition", "Pickup key/ball.", "1. Navigate to object 2. Face object 3. Execute pickup command."),
    ThoughtTemplate("Unlock Door", "Unlock and open door.", "1. Acquire key 2. Navigate to door 3. Face door 4. Execute toggle command."),
    ThoughtTemplate("Clear Path", "Move object aside.", "1. Navigate to object 2. Face it 3. Pickup 4. Turn and drop in empty cell."),
]

class MetaBuffer:
    def __init__(self, size=5):
        self.templates = {t.name: t for t in ALL_TEMPLATES[:size]}
        self._learn_counter = 0
        
    def retrieve(self, d_dict):
        desc = str(d_dict).lower()
        best, max_s = None, -1.0
        for n, t in self.templates.items():
            base_score = (t.success_rate if t.usage_count > 0 else 0.5)
            keyword_bonus = 0.0
            if "Obstacle Avoidance" in n and "lava" in desc: 
                keyword_bonus = 1.0
            elif "Unlock Door" in n and ("door" in desc or "key" in desc): 
                keyword_bonus = 1.0
            elif "Object Acquisition" in n and ("pickup" in desc or "ball" in desc): 
                keyword_bonus = 1.0
            elif "Direct Navigation" in n and not any(k in desc for k in ["lava", "door", "key", "pickup", "ball"]): 
                keyword_bonus = 1.0
            
            score = base_score + keyword_bonus
            if score > max_s: max_s, best = score, t
        return best
        
    def learn(self, memory, llm_client):
        self._learn_counter += 1
        name = f"Learned_{self._learn_counter}"
        trajectory = "\n".join([f"Obs: {o} -> Action: {a}" for o, a in memory])
        sys_prompt = "You are a MiniGrid expert. Summarize this successful trajectory into a 3-4 step generic reasoning pattern."
        user_prompt = f"Trajectory:\n{trajectory}\n\nGENERIC PATTERN:"
        pattern, _, _ = llm_client.query(sys_prompt, user_prompt)
        self.templates[name] = ThoughtTemplate(name, "Learned from history.", pattern.strip(), usage_count=1, success_rate=1.0)
        print(f"[MetaBuffer] Learned new summarized template: {name}")

class ProblemDistiller:
    @staticmethod
    def distill(obs, env=None):
        if env and hasattr(env, "unwrapped"):
            base = env.unwrapped
            d = {
                'agent_pos': list(base.agent_pos),
                'facing': ["right", "down", "left", "up"][int(base.agent_dir)],
                'front_object': base.grid.get(*base.front_pos).type if base.grid.get(*base.front_pos) else None,
                'nearby_objects': []
            }
            for x in range(base.grid.width):
                for y in range(base.grid.height):
                    cell = base.grid.get(x,y)
                    if not cell: continue
                    if cell.type == 'goal': d['target_pos'] = [x, y]
                    elif cell.type in ('key', 'door', 'ball', 'lava'):
                        d['nearby_objects'].append(f"{cell.type} at [{x}, {y}]")
            return d
        d = {'agent_pos':None, 'facing':None, 'target_pos':None, 'front_object':None, 'nearby_objects':[]}
        # FIXED REGEX: Handle both brackets and parentheses for coordinates
        m = re.search(r'Agent is at [\(\[](\d+),\s*(\d+)[\)\]] facing (\w+)', obs)
        if m: d['agent_pos'], d['facing'] = [int(m.group(1)), int(m.group(2))], m.group(3)
        m = re.search(r'Goal is at (.*?)\.', obs)
        if m: 
            pos_match = re.search(r'[\(\[](\d+),\s*(\d+)[\)\]]', m.group(1))
            if pos_match: d['target_pos'] = [int(pos_match.group(1)), int(pos_match.group(2))]
        return d

class LocalLLMClient:
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Loading {model_name} (Transformers Explicit Client)... ")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        
    def query(self, sys, user):
        start_time = time.time()
        messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=64, 
                do_sample=False, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response, 0, time.time() - start_time

class BoTAgent:
    def __init__(self, model='Qwen/Qwen2.5-7B-Instruct', h_size=3):
        self.llm = LocalLLMClient(model)
        self.buffer = MetaBuffer()
        self.h_size, self.memory = h_size, []
        
    def act(self, obs, env=None):
        dist = ProblemDistiller.distill(obs, env)
        t = self.buffer.retrieve(dist)
        aid = t.instantiate(dist) if t else "Analyze context and plan optimal steps."
        hist = self.memory[-self.h_size:] if self.h_size > 0 else []
        sys_prompt = "You are a MiniGrid agent. Format: Thought: <reasoning> Action: <move_forward|turn_left|turn_right|pickup|drop|toggle>"
        user_prompt = f"Current Obs: {obs}\nBuffer Template:\n{aid}\nRecent History: {hist}\nAction:"
        res, _, lat = self.llm.query(sys_prompt, user_prompt)
        act = "move_forward"
        m = re.search(r'Action:\s*(\w+)', res, re.IGNORECASE)
        if m: act = m.group(1).lower().strip()
        else:
            for va in ["turn_left", "turn_right", "move_forward", "pickup", "drop", "toggle"]:
                if va in res.lower(): act = va; break
        self.memory.append((obs, act))
        return act, res, lat
        
    def learn(self, total_reward):
        if total_reward > 0 and len(self.memory) > 0:
            self.buffer.learn(self.memory, self.llm)
            
    def reset(self):
        self.memory = []


# %%
env = MinigridTextWrapper("MiniGrid-LavaGapS6-v0")
agent = BoTAgent()

# %%
print('\n--- Evaluating on MiniGrid-Empty-8x8-v0 ---')
evaluate_agent(agent,
               env_name="MiniGrid-Empty-8x8-v0",
               num_episodes=10,
               max_steps_per_episode=100)

# %%
print('\n--- Evaluating on MiniGrid-LavaGapS6-v0 ---')
evaluate_agent(agent,
               env_name="MiniGrid-LavaGapS6-v0",
               num_episodes=10,
               max_steps_per_episode=100)

# %%
print('\n--- Evaluating on MiniGrid-LavaCrossingS9N2-v0 ---')
evaluate_agent(agent,
               env_name="MiniGrid-LavaCrossingS9N2-v0",
               num_episodes=10,
               max_steps_per_episode=100)



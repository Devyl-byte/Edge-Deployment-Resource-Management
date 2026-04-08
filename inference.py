import os
import json
from openai import OpenAI
from env import EdgeEnv
from models import EdgeAction

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "edge-deployment-env")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:3b")
HF_TOKEN = os.getenv("HF_TOKEN" ,"ollama-local")

client = OpenAI(
    base_url=API_BASE_URL, 
    api_key=HF_TOKEN
)

def run_inference():
    env = EdgeEnv()
    tasks = ["task_stable_edge", "task_fluctuating_memory", "task_low_battery_spike"]
    BENCHMARK_NAME = "EdgeDeployment"

    for task in tasks:
        # [START] 
        print(f"[START] task={task} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)
        
        step_count = 0
        rewards_list = [] 
        
        try:
            env.set_task(task)
            obs = env.reset()
            done = False
            
            while not done:
                
                print(f"[DEBUG] Observation: {obs.model_dump_json()}")
                
                prompt = f"""
                System Status: RAM: {obs.ram_usage:.1f}%, Battery: {obs.battery_level:.1f}%
                
                Rules:
                1. If RAM is > 80.0, "kill_process" must be true.
                2. If Battery is < 40.0, "throttle_cpu" must be true.
                3. Otherwise, set everything to false.

                Respond ONLY with a raw JSON object containing EXACTLY these three boolean keys. No markdown, no explanation.
                Example: {{"kill_process": false, "throttle_cpu": false, "route_to_cloud": false}}
                """

                error_msg = "null"
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0 # 0.0 forces the model to be completely robotic and follow the rules
                    )
                    
                    raw_output = response.choices[0].message.content.strip()
                    raw_output = raw_output.replace("```json", "").replace("```", "").strip()
                        
                    action_data = json.loads(raw_output)
                    
                    action = EdgeAction(
                        kill_process=action_data.get("kill_process", False),
                        throttle_cpu=action_data.get("throttle_cpu", False),
                        route_to_cloud=action_data.get("route_to_cloud", False)
                    )
                except Exception as e:
                    error_msg = f'"{str(e)}"'
                    print(f"[DEBUG] ⚠️ ERROR: {e}")
                    action = EdgeAction(kill_process=False, throttle_cpu=False, route_to_cloud=False)
                
                obs, reward, done, info = env.step(action)
                
                # Track values
                step_count += 1
                reward_val = reward.score
                rewards_list.append(reward_val)
                
                done_str = str(done).lower()
                action_str = action.model_dump_json().replace(" ", "")
                print(f"[STEP] step={step_count} action={action_str} reward={reward_val:.2f} done={done_str} error={error_msg}", flush=True)

        finally:
            # MANDATORY: Close environment and print the [END] line always
            try:
                env.close()
            except:
                pass

            final_average_score = sum(rewards_list) / step_count if step_count > 0 else 0.0
            success_str = "true" if final_average_score > 0.0 else "false"
            rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
            
            print(f"[END] success={success_str} steps={step_count} score={final_average_score:.2f} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    run_inference()
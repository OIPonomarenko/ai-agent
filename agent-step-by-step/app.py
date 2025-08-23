# --- minimal dependencies ---
import os, re, json, requests
import gradio as gr
import pandas as pd
from huggingface_hub import InferenceClient  # add to requirements.txt

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- provider selection (HF serverless text-generation by default; optional Groq) ---
def select_model():
    provider = os.getenv("PROVIDER", "hf").lower()
    if provider == "groq":
        # Groq uses chat route; pick any free-tier model you have access to
        return {"provider": "groq", "model": os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant")}
    # HF serverless text-generation (no chat route)
    return {"provider": "hf", "model": os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")}

class BasicAgent:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.cfg = select_model()
        self.hf = InferenceClient(token=os.getenv("HF_TOKEN")) if self.cfg["provider"] == "hf" else None

    # tiny arithmetic (e.g., "12 + 3", "7*8")
    def _maybe_calc(self, q: str):
        m = re.search(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)", q)
        if not m: return None
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        try:
            return str(int(eval(f"{a}{op}{b}")))  # integer form when possible
        except Exception:
            return None

    # optional: try fetching a helper file for this task_id
    def _fetch_file_text(self, task_id: str | None):
        if not task_id: return None
        try:
            r = requests.get(f"{self.api_url}/files/{task_id}", timeout=20)
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if "application/json" in ct:
                return json.dumps(r.json(), ensure_ascii=False)
            return r.text
        except Exception:
            return None

    # single LLM call; enforce bare answer
    def _llm(self, prompt: str) -> str:
        model = self.cfg["model"]
        if self.cfg["provider"] == "hf":
            try:
                # Try text-generation first
                out = self.hf.text_generation(
                    model=model, prompt=prompt, max_new_tokens=128, temperature=0.2
                )
                return out.strip()
            except Exception as e:
                # If the backend says “Supported task: conversational”, retry with chat
                if "supported task: conversational" in str(e).lower():
                    chat = self.hf.chat_completion(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=128,
                        temperature=0.2,
                    )
                    return chat.choices[0].message["content"].strip()
                raise

        # Groq (chat.completions)
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}"},
            json={"model": self.cfg["model"], "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.2, "max_tokens": 128},
            timeout=40,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()

    # change the template call to pass task_id as second arg
    def __call__(self, question: str, task_id: str | None = None) -> str:
        # 1) quick math
        calc = self._maybe_calc(question)
        if calc is not None:
            return calc

        # 2) tiny context from attached file (if any)
        ctx = self._fetch_file_text(task_id)
        sys = ("Answer exactly. Return only the final answer string with no prefixes or explanations. "
               "If the answer is a number, output only the number.")
        prompt = f"{sys}\n\nQuestion: {question}\n"
        if ctx:
            prompt += f"\nContext:\n{ctx[:2000]}\n"

        ans = self._llm(prompt).strip().splitlines()[0]
        # strip common wrappers just in case
        for pre in ("final answer:", "answer:", "final:", "prediction:"):
            if ans.lower().startswith(pre): ans = ans[len(pre):].strip()
        return ans

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else ""

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent(api_url=api_url)
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        questions_data = questions_data[:1]
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text, task_id)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    app = demo.queue()
    demo.launch(debug=False, share=False)

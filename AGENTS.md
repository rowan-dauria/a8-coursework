# Agent Onboarding & Project Map (AGENTS.md)

This workspace contains coursework material for the practical examination on Multi-Agent Systems and Agentic AI, as specified in [a8-coursework.md](./a8-coursework.md).

The workspace is structured as follows:
* **Parent Repository (Root)**: Contains the coursework description ([a8-coursework.md](./a8-coursework.md)), general tools/guides (like tmux and gcloud command sheets), and is used for completing the written theory questions (Part I.4) and the adaptive planning critique (Part II).
* **Gemma GRPO Subdirectory (`tpu-2026-fork/`)**: The relevant sub-repository for the practical Gemma GRPO reinforcement learning finetuning project on Google Cloud TPUs (Part I.1-I.3).

---

## 🎯 Project Overview (Gemma GRPO)
The objective of the TPU project is to train a small instruction-tuned model (`google/gemma-3-1b-it`) to solve grade-school math word problems from the **GSM8K** dataset using GRPO on TPU resources.
*   **Methodology:** Reinforcement learning
*   **Algorithm:** GRPO
*   **Efficiency:** Parameter-efficient finetuning using **LoRA** adapters
*   **Frameworks:** JAX / Flax / Tunix / Qwix

> [!IMPORTANT]
> **Active Environment & TPU VM Context:**
> * **Active TPU VM Name:** `dakolo` (University-provided)
> * **Project ID:** `tpu-2026`
> * **Zone:** `us-east5-a`
> * **Note on Previous VMs:** The VM named `rowan` on project `a8-coursework` was a personal test VM and is no longer the active target. Use `dakolo` on project `tpu-2026` for active project runs unless specifically told to use Rowan's personal VM.

---

## 🗺️ Repository Map (tpu-2026-fork)

### Setup & Infrastructure (in `tpu-2026-fork/`)
*   [`tpu-setup.md`](./tpu-2026-fork/tpu-setup.md): Steps for provisioning TPU, port-forwarding, and setting up environment.
*   [`bootstrap.sh`](./tpu-2026-fork/bootstrap.sh): Creates virtual environment and installs dependencies on TPU VM.
*   [`create_tpu_env.sh`](./tpu-2026-fork/create_tpu_env.sh): Gcloud script to provision a TPU VM on us-east5.
*   [`requirements.txt`](./tpu-2026-fork/requirements.txt): Pinned requirements for JAX, Flax, Orbax, and TensorBoard compatibility.

### Source Files (in `tpu-2026-fork/scripts/`)
*   [`config.py`](./tpu-2026-fork/scripts/config.py): Central location for all training hyperparameters and file paths.
*   [`data.py`](./tpu-2026-fork/scripts/data.py): GSM8K dataset loading, prompt template wrapping, and token formatting.
*   [`rewards.py`](./tpu-2026-fork/scripts/rewards.py): Four programmatic reward functions tracking formatting and math correctness.
*   [`model.py`](./tpu-2026-fork/scripts/model.py): Initializes Gemma-3 base model, TPU sharding mesh, and LoRA.
*   [`train.py`](./tpu-2026-fork/scripts/train.py): Main GRPO training execution loop using JAX and Tunix.
*   [`run_tmux.sh`](./tpu-2026-fork/scripts/run_tmux.sh): Launches background training inside a persistent tmux session.
*   [`evaluate.py`](./tpu-2026-fork/scripts/evaluate.py): Benchmarking script to evaluate model and adapter checkpoint accuracy.
*   [`chat.py`](./tpu-2026-fork/scripts/chat.py): Interactive REPL interface to prompt and test trained checkpoints.

---

## 💡 Important Rules & Context for Agents
1.  **Python Environment:** Always run scripts with ~/venvs/tunix virtual environment activated on the TPU VM.
2.  **Training Disconnections:** Always run training inside tmux to prevent shell disconnection crashes.
3.  **Checkpointing Volatility:** Copy checkpoints from volatile /tmp to home directory for backup.
4.  **Report Requirements:** Before editing the report, report assets, or report-related prose, read [`REPORT-REQS.md`](./report/REPORT-REQS.md) for the report-level submission, formatting, authorship, link, and logging requirements. When making prose changes to the report, also read [`STYLE.md`](./STYLE.md) and base the writing style on it without referencing its specific content.

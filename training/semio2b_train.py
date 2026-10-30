# Sémio-2 : entraîne un adaptateur LoRA sur Mistral-7B-Instruct-v0.2 (compatible Cloudflare Workers AI)
# Lancer dans Colab (GPU T4) : exec(...) depuis le notebook. Unsloth doit être installé.
import json, gzip, random, re, os, torch, inspect, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from huggingface_hub import hf_hub_download, HfApi
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

AUTEUR = "YM"
HF_REPO = "Medwebsite/semio-2b-mistral7b-lora"
MAX_STEPS = int(os.environ.get("SEMIO_STEPS", "1200"))
N_EVAL = 100
random.seed(3407)

SYSTEM = (f"Tu es Sémio, l'assistant médical créé par {AUTEUR} pour les étudiants en médecine. "
          "Tu réponds en français, de façon exacte et pédagogique.")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit", max_seq_length=1024, load_in_4bit=True)

def read_json_any(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        txt = f.read().strip()
    if txt.startswith("["):
        return json.loads(txt)
    return [json.loads(l) for l in txt.splitlines() if l.strip()]

def qcm_prompt(question, options, case=None, multi=False):
    s = f"Cas clinique : {case[:2500]}\n\n" if case else ""
    s += f"Question : {question}\n"
    for k in "ABCDE":
        if options.get(k):
            s += f"{k}. {options[k]}\n"
    s += "\n" + ("Plusieurs réponses peuvent être justes. " if multi else "Une seule réponse est juste. ")
    return s + "Donne la ou les bonnes réponses."

def qcm_answer(letters, options):
    letters = [l for l in letters if l in options]
    head = "Bonne réponse" if len(letters) == 1 else "Bonnes réponses"
    return f"{head} : **{', '.join(letters)}**\n" + "\n".join(f"- **{l}.** {options[l]}" for l in letters)

def conv(u, a):
    return [{"role": "user", "content": SYSTEM + "\n\n" + u}, {"role": "assistant", "content": a}]

rows, eval_rows = [], []
for cfg in ("mcqu", "mcqm"):
    for split in ("train", "validation"):
        p = hf_hub_download("ANR-MALADES/MediQAl", f"{cfg}/{split}.json", repo_type="dataset")
        for r in read_json_any(p):
            opts = {k.upper(): (r.get(f"answer_{k}") or "").strip() for k in "abcde"}
            good = [x.strip().upper() for x in str(r.get("correct_answers", "")).split(",") if x.strip()]
            if not good:
                continue
            q = qcm_prompt(r["question"], opts, r.get("clinical_case"), multi=(cfg == "mcqm"))
            if split == "validation" and cfg == "mcqu":
                eval_rows.append((q, good))
            elif split == "train":
                rows.append(conv(q, qcm_answer(good, opts)))
try:
    p = hf_hub_download("uy-rrodriguez/FrenchMedMCQA-extended", "train.json.gz", repo_type="dataset")
    for r in read_json_any(p):
        ans = r.get("answers") or {}
        opts = {k.upper(): (ans.get(k) or "").strip() for k in "abcde"}
        good = [x.strip().upper() for x in (r.get("correct_answers") or [])]
        if good:
            rows.append(conv(qcm_prompt(r["question"], opts, multi=len(good) > 1), qcm_answer(good, opts)))
except Exception as e:
    print("FrenchMedMCQA ignoré :", e)
# Données complémentaires (CC BY-NC 4.0) : darija, FMPC/EDN, anglais
HK_EVAL = {"fr_fmpc_gold": [], "darija_hakim_triage": []}
try:
    p = hf_hub_download("Jip7e/hakim-decider-data", "train_35k_curated.jsonl", repo_type="dataset")
    n_hk = 0
    for r in read_json_any(p):
        opts_l = r.get("options") or []
        ai = r.get("answer_idx")
        if not opts_l or ai is None or not (0 <= ai < len(opts_l)) or len(opts_l) > 5:
            continue
        opts = {"ABCDE"[i]: str(o).strip() for i, o in enumerate(opts_l)}
        rows.append(conv(qcm_prompt(r["question"], opts, (r.get("state") or "").strip() or None), qcm_answer(["ABCDE"[ai]], opts)))
        n_hk += 1
    print("Données complémentaires : exemples ajoutés", n_hk)
    p = hf_hub_download("Jip7e/hakim-decider-data", "eval_benchmarks.jsonl", repo_type="dataset")
    for r in read_json_any(p):
        if r.get("suite") in HK_EVAL and r.get("qtype") == "choice" and 2 <= len(r.get("options") or []) <= 5:
            opts = {"ABCDE"[i]: str(o).strip() for i, o in enumerate(r["options"])}
            HK_EVAL[r["suite"]].append((qcm_prompt(r["question"], opts, (r.get("state") or "").strip() or None), ["ABCDE"[r["answer_idx"]]]))
except Exception as e:
    print("Données complémentaires ignorées :", e)
for q, a in [("Qui es-tu ?", f"Je suis Sémio, un assistant médical créé par {AUTEUR} pour aider les étudiants en médecine."),
             ("Qui t'a créé ?", f"J'ai été créé par {AUTEUR}. Je m'appelle Sémio.")]:
    for _ in range(4):
        rows.append(conv(q, a))
print("Exemples :", len(rows), "| QCM d'évaluation :", len(eval_rows))

random.shuffle(rows)
ds = Dataset.from_list([{"text": tokenizer.apply_chat_template(c, tokenize=False)} for c in rows])
random.shuffle(eval_rows)
EVAL = eval_rows[:N_EVAL]
SETS = {"MediQAl": EVAL}
for k, v in HK_EVAL.items():
    random.shuffle(v)
    if v: SETS[k] = v[:N_EVAL]

def score1(items):
    ok = 0
    for q, good in items:
        msgs = [{"role": "user", "content": SYSTEM + "\n\n" + q + "\nRéponds uniquement par la lettre."}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if not torch.is_tensor(ids) else ids).to("cuda")
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=12, do_sample=False, max_length=None)
        rep = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        m = re.findall(r"\b([A-E])\b", rep.upper())
        ok += bool(m) and m[0] == good[0]
    return 100 * ok / len(items)

def score():
    FastLanguageModel.for_inference(model)
    res = {k: round(score1(v), 1) for k, v in SETS.items()}
    FastLanguageModel.for_training(model)
    return res

avant = score()
print("Score AVANT :", avant, flush=True)

model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32, lora_dropout=0, bias="none",
    target_modules=["q_proj", "v_proj"], use_gradient_checkpointing="unsloth", random_state=3407)
cfg_kw = dict(dataset_text_field="text", per_device_train_batch_size=2, gradient_accumulation_steps=4,
              warmup_steps=10, max_steps=MAX_STEPS, learning_rate=2e-4, logging_steps=20, optim="adamw_8bit",
              weight_decay=0.01, lr_scheduler_type="linear", seed=3407, report_to="none", output_dir="outputs2",
              disable_tqdm=True)
params = inspect.signature(SFTConfig.__init__).parameters
cfg_kw["max_length" if "max_length" in params else "max_seq_length"] = 1024
try:
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=SFTConfig(**cfg_kw))
except TypeError:
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds, args=SFTConfig(**cfg_kw))
trainer = train_on_responses_only(trainer, instruction_part="[INST]", response_part="[/INST]")
print("Entraînement…", MAX_STEPS, "étapes", flush=True)
trainer.train()

apres = score()
print("Score AVANT :", avant, " ->  APRÈS :", apres, flush=True)

out_dir = "/kaggle/working/semio2b_lora"
model.save_pretrained(out_dir)
cfg_path = os.path.join(out_dir, "adapter_config.json")
cfg = json.load(open(cfg_path))
cfg["model_type"] = "mistral"
json.dump(cfg, open(cfg_path, "w"), indent=2)
for f in os.listdir(out_dir):
    if f not in ("adapter_config.json", "adapter_model.safetensors"):
        os.remove(os.path.join(out_dir, f))
open(os.path.join(out_dir, "README.md"), "w").write(
    f"---\nlicense: cc-by-nc-4.0\nbase_model: mistralai/Mistral-7B-Instruct-v0.2\nlanguage: [fr]\ntags: [medical, lora]\n---\n"
    f"# Sémio-2b (LoRA)\n\nAdaptateur médical créé par {AUTEUR}, affiné sur MediQAl (CC BY 4.0), FrenchMedMCQA (Apache 2.0) et Jip7e/hakim-decider-data (CC BY-NC 4.0, usage non commercial).\n"
    f"Scores sur {N_EVAL} QCM jamais vus par jeu : avant {avant}, après {apres}.\nOutil pédagogique : ne remplace pas un avis médical.\n")
api = HfApi()
api.create_repo(HF_REPO, exist_ok=True)
api.upload_folder(folder_path=out_dir, repo_id=HF_REPO)
print("Publié : https://huggingface.co/" + HF_REPO, os.listdir(out_dir))

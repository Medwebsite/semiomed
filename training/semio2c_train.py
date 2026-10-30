# Sémio-2c : LoRA sur Mistral-7B-Instruct-v0.2 (compatible Cloudflare Workers AI), données libres uniquement.
# Étape 1 : traduction en darija de QCM libres (MediQAl, FrenchMedMCQA) par Gemma 4 E4B.
# Étape 2 : entraînement sur MediQAl + FrenchMedMCQA + MedMCQA + QCM traduits en darija.
import json, gzip, random, re, os, time, gc, inspect, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch
from huggingface_hub import hf_hub_download, HfApi

AUTEUR = "YM"
HF_REPO = "Medwebsite/semio-2c-mistral7b-lora"
MAX_STEPS = int(os.environ.get("SEMIO_STEPS", "1800"))
N_DARIJA = int(os.environ.get("SEMIO_DARIJA", "4000"))
TRAD_MINUTES = float(os.environ.get("SEMIO_TRAD_MIN", "110"))
N_EVAL = 100
WORK = "/kaggle/working"
random.seed(3407)

SYSTEM = (f"Tu es Sémio, l'assistant médical créé par {AUTEUR}. "
          "Tu réponds de façon exacte et pédagogique.")

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

# ---------- Données libres ----------
items = []          # (question, options, case, good, multi) pour l'entraînement en français
eval_rows = []
for cfg in ("mcqu", "mcqm"):
    for split in ("train", "validation"):
        p = hf_hub_download("ANR-MALADES/MediQAl", f"{cfg}/{split}.json", repo_type="dataset")
        for r in read_json_any(p):
            opts = {k.upper(): (r.get(f"answer_{k}") or "").strip() for k in "abcde"}
            opts = {k: v for k, v in opts.items() if v}
            good = [x.strip().upper() for x in str(r.get("correct_answers", "")).split(",") if x.strip()]
            if not good:
                continue
            if split == "validation" and cfg == "mcqu":
                eval_rows.append((qcm_prompt(r["question"], opts, r.get("clinical_case")), good))
            elif split == "train":
                items.append((r["question"], opts, r.get("clinical_case"), good, cfg == "mcqm"))
try:
    p = hf_hub_download("uy-rrodriguez/FrenchMedMCQA-extended", "train.json.gz", repo_type="dataset")
    for r in read_json_any(p):
        ans = r.get("answers") or {}
        opts = {k.upper(): (ans.get(k) or "").strip() for k in "abcde"}
        opts = {k: v for k, v in opts.items() if v}
        good = [x.strip().upper() for x in (r.get("correct_answers") or [])]
        if good:
            items.append((r["question"], opts, None, good, len(good) > 1))
except Exception as e:
    print("FrenchMedMCQA ignoré :", e)
print("QCM français :", len(items), flush=True)

en_rows = []
try:
    from datasets import load_dataset
    mm = load_dataset("openlifescienceai/medmcqa", split="train").shuffle(seed=3407).select(range(8000))
    for r in mm:
        opts = {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}
        if r["cop"] is None or not (0 <= r["cop"] <= 3):
            continue
        en_rows.append(conv(qcm_prompt(r["question"], opts), qcm_answer(["ABCD"[r["cop"]]], opts)))
except Exception as e:
    print("MedMCQA ignoré :", e)
print("QCM anglais (MedMCQA) :", len(en_rows), flush=True)

# Jeux d'évaluation (seulement pour mesurer, jamais pour entraîner)
SETS = {}
rng = random.Random(123)
rng.shuffle(eval_rows)
SETS["MediQAl"] = eval_rows[:N_EVAL]
try:
    p = hf_hub_download("Jip7e/hakim-decider-data", "eval_benchmarks.jsonl", repo_type="dataset")
    ev = {"fr_fmpc_gold": [], "darija_hakim_triage": []}
    for r in read_json_any(p):
        if r.get("suite") in ev and r.get("qtype") == "choice" and 2 <= len(r.get("options") or []) <= 5:
            opts = {"ABCDE"[i]: str(o).strip() for i, o in enumerate(r["options"])}
            ev[r["suite"]].append((qcm_prompt(r["question"], opts, (r.get("state") or "").strip() or None), ["ABCDE"[r["answer_idx"]]]))
    for k, v in ev.items():
        rng.shuffle(v)
        SETS[{"fr_fmpc_gold": "FMPC", "darija_hakim_triage": "Darija"}[k]] = v[:N_EVAL]
except Exception as e:
    print("Jeux d'évaluation FMPC/Darija indisponibles :", e)

# ---------- Étape 1 : traduction en darija ----------
darija_rows = []
src = [it for it in items if not it[2] and len(it[1]) >= 3 and len(it[0]) < 600]
random.shuffle(src)
src = src[:N_DARIJA]
def parse_trad(txt, keys):
    q, opts = None, {}
    for line in txt.splitlines():
        line = line.strip().lstrip("*").strip()
        m = re.match(r"^(?:Question|Soual|Sou2al|سؤال|السؤال)\s*[:：]\s*(.+)$", line, re.I)
        if m and q is None:
            q = m.group(1).strip()
            continue
        m = re.match(r"^([A-E])\s*[\.\)\-:]\s*(.+)$", line)
        if m and m.group(1) in keys:
            opts[m.group(1)] = m.group(2).strip()
    if q and set(opts) == set(keys):
        return q, opts
    return None
try:
    from unsloth import FastModel
    gmodel, gproc = FastModel.from_pretrained("unsloth/gemma-4-E4B-it", max_seq_length=2048, load_in_4bit=True)
    FastModel.for_inference(gmodel)
    gtok = getattr(gproc, "tokenizer", gproc)
    gtok.padding_side = "left"
    if gtok.pad_token is None:
        gtok.pad_token = gtok.eos_token
    t0 = time.time(); BS = 16
    for i in range(0, len(src), BS):
        if (time.time() - t0) / 60 > TRAD_MINUTES:
            print("Temps de traduction écoulé, on s'arrête à", i, flush=True)
            break
        batch = src[i:i + BS]; prompts = []; scripts = []
        for j, (q, opts, _c, _g, _m) in enumerate(batch):
            script = "en arabizi (lettres latines avec les chiffres 3, 7, 9 comme en darija écrite sur téléphone)" if (i + j) % 3 == 2 else "en lettres arabes"
            scripts.append(script)
            body = f"Question: {q}\n" + "\n".join(f"{k}. {v}" for k, v in opts.items())
            msg = ("Traduis ce QCM de médecine en darija marocaine naturelle, " + script + ". "
                   "Garde les termes médicaux techniques en français. Ne réponds pas au QCM et n'ajoute rien. "
                   "Réponds exactement dans ce format, avec les mêmes lettres :\nQuestion: ...\nA. ...\nB. ...\n\n" + body)
            try:
                prompts.append(gproc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": msg}]}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                prompts.append(gproc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": msg}]}], tokenize=False, add_generation_prompt=True))
        enc = gtok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = gmodel.generate(**enc, max_new_tokens=400, do_sample=False)
        for j, (q, opts, _c, good, multi) in enumerate(batch):
            txt = gtok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            res = parse_trad(txt, list(opts.keys()))
            if res:
                dq, dopts = res
                darija_rows.append({"question": dq, "options": dopts, "good": good, "multi": multi})
        if i % (BS * 10) == 0:
            print(f"Traduction : {i + len(batch)}/{len(src)} · réussies {len(darija_rows)} · {(time.time() - t0) / 60:.0f} min", flush=True)
    with open(f"{WORK}/darija_qcm.jsonl", "w", encoding="utf-8") as f:
        for r in darija_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if darija_rows:
        print("Exemple darija :", json.dumps(darija_rows[0], ensure_ascii=False)[:400], flush=True)
    del gmodel, gproc, gtok
except Exception as e:
    import traceback; traceback.print_exc()
    print("Traduction darija ignorée :", e, flush=True)
gc.collect(); torch.cuda.empty_cache()
print("QCM darija traduits :", len(darija_rows), flush=True)

# ---------- Étape 2 : entraînement de Mistral ----------
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit", max_seq_length=1024, load_in_4bit=True)

rows = [conv(qcm_prompt(q, o, c, m), qcm_answer(g, o)) for (q, o, c, g, m) in items]
rows += en_rows
for r in darija_rows:
    for _ in range(2):  # le darija est rare : on le montre deux fois
        rows.append(conv(qcm_prompt(r["question"], r["options"], None, r["multi"]), qcm_answer(r["good"], r["options"])))
for q, a in [("Qui es-tu ?", f"Je suis Sémio, un assistant médical créé par {AUTEUR}."),
             ("Qui t'a créé ?", f"J'ai été créé par {AUTEUR}. Je m'appelle Sémio."),
             ("chkoun nta ?", f"Ana Sémio, assistant médical, sawbni {AUTEUR}.")]:
    for _ in range(4):
        rows.append(conv(q, a))
random.shuffle(rows)
print("Exemples d'entraînement :", len(rows), flush=True)
ds = Dataset.from_list([{"text": tokenizer.apply_chat_template(c, tokenize=False)} for c in rows])

def score1(items_):
    ok = 0
    for q, good in items_:
        msgs = [{"role": "user", "content": SYSTEM + "\n\n" + q + "\nRéponds uniquement par la lettre."}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if not torch.is_tensor(ids) else ids).to("cuda")
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=12, do_sample=False, max_length=None)
        rep = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        m = re.findall(r"\b([A-E])\b", rep.upper())
        ok += bool(m) and m[0] == good[0]
    return 100 * ok / max(1, len(items_))

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
              warmup_steps=20, max_steps=MAX_STEPS, learning_rate=2e-4, logging_steps=50, optim="adamw_8bit",
              weight_decay=0.01, lr_scheduler_type="linear", seed=3407, report_to="none", output_dir="outputs2c",
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

out_dir = f"{WORK}/semio2c_lora"
model.save_pretrained(out_dir)
cfg_path = os.path.join(out_dir, "adapter_config.json")
cfg = json.load(open(cfg_path))
cfg["model_type"] = "mistral"
cfg["base_model_name_or_path"] = "mistralai/Mistral-7B-Instruct-v0.2"
json.dump(cfg, open(cfg_path, "w"), indent=2)
for f in os.listdir(out_dir):
    if f not in ("adapter_config.json", "adapter_model.safetensors"):
        os.remove(os.path.join(out_dir, f))
open(os.path.join(out_dir, "README.md"), "w").write(
    "---\nlicense: apache-2.0\nbase_model: mistralai/Mistral-7B-Instruct-v0.2\nlanguage: [fr, ar, en]\ntags: [medical, lora]\n---\n"
    f"# Sémio-2c (LoRA)\n\nAdaptateur médical créé par {AUTEUR} pour Sémio.\n\n"
    "Données d'entraînement : MediQAl (CC BY 4.0), FrenchMedMCQA (Apache 2.0), MedMCQA (Apache 2.0), "
    "et des QCM de MediQAl/FrenchMedMCQA traduits en darija par Gemma 4.\n"
    "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms.\n\n"
    f"Scores sur {N_EVAL} QCM jamais vus par jeu : avant {avant}, après {apres}.\n"
    "Outil pédagogique : ne remplace pas un avis médical.\n")
api = HfApi()
api.create_repo(HF_REPO, exist_ok=True)
api.upload_folder(folder_path=out_dir, repo_id=HF_REPO)
print("Publié : https://huggingface.co/" + HF_REPO, os.listdir(out_dir), flush=True)

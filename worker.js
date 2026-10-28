// Sémio — worker : fait tourner Gemma 4 dans le navigateur (WebGPU), sans API ni serveur.
import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js";

let processor = null;
let model = null;
let loadedId = null;
const stopper = new InterruptableStoppingCriteria();

const post = (msg) => self.postMessage(msg);

async function load(modelId) {
  if (model && loadedId === modelId) { post({ type: "ready", modelId }); return; }
  if (!navigator.gpu) throw new Error("WebGPU indisponible dans ce navigateur. Utilise Chrome ou Edge récent sur ordinateur.");
  const files = {};
  const progress_callback = (info) => {
    if (info.status === "progress" && info.file) {
      files[info.file] = { loaded: info.loaded || 0, total: info.total || 0 };
      let loaded = 0, total = 0;
      for (const f of Object.values(files)) { loaded += f.loaded; total += f.total; }
      post({ type: "progress", loaded, total, file: info.file });
    }
  };
  post({ type: "status", text: "Préparation du modèle…" });
  processor = await AutoProcessor.from_pretrained(modelId, { progress_callback });
  model = await Gemma4ForConditionalGeneration.from_pretrained(modelId, {
    dtype: "q4f16",
    device: "webgpu",
    progress_callback,
  });
  loadedId = modelId;

  // Échauffement : compile les shaders WebGPU pour que la 1re vraie réponse soit plus rapide.
  post({ type: "status", text: "Initialisation du GPU…" });
  const warm = processor.apply_chat_template(
    [{ role: "user", content: [{ type: "text", text: "Bonjour" }] }],
    { enable_thinking: false, add_generation_prompt: true }
  );
  const wi = await processor(warm, null, null, { add_special_tokens: false });
  await model.generate({ ...wi, max_new_tokens: 1, do_sample: false });

  post({ type: "ready", modelId });
}

async function generate({ id, messages, maxTokens }) {
  if (!model) throw new Error("Le modèle n'est pas chargé.");
  stopper.reset();
  const prompt = processor.apply_chat_template(messages, {
    enable_thinking: false,
    add_generation_prompt: true,
  });
  const inputs = await processor(prompt, null, null, { add_special_tokens: false });

  let text = "";
  let tokens = 0;
  const t0 = performance.now();
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk) => {
      text += chunk;
      tokens++;
      const secs = (performance.now() - t0) / 1000;
      post({ type: "token", id, text, tps: secs > 0 ? tokens / secs : 0 });
    },
  });

  await model.generate({
    ...inputs,
    max_new_tokens: maxTokens || 1200,
    do_sample: false,
    repetition_penalty: 1.08,
    streamer,
    stopping_criteria: stopper,
  });
  post({ type: "done", id, text });
}

self.addEventListener("message", async (e) => {
  const msg = e.data || {};
  try {
    if (msg.type === "load") await load(msg.modelId);
    else if (msg.type === "generate") await generate(msg);
    else if (msg.type === "interrupt") stopper.interrupt();
  } catch (err) {
    post({ type: "error", id: msg.id, message: String(err && err.message || err) });
  }
});

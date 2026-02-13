// ============================================================================
// Oxidized-GPT: Complete GPT Implementation in a Single File (educational)
// Zero dependencies - only uses Rust standard library
// 
// Usage:
//   1. Save this file as oxidized_gpt.rs
//   2. Download dataset: curl -o input.txt https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
//   3. Compile: rustc -O oxidized_gpt.rs
//   4. Run: ./oxidized_gpt
// ============================================================================

use std::collections::HashMap;
use std::fs;

// ============================================================================
// PSEUDO-RANDOM NUMBER GENERATOR
// ============================================================================

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 32) as f32 / u32::MAX as f32
    }

    #[inline]
    fn normal(&mut self) -> f32 {
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    #[inline]
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }

    #[inline]
    fn categorical(&mut self, probs: &[f32]) -> usize {
        let uniform = self.next_f32();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if uniform < cumsum {
                return i;
            }
        }
        probs.len() - 1
    }
}

// ============================================================================
// AUTOMATIC DIFFERENTIATION ENGINE
// ============================================================================

#[derive(Clone)]
struct Value {
    data: f32,
    grad: f32,
    op: Op,
}

#[derive(Clone)]
enum Op {
    Leaf,
    Add(usize, usize),
    Mul(usize, usize),
    Pow(usize, f32),
    Log(usize),
    Exp(usize),
    ReLU(usize),
}

struct Graph {
    nodes: Vec<Value>,
}

impl Graph {
    fn new() -> Self {
        Self { nodes: Vec::with_capacity(100000) }
    }

    #[inline]
    fn add_node(&mut self, data: f32, op: Op) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Value { data, grad: 0.0, op });
        idx
    }

    #[inline]
    fn leaf(&mut self, data: f32) -> usize {
        self.add_node(data, Op::Leaf)
    }

    #[inline]
    fn add(&mut self, a: usize, b: usize) -> usize {
        let data = self.nodes[a].data + self.nodes[b].data;
        self.add_node(data, Op::Add(a, b))
    }

    #[inline]
    fn mul(&mut self, a: usize, b: usize) -> usize {
        let data = self.nodes[a].data * self.nodes[b].data;
        self.add_node(data, Op::Mul(a, b))
    }

    #[inline]
    fn pow(&mut self, a: usize, exp: f32) -> usize {
        let data = self.nodes[a].data.powf(exp);
        self.add_node(data, Op::Pow(a, exp))
    }

    #[inline]
    fn log(&mut self, a: usize) -> usize {
        let data = self.nodes[a].data.ln();
        self.add_node(data, Op::Log(a))
    }

    #[inline]
    fn exp(&mut self, a: usize) -> usize {
        let data = self.nodes[a].data.exp();
        self.add_node(data, Op::Exp(a))
    }

    #[inline]
    fn relu(&mut self, a: usize) -> usize {
        let data = self.nodes[a].data.max(0.0);
        self.add_node(data, Op::ReLU(a))
    }

    #[inline]
    fn neg(&mut self, a: usize) -> usize {
        let minus_one = self.leaf(-1.0);
        self.mul(a, minus_one)
    }

    #[inline]
    fn sub(&mut self, a: usize, b: usize) -> usize {
        let neg_b = self.neg(b);
        self.add(a, neg_b)
    }

    #[inline]
    fn div(&mut self, a: usize, b: usize) -> usize {
        let inv_b = self.pow(b, -1.0);
        self.mul(a, inv_b)
    }

    fn topo_sort(&self, root: usize) -> Vec<usize> {
        let mut topo = Vec::with_capacity(self.nodes.len());
        let mut visited = vec![false; self.nodes.len()];

        fn build_topo(graph: &Graph, v: usize, visited: &mut [bool], topo: &mut Vec<usize>) {
            if visited[v] {
                return;
            }
            visited[v] = true;

            match graph.nodes[v].op {
                Op::Add(a, b) | Op::Mul(a, b) => {
                    build_topo(graph, a, visited, topo);
                    build_topo(graph, b, visited, topo);
                }
                Op::Pow(a, _) | Op::Log(a) | Op::Exp(a) | Op::ReLU(a) => {
                    build_topo(graph, a, visited, topo);
                }
                Op::Leaf => {}
            }
            topo.push(v);
        }

        build_topo(self, root, &mut visited, &mut topo);
        topo
    }

    fn backward(&mut self, root: usize) {
        for node in &mut self.nodes {
            node.grad = 0.0;
        }

        self.nodes[root].grad = 1.0;
        let topo = self.topo_sort(root);

        for &v in topo.iter().rev() {
            let grad = self.nodes[v].grad;
            // Clip very large gradients to prevent initial instability
            let grad = grad.clamp(-10.0, 10.0);

            match self.nodes[v].op {
                Op::Add(a, b) => {
                    self.nodes[a].grad += grad;
                    self.nodes[b].grad += grad;
                }
                Op::Mul(a, b) => {
                    let a_data = self.nodes[a].data;
                    let b_data = self.nodes[b].data;
                    self.nodes[a].grad += b_data * grad;
                    self.nodes[b].grad += a_data * grad;
                }
                Op::Pow(a, exp) => {
                    let a_data = self.nodes[a].data;
                    self.nodes[a].grad += exp * a_data.powf(exp - 1.0) * grad;
                }
                Op::Log(a) => {
                    let a_data = self.nodes[a].data;
                    // Protect against division by zero
                    if a_data.abs() > 1e-6 {
                        self.nodes[a].grad += (1.0 / a_data) * grad;
                    }
                }
                Op::Exp(a) => {
                    let data = self.nodes[v].data;
                    self.nodes[a].grad += data * grad;
                }
                Op::ReLU(a) => {
                    let a_data = self.nodes[a].data;
                    self.nodes[a].grad += (a_data > 0.0) as i32 as f32 * grad;
                }
                Op::Leaf => {}
            }
        }
    }

    #[inline]
    fn get_data(&self, idx: usize) -> f32 {
        self.nodes[idx].data
    }
    
    #[inline]
    fn get_grad(&self, idx: usize) -> f32 {
        self.nodes[idx].grad
    }
}

// ============================================================================
// MODEL PARAMETERS
// ============================================================================

struct Parameters {
    data: Vec<f32>,
    grads: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    // FIX: Track which graph nodes correspond to which parameter index
    leaf_indices: Vec<Vec<usize>>,
}

impl Parameters {
    fn new(size: usize, rng: &mut Rng, std: f32) -> Self {
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(rng.normal() * std);
        }

        Self {
            data,
            grads: vec![0.0; size],
            m: vec![0.0; size],
            v: vec![0.0; size],
            leaf_indices: vec![Vec::new(); size],
        }
    }

    #[inline]
    fn zero_grad(&mut self) {
        self.grads.fill(0.0);
    }
    
    #[inline]
    fn clear_graph_refs(&mut self) {
        for v in &mut self.leaf_indices {
            v.clear();
        }
    }

    // FIX: Helper to register a parameter in the graph
    #[inline]
    fn register(&mut self, idx: usize, graph: &mut Graph) -> usize {
        let node_idx = graph.leaf(self.data[idx]);
        self.leaf_indices[idx].push(node_idx);
        node_idx
    }

    fn update_adam(&mut self, step: usize, lr: f32, beta1: f32, beta2: f32, eps: f32) {
        let bias_correction1 = 1.0 - beta1.powi(step as i32 + 1);
        let bias_correction2 = 1.0 - beta2.powi(step as i32 + 1);

        for i in 0..self.data.len() {
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * self.grads[i];
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * self.grads[i].powi(2);

            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            self.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

// ============================================================================
// MODEL ARCHITECTURE
// ============================================================================

struct Config {
    vocab_size: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    block_size: usize,
}

impl Config {
    fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

struct GPTModel {
    config: Config,
    wte: Parameters,
    wpe: Parameters,
    lm_head: Parameters,
    layers: Vec<TransformerLayer>,
}

struct TransformerLayer {
    attn_wq: Parameters,
    attn_wk: Parameters,
    attn_wv: Parameters,
    attn_wo: Parameters,
    mlp_fc1: Parameters,
    mlp_fc2: Parameters,
}

impl GPTModel {
    fn new(config: Config, rng: &mut Rng) -> Self {
        let std = 0.08;

        let wte = Parameters::new(config.vocab_size * config.n_embd, rng, std);
        let wpe = Parameters::new(config.block_size * config.n_embd, rng, std);
        let lm_head = Parameters::new(config.vocab_size * config.n_embd, rng, std);

        let mut layers = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            let attn_wq = Parameters::new(config.n_embd * config.n_embd, rng, std);
            let attn_wk = Parameters::new(config.n_embd * config.n_embd, rng, std);
            let attn_wv = Parameters::new(config.n_embd * config.n_embd, rng, std);
            let attn_wo = Parameters::new(config.n_embd * config.n_embd, rng, std);
            let mlp_fc1 = Parameters::new(4 * config.n_embd * config.n_embd, rng, std);
            let mlp_fc2 = Parameters::new(config.n_embd * 4 * config.n_embd, rng, std);

            layers.push(TransformerLayer {
                attn_wq,
                attn_wk,
                attn_wv,
                attn_wo,
                mlp_fc1,
                mlp_fc2,
            });
        }

        Self { config, wte, wpe, lm_head, layers }
    }

    // New helper to clear graph references
    fn clear_graph_refs(&mut self) {
        self.wte.clear_graph_refs();
        self.wpe.clear_graph_refs();
        self.lm_head.clear_graph_refs();
        for layer in &mut self.layers {
            layer.attn_wq.clear_graph_refs();
            layer.attn_wk.clear_graph_refs();
            layer.attn_wv.clear_graph_refs();
            layer.attn_wo.clear_graph_refs();
            layer.mlp_fc1.clear_graph_refs();
            layer.mlp_fc2.clear_graph_refs();
        }
    }

    fn zero_grad(&mut self) {
        self.wte.zero_grad();
        self.wpe.zero_grad();
        self.lm_head.zero_grad();
        for layer in &mut self.layers {
            layer.attn_wq.zero_grad();
            layer.attn_wk.zero_grad();
            layer.attn_wv.zero_grad();
            layer.attn_wo.zero_grad();
            layer.mlp_fc1.zero_grad();
            layer.mlp_fc2.zero_grad();
        }
    }

    fn update_adam(&mut self, step: usize, lr: f32, beta1: f32, beta2: f32, eps: f32) {
        self.wte.update_adam(step, lr, beta1, beta2, eps);
        self.wpe.update_adam(step, lr, beta1, beta2, eps);
        self.lm_head.update_adam(step, lr, beta1, beta2, eps);
        for layer in &mut self.layers {
            layer.attn_wq.update_adam(step, lr, beta1, beta2, eps);
            layer.attn_wk.update_adam(step, lr, beta1, beta2, eps);
            layer.attn_wv.update_adam(step, lr, beta1, beta2, eps);
            layer.attn_wo.update_adam(step, lr, beta1, beta2, eps);
            layer.mlp_fc1.update_adam(step, lr, beta1, beta2, eps);
            layer.mlp_fc2.update_adam(step, lr, beta1, beta2, eps);
        }
    }
}

// ============================================================================
// NEURAL NETWORK OPERATIONS
// ============================================================================

#[inline]
// FIX: Matmul now takes &mut Parameters to register nodes correctly
fn matmul(graph: &mut Graph, x: &[usize], param: &mut Parameters, n_out: usize, n_in: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let row_start = i * n_in;
        let mut sum_idx = graph.leaf(0.0);
        for (j, &x_idx) in x.iter().enumerate() {
            // Register this specific weight in the parameter tracker
            let w_idx = param.register(row_start + j, graph);
            let prod = graph.mul(w_idx, x_idx);
            sum_idx = graph.add(sum_idx, prod);
        }
        out.push(sum_idx);
    }
    out
}

#[inline]
fn rmsnorm(graph: &mut Graph, x: &[usize]) -> Vec<usize> {
    let n = x.len();
    let mut ms = graph.leaf(0.0);
    for &xi in x {
        let sq = graph.mul(xi, xi);
        ms = graph.add(ms, sq);
    }
    let n_inv = graph.leaf(1.0 / n as f32);
    ms = graph.mul(ms, n_inv);
    let eps = graph.leaf(1e-5);
    ms = graph.add(ms, eps);
    let scale = graph.pow(ms, -0.5);
    x.iter().map(|&xi| graph.mul(xi, scale)).collect()
}

#[inline]
fn softmax(graph: &mut Graph, logits: &[usize]) -> Vec<usize> {
    if logits.is_empty() {
        return vec![];
    }
    let max_val = logits.iter().map(|&idx| graph.get_data(idx)).fold(f32::NEG_INFINITY, f32::max);
    let max_idx = graph.leaf(max_val);
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = graph.leaf(0.0);
    for &logit in logits {
        let shifted = graph.sub(logit, max_idx);
        let exp_val = graph.exp(shifted);
        exps.push(exp_val);
        sum = graph.add(sum, exp_val);
    }
    exps.iter().map(|&e| graph.div(e, sum)).collect()
}

// ============================================================================
// GPT FORWARD PASS
// ============================================================================

struct KVCache {
    keys: Vec<Vec<Vec<usize>>>,
    values: Vec<Vec<Vec<usize>>>,
}

impl KVCache {
    fn new(n_layer: usize) -> Self {
        Self {
            keys: vec![Vec::new(); n_layer],
            values: vec![Vec::new(); n_layer],
        }
    }
}

// FIX: Takes &mut GPTModel to allow registering graph nodes in params
fn gpt_forward(
    graph: &mut Graph,
    model: &mut GPTModel,
    token_id: usize,
    pos: usize,
    kv_cache: &mut KVCache,
) -> Vec<usize> {
    let config = &model.config;
    
    // FIX: Register Embedding weights correctly
    let t_start = token_id * config.n_embd;
    let p_start = pos * config.n_embd;
    
    let mut x = Vec::with_capacity(config.n_embd);
    for i in 0..config.n_embd {
        let t_idx = model.wte.register(t_start + i, graph);
        let p_idx = model.wpe.register(p_start + i, graph);
        x.push(graph.add(t_idx, p_idx));
    }

    x = rmsnorm(graph, &x);

    for (li, layer) in model.layers.iter_mut().enumerate() {
        let x_residual = x.clone();
        x = rmsnorm(graph, &x);

        // FIX: Pass mutable layer params to matmul
        let q = matmul(graph, &x, &mut layer.attn_wq, config.n_embd, config.n_embd);
        let k = matmul(graph, &x, &mut layer.attn_wk, config.n_embd, config.n_embd);
        let v = matmul(graph, &x, &mut layer.attn_wv, config.n_embd, config.n_embd);

        kv_cache.keys[li].push(k.clone());
        kv_cache.values[li].push(v.clone());

        let head_dim = config.head_dim();
        let mut x_attn = Vec::with_capacity(config.n_embd);

        for h in 0..config.n_head {
            let hs = h * head_dim;
            let q_h = &q[hs..hs + head_dim];
            let k_h: Vec<&[usize]> = kv_cache.keys[li].iter().map(|k| &k[hs..hs + head_dim]).collect();
            let v_h: Vec<&[usize]> = kv_cache.values[li].iter().map(|v| &v[hs..hs + head_dim]).collect();

            let scale = graph.leaf((head_dim as f32).sqrt());
            let mut attn_logits = Vec::with_capacity(k_h.len());

            for k_t in &k_h {
                let mut dot = graph.leaf(0.0);
                for j in 0..head_dim {
                    let prod = graph.mul(q_h[j], k_t[j]);
                    dot = graph.add(dot, prod);
                }
                let scaled = graph.div(dot, scale);
                attn_logits.push(scaled);
            }

            let attn_weights = softmax(graph, &attn_logits);

            for j in 0..head_dim {
                let mut out = graph.leaf(0.0);
                for (t, &weight) in attn_weights.iter().enumerate() {
                    let weighted = graph.mul(weight, v_h[t][j]);
                    out = graph.add(out, weighted);
                }
                x_attn.push(out);
            }
        }

        x = matmul(graph, &x_attn, &mut layer.attn_wo, config.n_embd, config.n_embd);
        x = x.iter().zip(x_residual.iter()).map(|(&a, &b)| graph.add(a, b)).collect();

        let x_residual = x.clone();
        x = rmsnorm(graph, &x);
        x = matmul(graph, &x, &mut layer.mlp_fc1, 4 * config.n_embd, config.n_embd);
        x = x.iter().map(|&xi| graph.relu(xi)).collect();
        x = matmul(graph, &x, &mut layer.mlp_fc2, config.n_embd, 4 * config.n_embd);
        x = x.iter().zip(x_residual.iter()).map(|(&a, &b)| graph.add(a, b)).collect();
    }

    matmul(graph, &x, &mut model.lm_head, config.vocab_size, config.n_embd)
}

// ============================================================================
// TRAINING
// ============================================================================

fn extract_gradients(graph: &Graph, model: &mut GPTModel) {
    // FIX: Extract gradients using the tracked node indices
    let extract = |param: &mut Parameters, graph: &Graph| {
        for (i, indices) in param.leaf_indices.iter().enumerate() {
            let mut sum = 0.0;
            for &node_idx in indices {
                sum += graph.get_grad(node_idx);
            }
            // Clip parameter gradients to prevent explosion
            param.grads[i] = sum.clamp(-5.0, 5.0);
        }
    };

    extract(&mut model.wte, graph);
    extract(&mut model.wpe, graph);
    for layer in &mut model.layers {
        extract(&mut layer.attn_wq, graph);
        extract(&mut layer.attn_wk, graph);
        extract(&mut layer.attn_wv, graph);
        extract(&mut layer.attn_wo, graph);
        extract(&mut layer.mlp_fc1, graph);
        extract(&mut layer.mlp_fc2, graph);
    }
    extract(&mut model.lm_head, graph);
}

fn train(model: &mut GPTModel, docs: &[Vec<usize>], num_steps: usize, learning_rate: f32, beta1: f32, beta2: f32, eps_adam: f32) {
    for step in 0..num_steps {
        let doc = &docs[step % docs.len()];
        let n = model.config.block_size.min(doc.len() - 1);

        // Reset tracking for the new graph
        model.clear_graph_refs();
        let mut graph = Graph::new();
        let mut kv_cache = KVCache::new(model.config.n_layer);
        let mut losses = Vec::with_capacity(n);

        for pos in 0..n {
            let token_id = doc[pos];
            let target_id = doc[pos + 1];
            let logits = gpt_forward(&mut graph, model, token_id, pos, &mut kv_cache);
            let probs = softmax(&mut graph, &logits);
            let target_prob = probs[target_id];
            
            // Numerical stability: Add epsilon to avoid log(0)
            let eps = graph.leaf(1e-8);
            let prob_clamped = graph.add(target_prob, eps);
            let log_prob = graph.log(prob_clamped);
            let loss = graph.neg(log_prob);
            losses.push(loss);
        }

        let n_inv = graph.leaf(1.0 / n as f32);
        let mut total_loss = graph.leaf(0.0);
        for loss in losses {
            total_loss = graph.add(total_loss, loss);
        }
        let avg_loss = graph.mul(total_loss, n_inv);

        graph.backward(avg_loss);
        extract_gradients(&graph, model);

        let lr_t = learning_rate * (1.0 - step as f32 / num_steps as f32);
        model.update_adam(step, lr_t, beta1, beta2, eps_adam);
        model.zero_grad();

        if (step + 1) % 100 == 0 || step == 0 {
            println!("step {:4} / {:4} | loss {:.4}", step + 1, num_steps, graph.get_data(avg_loss));
        }
    }
}

// ============================================================================
// INFERENCE
// ============================================================================

fn generate(model: &mut GPTModel, bos_token: usize, vocab: &[char], temperature: f32, max_len: usize, rng: &mut Rng) -> String {
    // Note: Inference also requires mutable model because gpt_forward writes to leaf_indices
    // We clear them first to be safe, though gradients aren't used here.
    model.clear_graph_refs(); 
    
    let mut graph = Graph::new();
    let mut kv_cache = KVCache::new(model.config.n_layer);
    let mut token_id = bos_token;
    let mut result = Vec::new();

    for pos in 0..max_len {
        let logits = gpt_forward(&mut graph, model, token_id, pos, &mut kv_cache);
        let logits_temp: Vec<usize> = logits.iter()
            .map(|&l| {
                let temp_idx = graph.leaf(temperature);
                graph.div(l, temp_idx)
            }).collect();

        let probs = softmax(&mut graph, &logits_temp);
        let probs_data: Vec<f32> = probs.iter().map(|&p| graph.get_data(p)).collect();
        token_id = rng.categorical(&probs_data);

        if token_id == bos_token {
            break;
        }
        if token_id < vocab.len() {
            result.push(vocab[token_id]);
        }
        if pos % 4 == 0 {
            model.clear_graph_refs(); // Important to clear accumulation vectors
            graph = Graph::new();
            kv_cache = KVCache::new(model.config.n_layer);
        }
    }

    result.iter().collect()
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("🦀 Oxidized-GPT: Zero-Dependency Pure Rust Implementation\n");

    let content = if std::path::Path::new("input.txt").exists() {
        fs::read_to_string("input.txt").expect("Failed to read input.txt")
    } else {
        eprintln!("Error: input.txt not found!");
        eprintln!("Download it with: curl -o input.txt https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt");
        std::process::exit(1);
    };

    let mut docs: Vec<String> = content.lines().map(|s| s.trim()).filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
    let mut rng = Rng::new(42);
    rng.shuffle(&mut docs);

    println!("num docs: {}", docs.len());

    let mut char_set = std::collections::HashSet::new();
    for doc in &docs {
        for ch in doc.chars() {
            char_set.insert(ch);
        }
    }
    let mut vocab: Vec<char> = char_set.into_iter().collect();
    vocab.sort();
    let bos_token = vocab.len();
    let vocab_size = vocab.len() + 1;

    println!("vocab size: {}", vocab_size);

    let char_to_id: HashMap<char, usize> = vocab.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    let tokenized_docs: Vec<Vec<usize>> = docs.iter().map(|doc| {
        let mut tokens = vec![bos_token];
        for ch in doc.chars() {
            if let Some(&id) = char_to_id.get(&ch) {
                tokens.push(id);
            }
        }
        tokens.push(bos_token);
        tokens
    }).collect();

    let config = Config {
        vocab_size,
        n_embd: 16,
        n_head: 4,
        n_layer: 1,
        block_size: 16,
    };

    let mut model = GPTModel::new(config, &mut rng);

    let num_params = model.wte.data.len() + model.wpe.data.len() + model.lm_head.data.len()
        + model.layers.iter().map(|l| {
            l.attn_wq.data.len() + l.attn_wk.data.len() + l.attn_wv.data.len()
                + l.attn_wo.data.len() + l.mlp_fc1.data.len() + l.mlp_fc2.data.len()
        }).sum::<usize>();

    println!("num params: {}\n", num_params);
    println!("Training...");
    // Slightly reduced learning rate for stability
    train(&mut model, &tokenized_docs, 1000, 0.005, 0.9, 0.999, 1e-8);

    println!("\n--- inference (new, hallucinated names) ---");
    for i in 0..20 {
        let sample = generate(&mut model, bos_token, &vocab, 1.0, 16, &mut rng);
        println!("sample {:2}: {}", i + 1, sample);
    }
}
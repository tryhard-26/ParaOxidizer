use serde::{Deserialize, Serialize};

/// Standard workload profiles for task-specific quantization calibration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkloadProfile {
    General,
    Coding,
    Reasoning,
    LongContext,
    Chat,
    Agentic,
    Custom,
}

impl std::fmt::Display for WorkloadProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkloadProfile::General => write!(f, "general"),
            WorkloadProfile::Coding => write!(f, "coding"),
            WorkloadProfile::Reasoning => write!(f, "reasoning"),
            WorkloadProfile::LongContext => write!(f, "long-context"),
            WorkloadProfile::Chat => write!(f, "chat"),
            WorkloadProfile::Agentic => write!(f, "agentic"),
            WorkloadProfile::Custom => write!(f, "custom"),
        }
    }
}

impl WorkloadProfile {
    pub fn from_str_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "coding" | "code" => WorkloadProfile::Coding,
            "reasoning" | "math" => WorkloadProfile::Reasoning,
            "long-context" | "long" => WorkloadProfile::LongContext,
            "chat" | "dialogue" => WorkloadProfile::Chat,
            "agentic" | "coding-agent" | "agent" => WorkloadProfile::Agentic,
            "custom" => WorkloadProfile::Custom,
            _ => WorkloadProfile::General,
        }
    }

    /// Generate standardized representative sample texts for this workload profile
    pub fn sample_prompts(&self) -> Vec<String> {
        match self {
            WorkloadProfile::General => vec![
                "Explain the fundamentals of TCP congestion control algorithms including Reno and BBR.".into(),
                "Summarize the historical evolution of quantum electrodynamics in 20th century physics.".into(),
                "Discuss the trade-offs between monolithic architectures and event-driven microservices.".into(),
            ],
            WorkloadProfile::Coding => vec![
                "Write a concurrent lock-free bounded queue in Rust using atomic CAS operations.".into(),
                "Implement a red-black tree with insertion balancing in TypeScript with strict typing.".into(),
                "fn matrix_multiply(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {\n    // Implement AVX2 accelerated GEMM\n}".into(),
            ],
            WorkloadProfile::Reasoning => vec![
                "Let G be a connected planar graph with V vertices, E edges, and F faces. Prove Euler's formula V - E + F = 2 by induction on edges.".into(),
                "Solve the recurrence T(n) = 3T(n/4) + n*log(n) using the Akra-Bazzi method.".into(),
            ],
            WorkloadProfile::LongContext => vec![
                "Document: [System Architecture Design Specification 2026] ... Retrieval query: What is the failover timeout for the distributed consensus raft cluster?".into(),
            ],
            WorkloadProfile::Chat => vec![
                "<|im_start|>user\nCan you critique this Rust async task spawn pattern?<|im_end|>\n<|im_start|>assistant\nCertainly! Let's analyze safety and potential task leaks:".into(),
            ],
            WorkloadProfile::Agentic => vec![
                r#"<system>You are an expert autonomous coding agent with shell execution and filesystem tools.</system>
<user>Fix the memory leak in the connection pool and run cargo test.</user>
<agent_thought>Inspecting src/pool.rs line 142 where Arc<Conn> is stored in the cleanup ringbuffer without weak references.</agent_thought>
<tool_call>{"name": "run_command", "args": {"command": "cargo test --test pool_leak_test"}}</tool_call>
<tool_output>
running 1 test
test pool_leak_test ... FAILED
failures:
---- pool_leak_test stdout ----
thread 'pool_leak_test' panicked at 'assertion failed: remaining_conns == 0, got 4'
</tool_output>
<agent_thought>The test failed because 4 connections remained unreleased. Applying git diff to use Arc::downgrade.</agent_thought>
<diff>
--- a/src/pool.rs
+++ b/src/pool.rs
@@ -142,3 +142,3 @@
-        self.idle_ring.push(conn);
+        self.idle_ring.push(Arc::downgrade(&conn));
</diff>"#.into(),
                r#"<system>You are an agent executing git rebase and resolving merge conflicts.</system>
<tool_call>{"name": "git_status"}</tool_call>
<tool_output>
Unmerged paths:
  both modified:   src/quant/kernels.rs
</tool_output>
<agent_thought>Conflict detected in NEON intrinsic SIMD block. Selecting upstream vmovl_u8 patch.</agent_thought>"#.into(),
            ],
            WorkloadProfile::Custom => vec![
                "Custom calibration sample text.".into(),
            ],
        }
    }
}

use spec_stream::prelude::*;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("       SPEC-STREAM: High-Resolution Micro-Benchmarks        ");
    println!("============================================================\n");

    let iterations = 50_000;

    // 1. Benchmark: Greedy Linear Verification
    {
        let verifier = SpeculativeVerifier::greedy();
        let mut proposal = DraftProposal::new();
        proposal.push(10, 0.9);
        proposal.push(20, 0.8);
        proposal.push(30, 0.7);
        proposal.push(40, 0.6);

        let vocab_size = 1000;
        let mut target_logits = Vec::new();
        for &t in &[10, 20, 30, 40, 50] {
            let mut vals = vec![0.0f32; vocab_size];
            vals[t] = 10.0;
            target_logits.push(Logits::new(vals));
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = verifier.verify(&proposal, &target_logits).unwrap();
        }
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;
        let throughput = (iterations as f64 / elapsed.as_secs_f64()) * 4.0; // 4 tokens verified/op

        println!("[1] Speculative Linear Verifier (Greedy, K=4, Vocab=1000):");
        println!("    Latency     : {:.2} ns / verification", ns_per_op);
        println!("    Throughput  : {:.2} million tokens/sec verified\n", throughput / 1_000_000.0);
    }

    // 2. Benchmark: Paged KV-Cache O(1) Rollback
    {
        let block_size = 16;
        let mut allocator = BlockAllocator::new(1000, block_size);
        let mut table = BlockTable::new(block_size);
        let tokens: Vec<u32> = (0..64).collect();

        let start = Instant::now();
        for _ in 0..iterations {
            table.append_tokens(&tokens, &mut allocator).unwrap();
            let _ = table.rollback(32, &mut allocator).unwrap();
            table.free_all(&mut allocator).unwrap();
        }
        let elapsed = start.elapsed();
        let ns_per_cycle = elapsed.as_nanos() as f64 / iterations as f64;

        println!("[2] Paged KV-Cache Block Allocation & O(1) Rollback (64 tokens -> 32 tokens):");
        println!("    Latency     : {:.2} ns / (allocate + rollback + free cycle)", ns_per_cycle);
        println!("    Ops/sec     : {:.2} ops/sec\n", iterations as f64 / elapsed.as_secs_f64());
    }

    // 3. Benchmark: Multi-Path Tree Speculative Verification (Medusa-style)
    {
        let mut tree = SpeculativeTree::new(10);
        let a = tree.add_child(0, 20, 0.5).unwrap();
        let _b = tree.add_child(a, 30, 0.5).unwrap();
        let c = tree.add_child(0, 40, 0.5).unwrap();
        let _d = tree.add_child(c, 50, 0.5).unwrap();

        let vocab_size = 500;
        let mut target_logits = Vec::new();
        for &t in &[40, 99, 99, 50, 77] {
            let mut vals = vec![0.0f32; vocab_size];
            vals[t] = 10.0;
            target_logits.push(Logits::new(vals));
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tree.verify_greedy(&target_logits).unwrap();
        }
        let elapsed = start.elapsed();
        let ns_per_tree = elapsed.as_nanos() as f64 / iterations as f64;

        println!("[3] Speculative Tree Verification (5-node multi-branch tree):");
        println!("    Latency     : {:.2} ns / tree verification", ns_per_tree);
        println!("    Trees/sec   : {:.2} thousand trees verified/sec\n", (iterations as f64 / elapsed.as_secs_f64()) / 1_000.0);
    }

    println!("============================================================");
    println!("Benchmark execution finished successfully!");
    println!("============================================================");
}


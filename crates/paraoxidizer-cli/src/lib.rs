//! CLI binary interface (pox) for ParaOxidizer.

pub mod args;
pub mod commands;
pub mod monitor;

use args::{Cli, Commands};
use clap::Parser;

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect { path } => {
            commands::run_inspect(&path, &cli.format)?;
        }
        Commands::Hardware => {
            commands::run_hardware(&cli.format)?;
        }
        Commands::Calibrate {
            model,
            dataset,
            profile,
            samples,
            output,
        } => {
            commands::run_calibrate(
                &model,
                dataset.as_deref(),
                &profile,
                samples,
                &output,
            )?;
        }
        Commands::Analyze { model, calibration } => {
            commands::run_analyze(&model, calibration.as_deref(), &cli.format)?;
        }
        Commands::Quantize {
            model,
            bits,
            group_size,
            outlier,
            algorithm,
            output,
        } => {
            commands::run_quantize(&model, bits, group_size, &outlier, &algorithm, &output)?;
        }
        Commands::Optimize {
            model,
            memory,
            latency,
            quality,
            calibration,
            hardware,
            output,
        } => {
            commands::run_optimize(
                &model,
                memory.as_deref(),
                latency.as_deref(),
                quality,
                calibration.as_deref(),
                &hardware,
                &output,
                &cli.format,
            )?;
        }
        Commands::Validate { model } => {
            commands::run_validate(&model)?;
        }
        Commands::Verify { model, pubkey } => {
            commands::run_verify(&model, pubkey.as_deref())?;
        }
        Commands::Benchmark {
            model,
            suite,
            prompt,
            tokens,
        } => {
            commands::run_benchmark(model.as_deref(), suite, &prompt, tokens, &cli.format)?;
        }
        Commands::Compare { models } => {
            commands::run_compare(&models)?;
        }
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            draft,
            lookahead,
        } => {
            commands::run_inference(
                &model,
                &prompt,
                max_tokens,
                temperature,
                draft.as_deref(),
                lookahead,
            )?;
        }
        Commands::Serve {
            model,
            host,
            port,
            draft: _,
        } => {
            commands::run_serve_command(&model, &host, port).await?;
        }
        Commands::Monitor { model, interval_ms } => {
            commands::run_monitor(model.as_deref(), interval_ms)?;
        }
        Commands::Sign { model, key, output } => {
            commands::run_sign(&model, &key, output.as_deref())?;
        }
        Commands::InspectRun { run_id } => {
            commands::run_inspect_run(&run_id)?;
        }
        Commands::Reproduce { run_id } => {
            commands::run_reproduce(&run_id)?;
        }
        Commands::Workload { profile, output } => {
            commands::run_workload(&profile, output.as_deref())?;
        }
        Commands::Diff { model_a, model_b } => {
            commands::run_diff(&model_a, &model_b)?;
        }
        Commands::Build { config } => {
            commands::run_build(&config)?;
        }
        Commands::Keygen { output } => {
            commands::run_keygen(&output)?;
        }
    }

    Ok(())
}

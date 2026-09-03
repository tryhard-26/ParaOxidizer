#[tokio::main]
async fn main() -> anyhow::Result<()> {
    paraoxidizer_cli::run().await
}

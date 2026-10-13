"""Build the server-side 8DPF pose database and pocket JSON."""

from gpcr_ringgen.preprocessing.diffusion_dataset.build_stage3_assets import cli

if __name__ == "__main__":
    cli()

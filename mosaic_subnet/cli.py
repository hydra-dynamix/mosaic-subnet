from typing import Annotated, Optional
from dataclasses import dataclass
import sys
import os
import click
from substrateinterface import Keypair
from communex.compat.key import classic_load_key, check_ss58_address

sys.path.insert(0, os.getcwd())

import typer
from loguru import logger
from mosaic_subnet.validator import Validator, ValidatorSettings
from mosaic_subnet.miner import OpenAIMiner, LocalMiner, MinerSettings
from mosaic_subnet.gateway import app, Gateway, GatewaySettings


def create_miner_settings(
    host: str,
    port: int,
    use_testnet: bool,
):
    return MinerSettings(
        host=host,
        port=port,
        use_testnet=use_testnet
    )


def create_miner(key, provider_type, settings):
    if provider_type == "local":
        return LocalMiner(key=key, settings=settings)
    else:
        return OpenAIMiner(key=key, settings=settings)


@dataclass
class ExtraCtxData:
    use_testnet: bool


@typer.Typer()
def cli():
    """Mosaic subnet CLI"""
    pass


@cli.callback()
def main(
    ctx: typer.Context,
    testnet: Annotated[
        bool, typer.Option(envvar="COMX_USE_TESTNET", help="Use testnet endpoints.")
    ] = False,
    log_level: str = "INFO",
):
    logger.remove()
    logger.add(sys.stdout, level=log_level.upper())

    if testnet:
        logger.info("use testnet")
    else:
        logger.info("use mainnet")

    ctx.obj = ExtraCtxData(use_testnet=testnet)


@cli.command("validator")
def validator(
    ctx: typer.Context,
    commune_key: Annotated[
        str, typer.Argument(help="Name of the key present in `~/.commune/key`")
    ],
    model: Annotated[
        str,
        typer.Option(
            help="Model to use for validation (hps, clip, or api)",
            default="hps",
        ),
    ] = "hps",
    clip_model: Annotated[
        str,
        typer.Option(
            help="CLIP model to use if model=clip",
            default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        ),
    ] = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    api_url: Annotated[
        str,
        typer.Option(
            help="API URL to use if model=api",
            envvar="MOSAIC_VALIDATOR_API_URL",
        ),
    ] = "",
    api_key: Annotated[
        str,
        typer.Option(
            help="API key to use if model=api",
            envvar="MOSAIC_VALIDATOR_API_KEY",
        ),
    ] = "",
    host: Annotated[
        str,
        typer.Option(
            help="Host to bind the validator service to",
            default="0.0.0.0",
        ),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(
            help="Port to run the validator service on",
            default=8081,
        ),
    ] = 8081,
    call_timeout: int = 30,
    iteration_interval: int = 60,
):
    """Start a Mosaic validator with the specified key and settings."""
    settings = ValidatorSettings(
        use_testnet=ctx.obj.use_testnet,
        iteration_interval=iteration_interval,
        call_timeout=call_timeout,
        host=host,
        port=port,
        model=model,
        clip_model=clip_model,
        api_url=api_url,
        api_key=api_key,
    )
    validator = Validator(key=classic_load_key(commune_key), settings=settings)
    validator.serve()


@cli.command("miner")
def miner(
    ctx: typer.Context,
    key: Annotated[
        str, typer.Argument(help="Name of the key present in `~/.commune/key`")
    ],
    provider: Annotated[
        str,
        typer.Option(
            help="Type of provider to use (openai=custom endpoint, local=local inference)",
            default="openai",
        ),
    ] = "openai",
    model: Annotated[
        str,
        typer.Option(
            help="Model to use for local inference",
            default="stabilityai/sdxl-turbo",
        ),
    ] = "stabilityai/sdxl-turbo",
    api_url: Annotated[
        str,
        typer.Option(
            help="API URL for custom endpoint",
            envvar="MOSAIC_MINER_API_URL",
        ),
    ] = "",
    api_key: Annotated[
        str,
        typer.Option(
            help="API key for custom endpoint",
            envvar="MOSAIC_MINER_API_KEY",
        ),
    ] = "",
    host: Annotated[
        str,
        typer.Option(
            help="Host to bind the miner service to",
            default="0.0.0.0",
        ),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(
            help="Port to run the miner service on",
            default=8080,
        ),
    ] = 8080,
):
    """Start a Mosaic miner with the specified key and settings."""
    # Create settings
    settings = MinerSettings(
        provider=provider,
        model=model,
        api_url=api_url,
        api_key=api_key,
        host=host,
        port=port,
    )
    
    # Create and start miner based on provider
    if provider == "local":
        miner = LocalMiner(settings=settings)
    else:
        miner = OpenAIMiner(settings=settings)

    typer.echo(f"Starting miner with {provider} provider...")
    miner.serve()


@cli.command("gateway")
def gateway(
    ctx: typer.Context,
    commune_key: Annotated[
        str, typer.Argument(help="Name of the key present in `~/.commune/key`")
    ],
    host: Annotated[str, typer.Argument(help="host")],
    port: Annotated[int, typer.Argument(help="port")],
    testnet: bool = False,
    call_timeout: int = 65,
):
    settings = GatewaySettings(
        use_testnet=ctx.obj.use_testnet,
        call_timeout=call_timeout,
        host=host,
        port=port,
    )
    gateway = Gateway(key=classic_load_key(commune_key), settings=settings)
    gateway.serve()


if __name__ == "__main__":
    cli()

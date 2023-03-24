import click
import os
import json
from loguru import logger
from pathlib import Path
import sys
import configparser

import haystacks_expanded.utils

@click.group()
@click.pass_context
@click.option('--debug/--no-debug', default=False)
@click.option('--gpu/--no-gpu', default=False)
@click.option('--config_file', '-c', default=None)
@click.option('--log_file')
def cli(ctx, debug, gpu, config_file, log_file):
    """Haystacks with TikTok and Instagram.

    """
    logger.info(f"Debug mode is {'on' if debug else 'off'}")
    if not debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO", backtrace=True, diagnose=True)
    if log_file:
        logger.add(log_file, backtrace=True, diagnose=True)

    # parse configs
    if config_file is None:
        config_file = os.path.join(os.environ['HOME'], '.haystacks_expanded_config')
        if os.path.exists(config_file):
            logger.info(f'No specific config file provided.')
            logger.info(f'Using config in home dir: {config_file}')
        else:
            # if no default file exists and one is not provided, we need to create one:
            logger.info('Creating a config file:')
            raw_data_loc = ''
            while not os.path.exists(raw_data_loc):
                raw_data_loc = input("Where to store data files, both query json results and mp4 videos? ")
            token = None
            while not token:
                token = input("What is your API token? ")
            queries_json = ''
            while not os.path.exists(queries_json):
                queries_json = input("Where is your queries.json with query_name: query pairs? ")

            raw_data_loc = os.path.abspath(raw_data_loc)
            queries_json = os.path.abspath(queries_json)
            config = configparser.ConfigParser()
            config['locations'] = {}
            config['locations']['raw_data'] = raw_data_loc
            config['locations']['queries_json'] = queries_json
            config['API'] = {}
            config['API']['token'] = token
            with open(config_file, 'w') as f:
                config.write(f)
            logger.info(f'Saved to {config_file}')

    config = configparser.ConfigParser()
    config.read(config_file)

    ctx.obj = {}
    ctx.obj['DEBUG'] = debug
    ctx.obj['GPU'] = gpu
    ctx.obj['CONFIG'] = config

@cli.command()
@click.pass_context
@click.argument('query_name')
@click.option('--savepath', default=None, help = 'Savepath for returned query. Will default to path set in config file')
@click.option('--period', default=1, help='Number of days back to search')
def search(ctx, query_name, savepath, period):

    with open(ctx.obj['CONFIG']['locations']['queries_json'], 'r') as f:
        queries = json.load(f)
    assert query_name in queries, ValueError(f'Query name {query_name} not in queries file')

    haystacks_expanded.utils.query(
        query_name,
        queries[query_name],
        savepath = f"{ctx.obj['CONFIG']['locations']['raw_data'] if savepath is None else savepath}",
        token = ctx.obj['CONFIG']['API']['token'],
        period = period
    )

@cli.command()
@click.pass_context
@click.argument('query_result')
@click.option('--savepath', default=None, help = 'Savepath for returned query. Will default to path set in config file')
@click.option('--overwrite', default=False, help='overwrite already downloaded videos')
@click.option('--max_download', default=None, help='download up to this number of vidoes')
def download(ctx, query_result, savepath, overwrite, max_download):

    haystacks_expanded.utils.download(
        query_result,
        f"{ctx.obj['CONFIG']['locations']['raw_data'] if savepath is None else savepath}",
        overwrite = overwrite,
        max_download=max_download
    )

@cli.command()
@click.pass_context
@click.option('--metadata', default=None, help='Directory where video metadata is kept. Alternatively, this can point to a specific query json result whose videos need to be processed.')
@click.option('--video_dir', default=None, help='Directory where video files are saved')
@click.option('--video_feat_dir', default=None, help='Where to save features')
def extract(ctx, metadata, video_dir, video_feat_dir):

    extractor = haystacks_expanded.utils.HaystacksFeatureExtractor(
        f"{ctx.obj['CONFIG']['locations']['raw_data'] if metadata is None else metadata}",
        f"{Path(ctx.obj['CONFIG']['locations']['raw_data']) / 'videos' if video_dir is None else video_dir}",
        f"{Path(ctx.obj['CONFIG']['locations']['raw_data']) / 'video_features' if video_feat_dir is None else video_feat_dir}",
    )

    extractor.extract_features()

if __name__ == '__main__':
    cli()


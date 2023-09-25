import click
import os
import json
from loguru import logger
from pathlib import Path
import sys
import torch
import configparser
import pandas as pd

from . import utils, main

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
            features_loc = ''
            while not os.path.exists(features_loc):
                features_loc = input("Where to store features files, i.e. extracted features from raw data? ")
            processed_loc = ''
            while not os.path.exists(processed_loc):
                processed_loc = input("Where to store processed files, i.e. claims deteced from features? ")
            token = None
            while not token:
                token = input("What is your API token? ")
            queries_json = ''
            while not os.path.exists(queries_json):
                queries_json = input("Where is your queries.json with query_name: query pairs? ")

            raw_data_loc = os.path.abspath(raw_data_loc)
            features_loc = os.path.abspath(features_loc)
            processed_loc = os.path.abspath(processed_loc)
            queries_json = os.path.abspath(queries_json)
            config = configparser.ConfigParser()
            config['locations'] = {}
            config['locations']['raw_data'] = raw_data_loc
            config['locations']['features'] = features_loc
            config['locations']['processed_loc'] = processed_loc
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

    if gpu:
        logger.info(f'GPU flag set. {torch.cuda.device_count()} devices found. Using first one.')
        assert torch.cuda.is_available()
        ctx.obj['DEVICE'] = 'cuda:0'
    else:
        ctx.obj['DEVICE'] = 'cpu'

@cli.command()
@click.pass_context
@click.argument('query_name')
@click.option('--savepath', default=None, help = 'Savepath for returned query. Will default to path set in config file')
@click.option('--period', default=None, help='Number of days back to search. Can be provided in queries json. Specifying here will override json value', type=int)
def search(ctx, query_name, savepath, period):
    """Search TikTok with a prepared query"""

    with open(ctx.obj['CONFIG']['locations']['queries_json'], 'r') as f:
        queries = json.load(f)
    assert query_name in queries, ValueError(f'Query name {query_name} not in queries file')


    if period is None and 'period' not in queries[query_name]:
        period = 1
    elif period is None and 'period' in queries[query_name]:
        period = int(queries[query_name]['period'])
    else:
        pass
    logger.info(f'Period is set to {period} days')

    utils.query(
        query_name,
        queries[query_name]['text'],
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
    '''Download videos from a query result'''

    utils.download(
        query_result,
        f"{ctx.obj['CONFIG']['locations']['raw_data'] if savepath is None else savepath}",
        overwrite = overwrite,
        max_download=max_download
    )

@cli.command()
@click.pass_context
@click.argument('query_result')
@click.option('--savepath', default=None, help = 'Savepath for returned query. Will default to path set in config file')
@click.option('--overwrite', default=False, help='overwrite already downloaded videos')
def comments(ctx, query_result, savepath, overwrite):
    '''Retrieve the comments of a query result'''

    savepath = Path(f"{ctx.obj['CONFIG']['locations']['raw_data'] if savepath is None else savepath}") / 'comments'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    savepath = savepath / (Path(query_result).stem + '_comments.json')
    if os.path.isfile(savepath) and not overwrite:
        raise FileExistsError(f'Outfile already exists at {savepath} and overwrite flag is False. Ending.')

    utils.comment_retrieval(
        query_result,
        savepath,
        token = ctx.obj['CONFIG']['API']['token']
    )

def common_options(function):
    function = click.option('--metadata', default=None, help='Directory where video metadata is kept. Alternatively, this can point to a specific query json result whose videos need to be processed.')(function)
    function = click.option('--video_dir', default=None, help='Directory where video files are saved')(function)
    function = click.option('--video_feat_dir', default=None, help='Where to save features')(function)
    function = click.option('--output', '-o', default=None, help='Filename of output file')(function)
    function = click.option('--overwrite', type=bool, default=False, is_flag=True)(function)
    return function

@cli.command()
@click.pass_context
@common_options
@click.option('--reextract', '-r', is_flag=True, default=False, help='whether to reextract features.')
def extract(ctx, metadata, video_dir, video_feat_dir, output, overwrite, reextract):
    '''Extract features from downloaded videos'''

    extractor = utils.HaystacksFeatureExtractor(
        f"{ctx.obj['CONFIG']['locations']['raw_data'] if metadata is None else metadata}",
        f"{Path(ctx.obj['CONFIG']['locations']['raw_data']) / 'videos' if video_dir is None else video_dir}",
        f"{Path(ctx.obj['CONFIG']['locations']['raw_data']) / 'video_features' if video_feat_dir is None else video_feat_dir}",
        device = ctx.obj['DEVICE']
    )

    extractor.extract_features(overwrite=reextract)

    if output is None and metadata is not None:
        output = Path(ctx.obj['CONFIG']['locations']['features']) / f"{Path(metadata).stem}_feat.csv"
    elif output is None and metadata is None:
        output = Path(ctx.obj['CONFIG']['locations']['features']) / f"all_feat.csv"

    extractor.save_to(output, overwrite)
    logger.info(f'Consolidated csv saved to {output}')

@cli.command()
@click.pass_context
@click.argument('infile')
def explode(ctx, infile):
    utils.sent_explode_func(infile)

@cli.command()
@click.pass_context
@click.option('--metadata', '-me', default=None, help='Directory where video metadata is kept. Alternatively, this can point to a specific query json result whose videos need to be processed.')
@click.option('--features_file', '-f', help='features csv to read in. Can be omitted if metadata option is provided.')
@click.option('--model', '-m', 
              default='Nithiwat/mdeberta-v3-base_claimbuster',
              help='Name of HuggingFace model to load for claim detection.'
              )
@click.option('--output', '-o', default=None, help='Filename of output file')
@click.option('--config', '-c',
              default='concatenated',
              help='Configuration for which features to use in detection.',
              type=click.Choice(['concatenated']))
@click.option('--overwrite', type=bool, default=False, is_flag=True)
@click.option('--intype', '-t', type=click.Choice(['p', # parliamentary debates
                                                   'q', # oral/written questions
                                                   't', # titktok videos
                                                   ]), default = 't')
def detect(ctx,
           metadata,
           features_file,
           model,
           output,
           config,
           overwrite,
           intype
           ):
    '''Detect claims from extracted features'''

    if intype == 't':
        # check that one of metadata or features_file is provided
        if metadata is None and features_file is None:
            raise ValueError('At least one of --features_file or --metadata must be provided')
        elif metadata and features_file is None:
            # attempt to find based on metadata file name if no features_file found
            features_file = Path(ctx.obj['CONFIG']['locations']['features']) / f"{Path(metadata).stem}_feat.csv"
            assert os.path.isfile(features_file), f"Attempted to find features file automatically based on metadata. {features_file} is not a valid file path."
    elif intype == 'p':
        pass

    # determine output path
    prefix_dict = {
        't': 'all',
        'p': 'parlspeech',
        'q': 'parlquest'
    }
    if output is None and metadata is not None:
        # if metadata is provided, it must be tiktok
        output = Path(ctx.obj['CONFIG']['locations']['processed_loc']) / f"{Path(metadata).stem}_detected.csv"
    elif output is None and metadata is None:
        output = Path(ctx.obj['CONFIG']['locations']['processed_loc']) / f"{prefix_dict[intype]}_detected.csv"
    logger.info(f'Output file set to {output}')

    # if output is existing and overwrite is not set, do nothing.
    if os.path.isfile(output) and not overwrite:
        logger.info(f'{output} file already exists and overwrite flag is not set. Ending.')
        return None

    # read in file
    infile = pd.read_csv(features_file)

    # do detection
    detector = main.ClaimDetector.from_transformers(model=model, device=ctx.obj['DEVICE'])
    claims = detector(infile['sentence'].to_list())

    # collect dataframe of results
    scores = pd.DataFrame.from_records([{score_item['label']:score_item['score'] for score_item in res} for res in claims])

    # combine and save
    result = pd.concat([infile, scores], axis=1)

    # if TikTok, add URL row
    if intype == 't':
        result['url'] = result['id'].apply(lambda some_id: f'https://tiktok.com/@tiktok/video/{some_id}')

    result.to_csv(output)


if __name__ == '__main__':
    cli()


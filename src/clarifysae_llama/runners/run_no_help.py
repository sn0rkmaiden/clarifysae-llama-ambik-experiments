from clarifysae_llama.runners.run_eval import parse_args, run_eval
from clarifysae_llama.config import load_yaml

if __name__ == '__main__':
    args = parse_args()
    run_eval(load_yaml(args.config))

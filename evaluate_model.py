import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

import models

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate language models using minimal pairs or prompting."
    )
    parser.add_argument("-i", "--input", type=str, 
                        default="stimuli/stimuli.csv",
                        help="Path to CSV file containing stimuli")
    parser.add_argument("-o", "--output", type=Path, default="data/model_outputs",
                        help="Path to output directory where output files will be written")
    parser.add_argument("--model", type=str, default="text-davinci-002")
    parser.add_argument("--model_type", type=str, default="openai", choices=["hf", "openai"])
    parser.add_argument("--eval_mode", type=str, choices=["minimal_pairs", "prompting"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Read data.
    print(f"Reading stimuli from {args.input}")
    df = pd.read_csv(args.input)

    # Get name of output folder.
    output_folder = Path(args.output, args.eval_mode)

    # Initialize model.
    print(f"Initializing model ({args.model}, type={args.model_type})")
    if args.model_type == "openai":
        with open("openai_key.txt", "r") as f:
            openai_key, openai_org = [l.strip() for l in f.readlines()]
    else:
        openai_key, openai_org = None, None
    m = models.LM(
        args.model, 
        args.model_type, 
        openai_key=openai_key, 
        openai_org=openai_org
    )
    # Get a model name that's safe for naming files (i.e., no "/").
    safe_model_name = args.model.split("/")[-1].lower()

    # Evaluate model using minimal pairs or prompting.
    if args.eval_mode == "minimal_pairs":
        all_token_surprisals = []
        meta_vars = ["phenomenon", "test_item", "original_condition"]
        for i, row in tqdm(df.iterrows(), total=len(df.index)):
            # Evaluate on grammatical and ungrammatical versions of each minimal pair.
            for condition in ["grammatical", "ungrammatical"]:
                sentence = row[f"sentence_{condition}"]
                if pd.isna(sentence):
                    df.loc[i, f"sum_surprisal_{condition}"] = None
                    continue
                sum_surprisal, token_surprisals = m.sentence_surprisal(sentence)
                df.loc[i, f"sum_surprisal_{condition}"] = sum_surprisal
                # Add meta information to token surprisals.
                token_surprisals["model"] = args.model
                token_surprisals["condition"] = condition
                for v in meta_vars:
                    token_surprisals[v] = row[v]
                all_token_surprisals.append(token_surprisals)
        # Save files.
        df.to_csv(Path(output_folder, "sentence_surprisals", f"{safe_model_name}.csv"), index=False)
        all_token_surprisals = pd.concat(all_token_surprisals)
        all_token_surprisals.to_csv(Path(output_folder, "token_surprisals", f"{safe_model_name}.csv"), index=False)
    
    else:
        # Read prompt.
        prompt = "Is the following sentence grammatically correct in English? [SENTENCE] Respond with C if it is correct, and N if it is not correct."

        # Evaluate model.
        meta_vars = ["phenomenon", "test_item", "condition"]
        for i, row in tqdm(df.iterrows(), total=len(df.index)):
            # Just evaluate on sentences from original Dentella et al. experiments.
            sentence = row.sentence
            full_prompt = prompt.replace(
                "[SENTENCE]", '"' + sentence + "." + '"'
            )
            response = m.generate(prompt)
            df.loc[i, "prompt"] = full_prompt
            df.loc[i, "response"] = response
        # Save files.
        output_folder = Path(args.output, "original")
        df.to_csv(Path(output_folder, f"{safe_model_name}.csv"), index=False)

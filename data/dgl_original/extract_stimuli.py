import pandas as pd

def sort_index(l):
    # l is expected to be a list of tuples (or pandas MultiIndex):
    # (phenomenon, test_item, condition)

    # First, sort by test_item. We need to be clever so that 2 comes after 1, not 10.
    s = sorted(l, key=lambda tup: int(tup[1].split(" ")[-1]))
    # Now, sort by phenomenon.
    s = sorted(s, key=lambda tup: tup[0])
    return s

if __name__ == "__main__":
    # Read data and get list of unique item identifiers.
    df = pd.read_csv("LM_grammaticality_data.csv").set_index(
        ["phenomenon", "test_item", "condition"]
    )
    unique_items = sort_index(df.index.unique())

    # Construct list of stimuli.
    stimuli = []
    for phenomenon, test_item, condition in unique_items:
        rows = df.loc[phenomenon, test_item, condition]
        sentences = rows.sentence.tolist()
        assert len(set(sentences)) == 1
        sentence = sentences[0]
        stimuli.append(dict(
            phenomenon=phenomenon,
            test_item=test_item,
            condition=condition,
            sentence=sentence
        ))

    # Save to CSV file.
    stimuli = pd.DataFrame(stimuli)
    stimuli.to_csv("../../stimuli/stimuli.csv", index=False)
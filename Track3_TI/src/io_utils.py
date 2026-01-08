import pandas as pd


def load_csv(
    path: str,
    conversation_column: str = None,
    topic_column: str = None,
    translation_column: str = None,
):
    """
    Load CSV and normalize columns.

    - conversation_column: input text (ASR / dialogue)
    - topic_column: ground-truth topics (comma-separated)
    - translation_column: optional translated text column
    """

    df = pd.read_csv(path)

    if conversation_column is not None and conversation_column in df.columns:
        df[conversation_column] = (
            df[conversation_column]
            .astype(str)
            .str.strip()
        )


    if translation_column is not None:
        if translation_column not in df.columns:
            df[translation_column] = ""
        else:
            df[translation_column] = (
                df[translation_column]
                .fillna("")
                .astype(str)
                .str.strip()
            )


    if topic_column is not None and topic_column in df.columns:
        df[topic_column] = (
            df[topic_column]
            .astype(str)
            .str.lower()
            .str.replace("&", ",", regex=False)
            .str.split(",")
        )

        df[topic_column] = df[topic_column].apply(
            lambda topics: [t.strip() for t in topics if t.strip()]
        )

    return df


def save_csv(df: pd.DataFrame, path: str):
    
    df.to_csv(path, index=False)

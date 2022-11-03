from lda_over_time.lda_over_time import LdaOverTime
from lda_over_time.models.lda_seq_model import LdaSeqModel

import pandas as pd

def convertScore(string: str) -> float:
    return float(string.replace(',', ''))

prefeiturasConverters = {
    'Overperforming Score (weighted  —  Likes 1x Shares 1x Comments 1x Love 1x Wow 1x Haha 1x Sad 1x Angry 1x Care 1x )':
        convertScore,
    'Sponsor Name': str,
    'Sponsor Category': str,
}

def main():
    # load prefeituras.csv
    df = pd.read_csv('~/Projects/pfg-tutorial/data/prefeituras.csv',
                     converters=prefeiturasConverters)

    df = df[['Message', 'Post Created Date']]
    df = df.dropna()
    df = df.sample(1000)

    dates = df['Post Created Date'].values
    msgs = df['Message'].values

    model = LdaSeqModel(msgs, dates, '%Y-%m-%d', '1M', 20, None, None)

    dtm = LdaOverTime(model)

    dtm.plot('')

    dtm.showvis(1)

if __name__ == '__main__':
    main()


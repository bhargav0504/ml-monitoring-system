import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    reference = pd.read_csv(os.path.join(BASE_DIR, 'reference_data.csv'))
    # create a shifted distribution to simulate drift
    noise = np.random.normal(0, 0.4, size=reference.shape)
    production = reference.copy()
    numeric_cols = production.select_dtypes(include=['float64', 'int64']).columns
    production[numeric_cols] = production[numeric_cols] + noise[:, :len(numeric_cols)]

    production.to_csv(os.path.join(BASE_DIR, 'production_data.csv'), index=False)
    print('Generated production_data.csv')


if __name__ == '__main__':
    main()

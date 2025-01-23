# make train and test combinations
from collections import defaultdict
from hamison_datasets.UPV.load_upv import load_upv_dataset

upv = load_upv_dataset(include_html=False, extract_text=False)

campañas = upv['¿Qué campaña?'].unique().tolist()

train_test_combos = defaultdict(list)

for i in range(1, len(campañas)):
    cs_train, cs_test = campañas[:i], campañas[i:]
    # single train item
    train_test_combos[cs_train[-1]] = cs_test
    if len(cs_train) > 1:
        cs_train = tuple(cs_train)
        train_test_combos[cs_train] = cs_test

train_test_combos_all = defaultdict(list)

for i in range(1, len(campañas)):
    cs_train, cs_test = campañas[:i], campañas[i:]
    train_test_combos_all[cs_train[-1]] = cs_test
    if len(cs_train) > 1:
        for j in range(len(cs_train)-1):
            new_train_cs = tuple(cs_train[j:])
            if len(new_train_cs) > 1:
                train_test_combos_all[new_train_cs] = cs_test


if __name__ == "__main__":

    for cs_train, cs_test in train_test_combos.items():
        print('train:', cs_train, end=' – ')

        if isinstance(cs_train, str):
            cs_train = [cs_train]
        print(upv[upv['¿Qué campaña?'].isin(cs_train)]
              ['¿Qué campaña?'].value_counts().to_list(), 'items')

        print('test: ', cs_test)
        print()

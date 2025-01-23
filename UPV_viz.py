import marimo

__generated_with = "0.9.27"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        ### To do:

        """
    )
    return


@app.cell
def __():
    # dataset interface should be {'train', 'dev', 'test'?, 'X_column', 'label_column'}
    import pandas as pd
    from hamison_datasets.UPV.load_upv import load_upv_dataset

    upv = load_upv_dataset()
    # upv_notmapped = load_upv_dataset(map_values=False)

    ds = upv[upv['text'].notnull()]
    campañas = upv['¿Qué campaña?'].unique().tolist()
    campañas_order = {v: k for k, v in enumerate(campañas)}

    upv_features = ['1. Espacio',
                    '2. Texto', '3. Multimedia', '4. Elemento desinformador',
                    '5.  Tipo multimedia', '6. Alteración multimedia',
                    '7. Cuenta emisora', '8. Fuente',
                    '9. Protagonista', '10. Atributo',
                    '11. Macro tema', '12. Populismo',
                    '13. Ataque', '14. Tipo de ataque']
    upv_features = [c for c in upv.columns if '.' in c]
    # for feat in upv_features:
    #     print(upv[feat].value_counts(), '\nTotal:',
    #           upv[(upv['text'].notnull() & upv[feat].notnull())].shape[0], '\n')
    upv.columns
    return (
        campañas,
        campañas_order,
        ds,
        load_upv_dataset,
        pd,
        upv,
        upv_features,
    )


@app.cell
def __(pd, upv):
    # load results GPT
    from sklearn.metrics import f1_score


    preds_gpt_9 = pd.read_json(
        'results/results_gpt-4o-mini_9.json'
    ).set_index('ID_Medio_Crono').rename(
        columns={'prediction': 'prediction_9', 'reasoning': 'reasoning_9'}
    )

    preds_gpt_1112 = pd.read_json(
        'results_gpt-4o-mini_11-12.json'
    ).set_index('ID_Medio_Crono').rename(
        columns={'theme': 'prediction_11', 'theme_reasoning': 'reasoning_11',
                 'populist_strategy': 'prediction_12', 'populist_strategy_reasoning': 'reasoning_12', }
    )

    preds_gpt_1314 = pd.read_json(
        'results_gpt-4o-mini_13-14.json'
    ).set_index('ID_Medio_Crono').rename(
        columns={'attack': 'prediction_13', 'attack_reasoning': 'reasoning_13',
                 'type_of_attack': 'prediction_14', 'type_of_attack_reasoning': 'reasoning_14', }
    )

    gpt = preds_gpt_9.join(preds_gpt_1112).join(preds_gpt_1314)

    gpt = upv[[
        '¿Qué campaña?',
        '9. Protagonista', '10. Atributo',
        '11. Macro tema', '12. Populismo',
        '13. Ataque', '14. Tipo de ataque']].copy().join(gpt)
    # label_preds = label_preds[label_preds['prediction'].notna()]


    gpt
    return f1_score, gpt, preds_gpt_1112, preds_gpt_1314, preds_gpt_9


@app.cell
def __(f1_score, feat, g, gpt, pd):
    predicted_vars = [col.split('_')[-1] for col in gpt.columns if col.startswith('prediction')]
    resultsGPT = []
    for v in predicted_vars:
        _feat = next((col for col in gpt if col.startswith(v)))
        pred_col = f'prediction_{v}'
        for _campaña, _g in gpt.groupby('¿Qué campaña?'):
            g_ = g[g[feat].notna() & g[pred_col].notna()]
            _f1 = f1_score(y_true=g_[feat], y_pred=g_[pred_col], average='macro')
            resultsGPT.append({'feat': _feat, 'test': _campaña, 'f1': _f1, 'clf': 'gpt-4o-mini'})
    resultsGPT = pd.DataFrame(resultsGPT)
    return g_, pred_col, predicted_vars, resultsGPT, v


@app.cell
def __(display, f1_score, g, pd, preds_gpt_wlabels):
    resultsGPT_1 = []
    for _campaña, _g in preds_gpt_wlabels.groupby('¿Qué campaña?'):
        display(_campaña, _g['prediction_9'].isna().value_counts(dropna=False))
        _g = g[g['prediction_9'].notna()]
        _f1 = f1_score(y_true=g['9. Protagonista'], y_pred=g['prediction_9'], average='macro')
        resultsGPT_1.append({'feat': '9. Protagonista', 'clf': 'gpt-4o-mini', 'test': _campaña, 'f1': _f1})
    for _campaña, _g in preds_gpt_wlabels.groupby('¿Qué campaña?'):
        display(_campaña, _g['prediction_11'].isna().value_counts(dropna=False))
        _g = g[g['prediction_11'].notna()]
        _f1 = f1_score(y_true=g['11. Macro tema'], y_pred=g['prediction_11'], average='macro')
        resultsGPT_1.append({'feat': '11. Macro tema', 'clf': 'gpt-4o-mini', 'test': _campaña, 'f1': _f1})
    for _campaña, _g in preds_gpt_wlabels.groupby('¿Qué campaña?'):
        display(_campaña, _g['prediction_11'].isna().value_counts(dropna=False))
        _g = g[g['prediction_12'].notna()]
        _f1 = f1_score(y_true=g['12. Populismo'], y_pred=g['prediction_12'], average='macro')
        resultsGPT_1.append({'feat': '12. Populismo', 'clf': 'gpt-4o-mini', 'test': _campaña, 'f1': _f1})
    for _campaña, _g in preds_gpt_wlabels.groupby('¿Qué campaña?'):
        display(_campaña, _g['prediction_11'].isna().value_counts(dropna=False))
        _g = g[g['prediction_13'].notna()]
        _f1 = f1_score(y_true=g['13. Ataque'], y_pred=g['prediction_13'], average='macro')
        resultsGPT_1.append({'feat': '13. Ataque', 'clf': 'gpt-4o-mini', 'test': _campaña, 'f1': _f1})
    for _campaña, _g in preds_gpt_wlabels.groupby('¿Qué campaña?'):
        display(_campaña, _g['prediction_11'].isna().value_counts(dropna=False))
        _g = g[g['prediction_14'].notna()]
        _f1 = f1_score(y_true=g['14. Tipo de ataque'], y_pred=g['prediction_14'], average='macro')
        resultsGPT_1.append({'feat': '14. Tipo de ataque', 'clf': 'gpt-4o-mini', 'test': _campaña, 'f1': _f1})
    resultsGPT_1 = pd.DataFrame(resultsGPT_1)
    return (resultsGPT_1,)


@app.cell
def __(display, f1_score, feat, g, pd, upv, upv_features):
    baselines = []
    for _feat in upv_features:
        df = upv[['¿Qué campaña?', feat]][upv[feat].notna()].copy()
        for _camp, _g in df.groupby('¿Qué campaña?'):
            try:
                g['mode'] = g[feat].mode()[0]
                baseline_f1 = f1_score(y_true=g[feat], y_pred=g['mode'], average='macro')
                baselines.append({'feat': _feat, 'test': _camp, 'f1': baseline_f1, 'clf': 'baseline'})
            except Exception as e:
                print(_feat, _camp)
                debug = g[feat].mode()
                display(_g)
    baselines = pd.DataFrame(baselines)
    return baseline_f1, baselines, debug, df


@app.cell
def __(baselines, campañas_order, pd, resultsGPT_1):
    def list_to_tuple(ob):
        if isinstance(ob, list):
            return tuple(ob)
        else:
            return ob
    resultsRF = pd.read_json('results/results_RF.json')
    resultsRF['clf'] = 'RandomForest'
    resultsRF_CV = pd.read_json('results/results_RF_CV.json')
    resultsRF_CV['clf'] = 'RandomForest_CV'
    resultsBETO_1e4 = pd.read_json('results/9/results_BETO_1e4.json')
    resultsBETO_1e4['clf'] = 'BETO'
    resultsBETO_1114 = pd.read_json('results/results_BETO_11-14.json')
    resultsBETO_1114['clf'] = 'BETO'
    resultsBETO = resultsBETO_1114
    resultsBETO_CV = pd.read_json('results/results_BETO_CV.json')
    resultsBETO_CV['clf'] = 'BETO_CV'
    results = pd.concat([baselines, resultsRF, resultsBETO, resultsRF_CV, resultsBETO_CV, resultsGPT_1]).reset_index(drop=True)
    results['train'] = results['train'].apply(list_to_tuple)
    results = results.sort_values(by=['test'], key=lambda x: x.map(campañas_order))
    results_multiple_traincs = results[results.train.apply(type) == tuple]
    results_single_traincs = results[results.train.apply(type) != tuple]
    results.loc[results.clf.isin(['baseline', 'BETO_CV', 'RandomForest_CV', 'gpt-4o-mini']), 'train'] = ''
    return (
        list_to_tuple,
        results,
        resultsBETO,
        resultsBETO_1114,
        resultsBETO_1e4,
        resultsBETO_CV,
        resultsRF,
        resultsRF_CV,
        results_multiple_traincs,
        results_single_traincs,
    )


@app.cell
def __(campañas_order, display, feat, g, results):
    import altair as alt

    def join_elements(ob):
        if isinstance(ob, list) or isinstance(ob, tuple):
            return '+'.join(ob)
        else:
            return ob
    for _feat, _g in results.groupby('feat', sort=True):
        if _feat not in ['11. Macro tema', '12. Populismo', '13. Ataque', '14. Tipo de ataque']:
            continue
        byfeat = g.sort_values('f1', ascending=False)[['feat', 'clf', 'train', 'test', 'f1']]
        byfeat['train'] = byfeat['train'].apply(join_elements)
        selection = alt.selection_point(fields=['clf'], bind='legend')
        hover = alt.selection_point(name='highlight', on='pointerover', empty=False)
        when_hover = alt.when(hover)
        stroke_width = alt.when(hover).then(alt.value(1)).otherwise(alt.value(0))
        stroke_color = alt.when(hover).then(alt.ColorValue('black')).otherwise(alt.ColorValue('white'))
        color = alt.condition(selection, alt.Color('clf:N', title='Estimator').scale(range=['red', 'orange', 'blue', 'seagreen', 'coral', '#7D3C98']), alt.value('lightgray'))
        points = alt.Chart(byfeat, width=420, title=feat).mark_point(filled=True, size=50, opacity=0.6, stroke='black').encode(y='f1', tooltip='train', color=color, order='clf', strokeWidth=stroke_width).encode(alt.X('test', sort=list(campañas_order.keys()), title='Campaña test'), alt.Y('f1')).add_params(selection, hover)
        display(points)
    return (
        alt,
        byfeat,
        color,
        hover,
        join_elements,
        points,
        selection,
        stroke_color,
        stroke_width,
        when_hover,
    )


@app.cell
def __(camp, g, g_vals, palette, upv):
    import seaborn as sns
    import matplotlib.pyplot as plt
    _feat_groups = [['9. Protagonista reduced', '10. Atributo']]
    _palette = sns.color_palette('blend:#fff,#7AB', as_cmap=True)
    for _g in _feat_groups:
        for _camp, h in upv.groupby('¿Qué campaña?'):
            _g_vals = h[g].value_counts(sort=False).unstack().fillna(0).astype(int)
            _g_plot = sns.heatmap(g_vals, cmap=palette, annot=True, fmt='0', cbar=False)
            _g_plot.set(title=_camp)
            plt.show()
            _f_name = '-'.join([c.split('.')[0] for c in g]) + f'_{camp}'
            _g_plot.figure.savefig(f'2dplots/{_f_name}.png')
    return h, plt, sns


@app.cell
def __(display, feat, palette, plt, sns, upv):
    upv_features_1 = ['9. Protagonista', '9. Protagonista reduced', '10. Atributo', '11. Macro tema', '12. Populismo', '13. Ataque', '14. Tipo de ataque', '14. Tipo de ataque reduced']
    for _feat in upv_features_1:
        grouped = upv.groupby('¿Qué campaña?', sort=False)[feat]
        counts = grouped.value_counts(normalize=False).unstack().fillna(0)
        norm_counts = grouped.value_counts(normalize=True).unstack().fillna(0)
        display(counts)
        _palette = sns.color_palette('hls', len(counts.columns))
        lineplot = sns.lineplot(norm_counts, palette=palette)
        if len(counts.columns) > 2:
            sns.move_legend(lineplot, 'upper left', bbox_to_anchor=(1, 1))
        lines_f_name = feat.split('.')[0] + '_lines'
        lineplot.figure.savefig(f'1dplots/{lines_f_name}.png')
        plt.show()
        plt.clf()
        stacked = counts.plot.bar(stacked=True, color=palette)
        if len(counts.columns) > 2:
            sns.move_legend(stacked, 'upper left', bbox_to_anchor=(1, 1))
        bars_f_name = feat.split('.')[0] + '_bars'
        stacked.figure.savefig(f'1dplots/{bars_f_name}.png')
        plt.show()
        plt.clf()
    return (
        bars_f_name,
        counts,
        grouped,
        lineplot,
        lines_f_name,
        norm_counts,
        stacked,
        upv_features_1,
    )


@app.cell
def __(g, g_vals, palette, plt, sns, upv):
    _feat_groups = [['2. Texto', '3. Multimedia', '4. Elemento desinformador'], ['5.  Tipo multimedia', '6. Alteración multimedia'], ['5.  Tipo multimedia reduced', '6. Alteración multimedia'], ['7. Cuenta emisora', '8. Fuente'], ['7. Cuenta emisora reduced', '8. Fuente'], ['9. Protagonista', '10. Atributo'], ['9. Protagonista reduced', '10. Atributo'], ['11. Macro tema', '12. Populismo']]
    _palette = sns.color_palette('blend:#fff,#7AB', as_cmap=True)
    for _g in _feat_groups:
        _g_vals = upv[g].value_counts(sort=False).unstack().fillna(0).astype(int)
        _g_plot = sns.heatmap(g_vals, cmap=palette, annot=True, fmt='0', cbar=False)
        plt.show()
        _f_name = '-'.join([c.split('.')[0] for c in g])
        _g_plot.figure.savefig(f'2dplots/{_f_name}.png')
    return


@app.cell
def __(upv):
    # spotted an anotation error
    upv[(upv['2. Texto'] == 'Sí') & (upv['3. Multimedia'] == 'No') &
        (upv['4. Elemento desinformador'] == 'Elemento multimedia')]['URL'].values
    return


@app.cell
def __(upv):
    upv['text'].str.split().str.len().describe()
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()


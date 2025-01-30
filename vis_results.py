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
    # load results
    import pandas as pd

    campañas_order = {'Gen19': 0, 'Gal-Eus20': 1,
                      'Cat21': 2, 'Mad21': 3, 'CyL22': 4, 'And22': 5}


    def list_to_tuple(ob):
        if isinstance(ob, list):
            return tuple(ob)
        else:
            return ob


    baselines = pd.read_json('results/baselines.json')

    resultsGPT = pd.read_json('results/results_GPT.json')

    resultsRF = pd.read_json('results/results_RF.json')
    resultsRF['clf'] = 'RandomForest'

    resultsRF_CV = pd.read_json('results/results_RF_CV.json')
    resultsRF_CV['clf'] = 'RandomForest_CV'

    resultsSVM = pd.read_json('results/results_SVM.json')
    resultsSVM['clf'] = 'SVM'

    resultsSVM_CV = pd.read_json('results/results_SVM_CV.json')
    resultsSVM_CV['clf'] = 'SVM_CV'

    # resultsBETO = pd.read_json('results_BETO.json')
    # resultsBETO['clf'] = 'BETO'

    resultsBETO_1e4 = pd.read_json('results/9/results_BETO_1e4.json')
    resultsBETO_1e4['clf'] = 'BETO'

    resultsBETO_1114 = pd.read_json('results/results_BETO_11-14.json')
    resultsBETO_1114['clf'] = 'BETO'

    resultsBETO = pd.concat([resultsBETO_1e4, resultsBETO_1114])

    resultsBETO_CV = pd.read_json('results/results_BETO_CV.json')
    resultsBETO_CV['clf'] = 'BETO_CV'


    results = pd.concat([
                        baselines,
                        resultsRF_CV, resultsSVM_CV, resultsBETO_CV,
                        resultsRF, resultsSVM, resultsBETO,
                        resultsGPT,
                        ]).reset_index(drop=True)

    # sort
    results['train'] = results['train'].apply(list_to_tuple)
    results = results.sort_values(
        by=['test'], key=lambda x: x.map(campañas_order))
    # results['predictions'] = results['preds'].apply(
    #     lambda l: [p['prediction'] for p in l])
    results_multiple_traincs = results[results.train.apply(type) == tuple]
    results_single_traincs = results[results.train.apply(type) != tuple]
    return (
        baselines,
        campañas_order,
        list_to_tuple,
        pd,
        results,
        resultsBETO,
        resultsBETO_1114,
        resultsBETO_1e4,
        resultsBETO_CV,
        resultsGPT,
        resultsRF,
        resultsRF_CV,
        resultsSVM,
        resultsSVM_CV,
        results_multiple_traincs,
        results_single_traincs,
    )


@app.cell
def __(campañas_order, display, results):
    # group results by campaign and visualize groups
    import altair as alt
    from altair import datum

    color_range = ['coral', 'blue', 'red',
                   'orange', '#7D3C98', 'seagreen', 'coral']
    color_range_b = ['gray', 'blue', 'red',
                     'orange', '#7D3C98', 'seagreen', 'coral']

    results.loc[results.clf.isin(
        ['baseline', 'BETO_CV', 'RandomForest_CV', 'SVM_CV', 'gpt-4o-mini']), 'train'] = ''

    results['is_baseline'] = (results['clf'].str.contains('CV') |
                              results['clf'].str.contains('baseline'))

    data = results[
        ['feat', 'is_baseline', 'clf', 'train', 'test', 'f1']]
    # .sort_values('f1', ascending=False)


    displaying_feats = [
        '9. Protagonista', '10. Atributo',
        '9. Protagonista reduced',
        '11. Macro tema', '12. Populismo',
        '13. Ataque', '14. Tipo de ataque',
        # '14. Tipo de ataque reduced'
    ]


    def join_elements(ob):
        if isinstance(ob, list) or isinstance(ob, tuple):
            return '+'.join(ob)
        else:
            return ob


    data.loc[:, 'train'] = data['train'].apply(join_elements)


    selection = alt.selection_point(
        # bind=input_dropdown,
        fields=['clf'], bind='legend')
    when_sel = alt.when(selection)

    hover = alt.selection_point(
        name="highlight", on="pointerover", empty=False)
    when_hover = alt.when(hover)

    stroke_width = (
        alt.when(hover).then(alt.value(1)).otherwise(alt.value(0))
    )

    color_baselines = alt.Color('clf:N', title='Baseline',
                                sort=['BETO_CV', 'RandomForest_CV', 'SVM_CV', 'baseline']).scale(scheme='tableau10')

    color_estimators = alt.Color('clf:N', title='Estimator',
                                 sort=['BETO', 'RandomForest', 'SVM', 'gpt-4o-mini']).scale(scheme='tableau10')


    for feat, g in data.groupby('feat', sort=True):
        if feat not in displaying_feats:
            continue

        chart = alt.Chart(g, width=520, title=feat).encode(
            x=alt.X('test', sort=list(campañas_order.keys()), title='Campaña test'),
            y='f1:Q',
        )

        points = chart.transform_filter(datum.is_baseline == False).mark_point(
            size=50, opacity=0.8,  filled=True,
            strokeWidth=1,
        ).encode(
            xOffset=alt.XOffset('jitter:Q',
                                scale=alt.Scale(range=[60, 30])),
            tooltip='train',  # tooltip=['clf', 'train', 'test', 'f1']
            color=alt.condition(selection, color_estimators,
                                alt.value('lightgray')),
            stroke=alt.when(hover).then(
                alt.ColorValue("black")).otherwise(alt.ColorValue("white")),
            # order='clf',
            order=when_sel.then(alt.value(1)).otherwise(alt.value(0))
        ).add_params(selection, hover).transform_calculate(jitter="sqrt(-2*log(random()))*cos(2*PI*random())")

        ticks = chart.transform_filter(datum.is_baseline == True).mark_line(
            # size=200, shape='stroke',
            interpolate='step',
            opacity=0.5, strokeWidth=1,
        ).encode(
            stroke=alt.when(hover).then(
                alt.ColorValue("black")).otherwise(color_baselines),
            order='clf',
        )

        # text = points.mark_text(
        #     align='left',
        #     baseline='middle',
        #     dx=7,
        #     fontSize=9
        # ).encode(
        #     text='train',
        #     color=alt.value("black")
        # )

        display(ticks + points)  # + text)

        # for camp, h in byfeat.groupby('test', sort=False):
        #     display(h)

        # for camp in campañas_order:
        #     display(byfeat[byfeat.test == camp])
    return (
        alt,
        chart,
        color_baselines,
        color_estimators,
        color_range,
        color_range_b,
        data,
        datum,
        displaying_feats,
        feat,
        g,
        hover,
        join_elements,
        points,
        selection,
        stroke_width,
        ticks,
        when_hover,
        when_sel,
    )


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()


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
                        resultsRF, resultsBETO,
                        resultsRF_CV, resultsBETO_CV,
                        #  resultsBETO_1e4, resultsBETO_CV_1e4,
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

    results.loc[results.clf.isin(
        ['baseline', 'BETO_CV', 'RandomForest_CV', 'gpt-4o-mini']), 'train'] = ''
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
        results_multiple_traincs,
        results_single_traincs,
    )


@app.cell
def __(campañas_order, display, results):
    # group results by campaign and visualize groups
    import altair as alt


    def join_elements(ob):
        if isinstance(ob, list) or isinstance(ob, tuple):
            return '+'.join(ob)
        else:
            return ob


    for feat, g in results.groupby('feat', sort=True):
        if feat not in [
            # '9. Protagonista', '10. Atributo',
            #  '9. Protagonista reduced',
            '11. Macro tema', '12. Populismo',
            '13. Ataque', '14. Tipo de ataque',
            # '14. Tipo de ataque reduced'
        ]:
            continue

        byfeat = g.sort_values('f1', ascending=False)[
            ['feat', 'clf', 'train', 'test', 'f1']]

        byfeat['train'] = byfeat['train'].apply(join_elements)

        selection = alt.selection_point(
            fields=['clf'], bind='legend')  # bind=input_dropdown,

        hover = alt.selection_point(
            name="highlight", on="pointerover", empty=False)
        when_hover = alt.when(hover)
        stroke_width = (
            alt.when(hover).then(alt.value(1)).otherwise(alt.value(0))
        )
        stroke_color = (
            alt.when(hover).then(alt.ColorValue("black")
                                 ).otherwise(alt.ColorValue("white"))
        )

        color = alt.condition(
            selection,
            alt.Color('clf:N', title='Estimator').scale(range=[
                'red', 'orange', 'blue', 'seagreen', 'coral', '#7D3C98']),
            alt.value('lightgray'))

        # order = alt.condition(
        # selection, 1, 0)

        points = alt.Chart(byfeat, width=420, title=feat).mark_point(
            filled=True, size=50, opacity=0.6, stroke='black'
        ).encode(
            y='f1', tooltip='train',  # tooltip=['clf', 'train', 'test', 'f1']
            color=color, order='clf',
            strokeWidth=stroke_width
        ).encode(
            alt.X('test', sort=list(campañas_order.keys()), title='Campaña test'),
            alt.Y('f1')
        ).add_params(selection, hover)  # .transform_filter(selection)

        # text = points.mark_text(
        #     align='left',
        #     baseline='middle',
        #     dx=7,
        #     fontSize=9
        # ).encode(
        #     text='train',
        #     color=alt.value("black")
        # )

        display(points)  # + text)

        # for camp, h in byfeat.groupby('test', sort=False):
        #     display(h)

        # for camp in campañas_order:
        #     display(byfeat[byfeat.test == camp])
    return (
        alt,
        byfeat,
        color,
        feat,
        g,
        hover,
        join_elements,
        points,
        selection,
        stroke_color,
        stroke_width,
        when_hover,
    )


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()


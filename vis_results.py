import marimo

__generated_with = "0.10.19"
app = marimo.App(layout_file="layouts/vis_results.slides.json")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import altair as alt
    from altair import datum

    lang = 'ES'
    return alt, datum, lang, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Expert analysis of Political Disinformation: Can we automate it?""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## We have a dataset of 481 articles published by Spanish fact-checking media[^1] about disinformation in the political domain.

        ## The dataset comprises the period Nov. 2019 – June 2022, in which 6 elections took place: one national election (_Gen19_) followed by 5 regional elections (_Gal-Eus20, Cat21, Mad21, CyL22, And22_).


        [^1]: Maldita.es, Newtral, Efe Verifica, Verificat
        """
    )
    return


@app.cell
def _(lang, mo):


    mo.vstack([
        mo.md(
            '## The misinformations have been labeled by experts, through content analysis, along 14 dimensions in 5 groups:' 
            if lang=='EN' else 
            '## Los desmentidos han sido caracterizados por expertos, mediante análisis de contenido, a lo largo de 14 variables en 5 grupos:' ),

        mo.hstack([
            mo.vstack([
                mo.md('''
                #### A. Ámbito 
                1. Ámbito de difusión
                
                #### B. Formato 
                2. Presencia de texto
                3. Presencia de recurso multimedia
                4. Elemento desinformador 
                5. Tipo de recurso multimedia

                ---

                '''),
                mo.md(f'''
                _Intentamos predecir las variables en estos grupos_
                {mo.icon('lucide:arrow-right')}''')
            ], gap=5),
            
            mo.md('''
            #### C. Autoridad epistemológica  
            7. <mark>(Tipo de) Cuenta emisora
            8. <mark>(Tipo de) Fuente
            
            #### D. Protagonismo 
            9. <mark>(Tipo de) Protagonista 
            10. <mark>Atributo
            
            #### E. Temática y discurso
            11. <mark>Macro tema
            12. <mark>Estrategia populista
            13. <mark>(Presencia de) Ataque
            14. <mark>Tipo de ataque
            ''')
        ], justify='space-around')
    ], gap=3)
    return


@app.cell
def _():
    return


@app.cell
def _(alt):
    def stacked_bars(data, x, y, color):

        return alt.Chart(data).mark_bar().encode(
            x=x,
            y=y,
            color=color
        )


    return (stacked_bars,)


@app.cell
def _(mo):
    mo.md("""## RQ1: ¿Es posible predecir automáticamente las variables?""")
    return


@app.cell
def _():
    # load results
    import pandas as pd
    from numpy import nan
    from pathlib import Path

    paths = list(Path('results').glob('*.json'))

    results = []
    for p in paths:
        df = pd.read_json(p)
        if 'clf' not in df.columns:
            df['clf'] = p.stem
        results.append(df)

    results = pd.concat(results).reset_index()[
        ['feat', 'clf', 'train', 'test', 'accuracy', 'f1']]

    campañas_order = {'Gen19': 0, 'Gal-Eus20': 1,
                      'Cat21': 2, 'Mad21': 3, 'CyL22': 4, 'And22': 5}


    def list_to_tuple(ob):
        if isinstance(ob, list):
            return tuple(ob)
        elif isinstance(ob, str):
            return tuple([ob])
        else:
            return ob


    # prepare
    results.loc[results['clf'].str.contains('CV'),'train'] = results['test']
    #results.loc[results['clf'].str.contains('baseline'),'train'] = results['test']
    results['train'] = results['train'].apply(list_to_tuple)
    results = results.sort_values(
        by=['test'], key=lambda x: x.map(campañas_order))
    # results['predictions'] = results['preds'].apply(
    #     lambda l: [p['prediction'] for p in l])


    # results.loc[results.clf.isin(
        # ['baseline', 'BETO_CV', 'RandomForest_CV', 'SVM_CV', 'gpt-4o-mini']), 'train'] = ''

    results['is_baseline'] = (results['clf'].str.contains('CV') |
                              results['clf'].str.contains('baseline'))


    def join_elements(ob):
        if isinstance(ob, list) or isinstance(ob, tuple):
            return '+'.join(ob)
        else:
            return ob


    results['train_repr'] = results['train'].apply(join_elements)
    results['train_len'] = results['train'].str.len()
    return (
        Path,
        campañas_order,
        df,
        join_elements,
        list_to_tuple,
        nan,
        p,
        paths,
        pd,
        results,
    )


@app.cell
def _(campañas_order, nan, pd, results):
    def define_strategy(row):
        if 'gpt' in row.clf.lower():
            return 'zero shot'
        elif 'baseline' in row.clf.lower():
            return 'baseline'
        if pd.notna(row.train):
            if len(row.train) == 1:
                if row.train[0] == row.test:
                    return 'CV'
                elif (campañas_order[row.train[0]] ==
                      campañas_order[row.test]-1):
                    return 'previous one'
                else:
                    return 'another one'
            elif len(row.train) == campañas_order[row.test]:
                return 'all previous'
            else:
                return 'another combination'
        else:
            return nan

    results['training_strategy'] = results.apply(define_strategy, axis=1)

    results.sort_index()
    return (define_strategy,)


@app.cell
def _(results):
    results['training_strategy'].value_counts()
    return


@app.cell
def _(results):
    data = results
    return (data,)


@app.cell
def _(mo, results):
    mo.ui.data_explorer(results)
    return


@app.cell
def _(alt, campañas_order, datum):
    def my_chart(data, title, chartWidth=480):

        color_range = ['coral', 'blue', 'red',
                       'orange', '#7D3C98', 'seagreen', 'coral']
        color_range_b = ['gray', 'blue', 'red',
                         'orange', '#7D3C98', 'seagreen', 'coral']

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

        color_baselines = alt.Color(
            'clf:N', title='Baseline',
            sort=['BETO_CV', 'RandomForest_CV', 'SVM_CV', 'baseline']
        ).scale(scheme='tableau10')

        color_estimators = alt.Color(
            'clf:N', title='Estimator',
            sort=['BETO', 'RandomForest', 'SVM', 'gpt-4o-mini']
        ).scale(scheme='tableau10')

        conditional_color = alt.condition(
            selection, color_estimators, alt.value('lightgray'))



        chart = alt.Chart(
            data, width=chartWidth, title=title
        ).encode(
            x=alt.X(
                'test', 
                sort=list(campañas_order.keys()), 
                title='Campaña test',
                scale=alt.Scale(domain=campañas_order.keys())
            ),

            y=alt.Y(
                'f1:Q', 
                scale=alt.Scale(domain=(0,1)))
        )

        baseline = chart.transform_filter(
            datum.clf == 'baseline'
        ).mark_area(
            line={"color": "black", "opacity":0.3, "strokeWidth":1.5}, 
            interpolate='step',
            opacity=0.1, color='black',
        )

        points = chart.transform_filter(datum.is_baseline == False).mark_point(
            opacity=0.7,  
            strokeWidth=3,
            filled=True,
            size=50,
        ).encode(
            tooltip=['clf', 'train', 'test', 'f1'],
            order=when_sel.then(alt.value(1)).otherwise(alt.value(0)),

            xOffset=alt.XOffset('clf:N',
                scale=alt.Scale(range=(20,chartWidth/6-20)),
            ),

            color=conditional_color,
            shape='training_strategy:N',
        ).add_params(
            selection,
            hover
        ).transform_calculate(
            jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
        )

        ticks = chart.transform_filter(datum.clf == 'RandomForest_CV').mark_line(
            interpolate='step',
            opacity=0.5, strokeWidth=1,
        ).encode(
            stroke=alt.when(hover).then(
                alt.ColorValue("black")).otherwise(color_baselines),
            order='clf',
        )

        return baseline + points
    return (my_chart,)


@app.cell(hide_code=True)
def _(mo):
    feats = [
        '9. Protagonista', '10. Atributo',
        '9. Protagonista reduced',
        '11. Macro tema', '12. Populismo',
        '13. Ataque', '14. Tipo de ataque',
        # '14. Tipo de ataque reduced'
    ]
    display_feat = mo.ui.dropdown(
        options=feats, label="Variable:", value='11. Macro tema'
    )

    training_strategies = [
        'previous one', 'all previous', 'another one', 'another combination']
    display_strategy = mo.ui.multiselect(
        options=training_strategies, label="Training strategy:", value=['previous one']
    )

    clfs = ['BETO', 'RandomForest', 'SVM', 'gpt-4o-mini']
    display_clf = mo.ui.multiselect(
        options=clfs, label="Estimator:", value=['BETO', 'SVM'])


    mo.hstack([display_feat, display_strategy, display_clf])
    return (
        clfs,
        display_clf,
        display_feat,
        display_strategy,
        feats,
        training_strategies,
    )


@app.cell
def _(data, display_clf, display_feat, display_strategy, my_chart):
    data_sel = data[
        (data['feat']==display_feat.value)
        & (data['clf'].isin(display_clf.value))
        & (data['training_strategy'].isin(display_strategy.value))
    ]
    my_chart(data=data_sel, title=display_feat.value)
    return (data_sel,)


if __name__ == "__main__":
    app.run()

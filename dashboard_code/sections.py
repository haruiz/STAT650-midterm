import numpy as np
import pandas as pd
from dash import dcc, html, Output, Input
from plotly import express as px
from sklearn.metrics import r2_score

from dashboard_section import DashboardSection
from dataset import Dataset
import dash_bootstrap_components as dbc
import plotly.graph_objects as go  # or plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


class Q1Section(DashboardSection):
    def __init__(self, app, dataset: Dataset):
        super().__init__(
            app, dataset, "Question 1", "Who are the top emitters per continent?"
        )
        self._dropdown = dcc.Dropdown(
            id="continents-dropdown",
            value="North America",
            options=[
                {"label": continent, "value": continent}
                for continent in self.dataset.get_continents()
            ],
        )
        self._graph = dcc.Graph(id="continents-graph")

    def __mk_co2_emissions_by_continent_plot(self, continent):
        """
        make co2 emission plot
        :return:
        """
        data = self.dataset.get_co2_emissions_by_continent(continent, top=10)
        return px.line(
            title="", data_frame=data, x="year", y="co2", color="country", markers=True
        )

    def __mk_corr_heatmap_plot(self):
        corr = self.dataset.data.corr(method="pearson")
        fig = px.imshow(
            corr, text_auto=True, height=500
        )  # create corr features heatmap
        return dcc.Graph(figure=fig)

    def layout(self) -> html.Div:
        """
        Returns the layout for the Q1 section
        :return:
        """
        return html.Div(
            children=[
                dbc.Row(
                    [
                        dbc.Col([html.Div([self._dropdown, self._graph])]),
                        dbc.Col([html.Div([self.__mk_corr_heatmap_plot()])]),
                    ]
                )
            ]
        )

    def callbacks(self, app):
        @app.callback(
            Output("continents-graph", "figure"), Input("continents-dropdown", "value")
        )
        def value_changed(value):
            return self.__mk_co2_emissions_by_continent_plot(value)


class Q2Section(DashboardSection):
    def __init__(self, app, dataset: Dataset):
        super().__init__(
            app,
            dataset,
            "Question 2",
            "Which countries from each continent has seen the greatest increases and decreases in CO2 "
            "efficiency from 2000 to 2018?",
        )

    def __make_plot_countries_with_greatest_co2_reduction(self):
        data = (
            self.dataset.get_countries_data_with_greatest_reduction_in_co2_per_unit_energy()
        )
        fig = px.line(
            data,
            x="year",
            y="co2_per_unit_energy",
            color="country",
            title="Countries with greatest reduction in CO2 per unit energy",
        )
        return dcc.Graph(figure=fig)

    def __make_plot_countries_with_least_co2_reduction(self):
        data = (
            self.dataset.get_countries_data_with_least_reduction_in_co2_per_unit_energy()
        )
        fig = px.line(
            data,
            x="year",
            y="co2_per_unit_energy",
            color="country",
            title="Countries with least reduction in CO2 per unit energy",
        )
        return dcc.Graph(figure=fig)

    def layout(self) -> html.Div:
        """
        Returns the layout for the Q1 section
        :return:
        """
        graph1 = self.__make_plot_countries_with_greatest_co2_reduction()
        graph2 = self.__make_plot_countries_with_least_co2_reduction()
        return html.Div(
            children=[
                dbc.Row(
                    [
                        dbc.Col([html.Div([graph1])]),
                        dbc.Col([html.Div([graph2])]),
                    ]
                )
            ]
        )


def regression_plot(
        data: pd.DataFrame, x, y, hue=None, title=None, annotation_loc=None, labels=None
):
    x = data[x].to_numpy()
    y = data[y].to_numpy()
    r, p_val = pearsonr(x, y)

    model = LinearRegression()
    model.fit(x[:, np.newaxis], y[:, np.newaxis])
    coeff = model.coef_.round(3)
    print(coeff)

    fig = px.scatter(
        data,
        x=x,
        y=y,
        trendline="ols",
        title=title,
        color=hue,
        trendline_scope="overall",
        labels=labels,
    )
    if annotation_loc:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=annotation_loc[0],
            y=annotation_loc[1],
            text=f"R={r:0.2f}",
            arrowhead=2,
        )
    return dcc.Graph(figure=fig)


class Q3Section(DashboardSection):
    def __init__(self, app, dataset: Dataset):
        super().__init__(
            app,
            dataset,
            "Question 3",
            "What is the relationship between population and CO2 emission?",
        )

    def mk_regression_plot(self):
        data = self.dataset.data
        data_2018 = data[data.year == 2018]
        return regression_plot(
            data_2018,
            x="population",
            y="co2",
            hue="continent",
            title="Emissions vs. Population in 2018",
        )

    def mk_regression_plot_after_log(self):
        data = self.dataset.data
        data_2018_loglog = data[data.year == 2018].copy()
        data_2018_loglog["population"] = np.log(data_2018_loglog["population"])
        data_2018_loglog["co2"] = np.log(data_2018_loglog["co2"])
        return regression_plot(
            data_2018_loglog,
            x="population",
            y="co2",
            hue="continent",
            annotation_loc=(0.25, 0.4),
            title="Log Emissions vs. Log Population in 2018",
        )

    def layout(self) -> html.Div:
        graph1 = self.mk_regression_plot()
        graph2 = self.mk_regression_plot_after_log()
        return html.Div(
            children=[
                dbc.Row(
                    [
                        dbc.Col([html.Div([graph1])]),
                        dbc.Col([html.Div([graph2])]),
                    ]
                )
            ]
        )


class Q4Section(DashboardSection):
    def __init__(self, app, dataset: Dataset):
        super().__init__(
            app,
            dataset,
            "Question 4",
            "What is the relationship between GDP and CO2 emissions?",
        )

    def mk_regression_plot(self):
        data = self.dataset.data
        data_2018 = data[data.year == 2018]
        return regression_plot(
            data_2018,
            x="gdp",
            y="co2",
            hue="continent",
            title="CO2 Emissions vs. GDP in 2018",
        )

    def mk_regression_plot_after_log(self):
        data = self.dataset.data
        data_2018_loglog = data[data.year == 2018].copy()
        data_2018_loglog["gdp"] = np.log(data_2018_loglog["gdp"])
        data_2018_loglog["co2"] = np.log(data_2018_loglog["co2"])
        return regression_plot(
            data_2018_loglog,
            x="gdp",
            y="co2",
            hue="continent",
            annotation_loc=(0.25, 0.4),
            title="Log-Log CO2 Emissions vs. GDP in 2018",
        )

    def layout(self) -> html.Div:
        graph1 = self.mk_regression_plot()
        graph2 = self.mk_regression_plot_after_log()
        return html.Div(
            children=[
                dbc.Row([
                    html.P(children=[
                        "GDP: Gross domestic product measured in international-$ using 2011 prices to adjust for price "
                        "changes over time (inflation) and price differences between countries. Calculated by "
                        "multiplying GDP "
                        "per capita with population. "
                        ])]),
                dbc.Row(
                    [
                        dbc.Col([html.Div([graph1])]),
                        dbc.Col([html.Div([graph2])]),
                    ]
                )
            ]
        )

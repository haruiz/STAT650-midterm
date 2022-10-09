import pandas as pd


class Dataset:
    def __init__(self, data_file: str = "data_final.csv"):
        self.data = pd.read_csv(data_file)

    def get_data(self) -> pd.DataFrame:
        """
        Returns the data
        :return:
        """
        return self.data

    def get_continents(self) -> list:
        """
        Returns a list of continents
        :return:
        """
        return self.data["continent"].unique().tolist()

    def get_countries(self, continent: str) -> list:
        """
        Returns a list of countries for a given continent
        :param continent:
        :return:
        """
        return (
            self.data[self.data["continent"] == continent]["country"].unique().tolist()
        )

    def get_observations_by_continent(self, continent: str) -> pd.DataFrame:
        """
        Returns the observations for a given continent
        :param continent:
        :return:
        """
        return self.data[self.data["continent"] == continent]

    def get_observations_by_country(self, country: str) -> pd.DataFrame:
        """
        Returns the observations for a given country
        :param country:
        :return:
        """
        return self.data[self.data["country"] == country]

    def get_co2_emissions_by_continent(
        self, continent: str, top: int = 10
    ) -> pd.DataFrame:
        """
        Returns the co2 emissions for a given continent
        :param top: the number of top countries to return
        :param continent: the continent
        :return:
        """
        continent_data = self.get_observations_by_continent(continent)
        continent_data = continent_data.sort_values("co2", ascending=False)
        top_n_countries = continent_data["country"].unique().tolist()[:top]
        continent_data = continent_data[continent_data["country"].isin(top_n_countries)]
        continent_data = continent_data.sort_values(by="year")
        return continent_data

    def get_co2_per_unit_data_from_2000_to_2018(self):
        # get countries that have observations in 2000 or 2018

        data_2000 = self.data[self.data.year == 2000]
        data_2018 = self.data[self.data.year == 2018]
        # get list of countries that have observations in both years
        common_countries = list(set(data_2000.country) & set(data_2018.country))
        # stack dataframes
        data_2000_2018 = pd.concat(
            [
                data_2000[
                    [country in common_countries for country in data_2000.country]
                ],
                data_2018[
                    [country in common_countries for country in data_2018.country]
                ],
            ]
        ).sort_values(["country", "year"])
        # group with country and continent and get percent change
        pct_change_co2_per_unit_energy = (
            data_2000_2018.loc[:, ["country", "co2_per_unit_energy", "continent"]]
            .groupby(["continent", "country"], group_keys=True)
            .apply(pd.Series.pct_change)
            .dropna()
            .reset_index()
            .drop("level_2", axis=1)
        )
        return pct_change_co2_per_unit_energy

    def get_countries_data_with_greatest_reduction_in_co2_per_unit_energy(self):
        pct_change_co2_per_unit_energy = self.get_co2_per_unit_data_from_2000_to_2018()
        results = (
            pct_change_co2_per_unit_energy.sort_values("co2_per_unit_energy")
            .groupby(["continent"])
            .head(1)
        )
        # get countries with the greatest increase in co2 per unit energy
        list_of_countries = results["country"].to_list()
        return self.data[
            [country in list_of_countries for country in self.data.country.tolist()]
        ][self.data.year >= 2000]

    def get_countries_data_with_least_reduction_in_co2_per_unit_energy(self):
        # get data for given continent
        pct_change_co2_per_unit_energy = self.get_co2_per_unit_data_from_2000_to_2018()
        results = (
            pct_change_co2_per_unit_energy.sort_values(
                "co2_per_unit_energy", ascending=False
            )
            .groupby(["continent"])
            .head(1)
        )
        # get countries with the least increase in co2 per unit energy
        list_of_countries = results["country"].to_list()
        return self.data[
            [country in list_of_countries for country in self.data.country.tolist()]
        ][self.data.year >= 2000]

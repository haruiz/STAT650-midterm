from dashboard import Dashboard

if __name__ == "__main__":

    description = (
        "Planet earth is experiencing tremendous global climate change effects in forest fires, droughts, floods, "
        "and extreme weather conditions. These impacts are mainly associated with increased carbon dioxide, methane, "
        "and other greenhouse gasses in our atmosphere caused by human activities. According to the United States "
        "Environmental Protection Agency, the largest source of greenhouse gas (GHG) emission in the United States is "
        "energy usage, in terms of burning fossil fuels for electricity, heat, and transportation. The rise in "
        "energy-related CO2 emissions has pushed greenhouse gas emissions from energy to their highest level in 2018. "
        "This case study presents global trends in net CO2 emission across different continents over time. All the "
        "code could be find in the link below:"
    )

    Dashboard(
        data_file="data_final.csv",
        title=(
            "CO2 Emissions and Trends, Continent and Country Level, A Case of Study. \n Mid-Term Report STAT 650"
        ),
        description=description,
    ).run()

from data.data_load import test_data_raw, train_data_raw, valid_data_raw

train_data_raw.assign(
    de=lambda x: x.translation.apply(lambda y: y["de"]),
    en=lambda x: x.translation.apply(lambda y: y["en"]),
)

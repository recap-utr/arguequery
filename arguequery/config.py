import typed_settings as ts


@ts.settings(frozen=True)
class Config:
    address: str = "localhost:50200"
    nlp_address: str = "localhost:50100"

from pathlib import Path
from streamlit.testing.v1 import AppTest
from image_pipeline import connect, ingest_bytes
from test_pipeline import png
 
 
def test_empty_and_populated_app(tmp_path, monkeypatch):
    database = tmp_path / "catalog.sqlite3"
    monkeypatch.setenv("IMAGE_CATALOG_DB", str(database))
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    assert any("No images yet" in x.value for x in at.info)
    con = connect(database)
    ingest_bytes(con, png(), "sample.png", "upload:sample")
    ingest_bytes(con, b"bad", "corrupt.png", "upload:bad")
    con.close()
    at.run()
    assert not at.exception
    assert at.metric[0].value == "2"
    at.text_input(key="search").set_value("sample").run()
    assert not at.exception
    assert at.metric[0].value == "1"
    at.text_input(key="search").set_value("no-match").run()
    assert not at.exception
    assert at.metric[0].value == "0"
 
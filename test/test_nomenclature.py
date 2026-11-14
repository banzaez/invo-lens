from fastapi.testclient import TestClient

from invo_lens.main import app
from invo_lens.schemas import InvoiceItem, NomenclatureItem
from invo_lens.services.nomenclature_store import SQLiteNomenclatureStore


def test_nomenclature_store_upsert_list_match_and_delete(tmp_path):
    store = SQLiteNomenclatureStore(str(tmp_path / "test_nom.db"))
    count = store.upsert_items(
        "client-1",
        [
            NomenclatureItem(code="C-100", name="Кабель ВВГ 3х2.5", sku="VVG-325", barcode="4820001234567", unit="м"),
            NomenclatureItem(code="C-200", name="Автомат 16А", sku=None, unit="шт"),
        ],
    )
    assert count == 2
    assert len(store.list_items("client-1")) == 2
    assert store.list_items("client-2") == []

    items = [
        InvoiceItem(barcode="4820001234567", name="Кабель (другое название)"),
        InvoiceItem(sku="vvg-325", name="Кабель силовий"),
        InvoiceItem(sku=None, name="  автомат   16а "),
        InvoiceItem(sku="UNKNOWN", name="Неизвестный товар"),
    ]
    store.match_invoice_items("client-1", items)

    assert items[0].matched_code == "C-100"
    assert items[1].matched_code == "C-100"
    assert items[2].matched_code == "C-200"
    assert items[3].matched_code is None

    deleted = store.delete_client_items("client-1")
    assert deleted == 2
    assert store.list_items("client-1") == []


def test_nomenclature_api_endpoints(tmp_path, monkeypatch):
    test_store = SQLiteNomenclatureStore(str(tmp_path / "api_nom.db"))
    monkeypatch.setattr("invo_lens.main.nomenclature_store", test_store)

    client = TestClient(app)
    resp = client.post(
        "/api/v1/nomenclature",
        json={
            "client_id": "acme",
            "items": [
                {"code": "A1", "name": "Бумага А4", "sku": "PAP-A4", "unit": "пач"},
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"client_id": "acme", "upserted": 1}

    resp_list = client.get("/api/v1/nomenclature/acme")
    assert resp_list.status_code == 200
    assert len(resp_list.json()) == 1
    assert resp_list.json()[0]["code"] == "A1"

    resp_del = client.delete("/api/v1/nomenclature/acme")
    assert resp_del.status_code == 200
    assert resp_del.json() == {"client_id": "acme", "deleted": 1}


def test_fuzzy_and_number_guard(tmp_path):
    store = SQLiteNomenclatureStore(str(tmp_path / "fuzzy.db"))
    store.upsert_items(
        "c1",
        [
            NomenclatureItem(code="BOLT-6", name="Болт М6х20 оцинкованный"),
            NomenclatureItem(code="PAP-1", name="Бумага офисная А4 80г"),
        ],
    )
    items = [
        InvoiceItem(name="Болт М8х40 оцинкованный"),
        InvoiceItem(name="Бумага офисная А4 80г."),
    ]
    store.match_invoice_items("c1", items)
    assert items[0].matched_code is None
    assert items[1].matched_code == "PAP-1"


def test_local_embedding_match(tmp_path, monkeypatch):
    monkeypatch.setattr("invo_lens.services.nomenclature_store.settings.lm_studio_embedding_model", "local-emb")

    def fake_embed(text: str):
        t = text.lower()
        if "доставк" in t or "перевез" in t:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    monkeypatch.setattr("invo_lens.services.nomenclature_store._get_local_embedding", fake_embed)

    store = SQLiteNomenclatureStore(str(tmp_path / "emb.db"))
    store.upsert_items("c1", [NomenclatureItem(code="SRV-DEL", name="Доставка груза")])

    items = [InvoiceItem(name="Послуга з перевезення")]
    store.match_invoice_items("c1", items)
    assert items[0].matched_code == "SRV-DEL"

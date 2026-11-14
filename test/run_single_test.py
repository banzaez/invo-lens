import json
from pathlib import Path
import time

from fastapi.testclient import TestClient

from invo_lens.main import app


def main() -> None:
    sample_path = Path("test/example/image copy.png")
    results_dir = Path("test/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{sample_path.stem}.json"

    client = TestClient(app)

    with sample_path.open("rb") as f:
        resp = client.post(
            "/api/v1/jobs",
            files=[("files", (sample_path.name, f, "application/pdf"))],
        )

    assert resp.status_code == 202, f"Unexpected status: {resp.status_code} {resp.text}"
    job_id = resp.json()["job_id"]
    print(f"Created job: {job_id}")

    job_data = None
    for _ in range(60):
        get_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert get_resp.status_code == 200, get_resp.text
        job_data = get_resp.json()
        if job_data["status"] in ("completed", "failed"):
            break
        time.sleep(1)

    output_path.write_text(
        json.dumps(job_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved result to {output_path}")
    print(json.dumps(job_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

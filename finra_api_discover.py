"""
FINRA 공식 API 진단 스크립트 -- 본 작업(feature_engineering) 전에 먼저 실행

목적:
    1) OAuth 인증이 정상 동작하는지 확인
    2) Reg SHO Daily Short Sale Volume의 정확한 production dataset 이름을 찾음
       (문서에 명시적으로 안 나와있어서 /datasets 목록에서 동적으로 검색)
    3) 그 dataset의 실제 필드명(종목코드, 날짜, 공매도량 등)을 /metadata로 확인

환경변수 필요 (.env 파일에 넣고 python-dotenv로 로드):
    FINRA_CLIENT_ID
    FINRA_CLIENT_SECRET

사용법:
    python finra_api_discover.py
"""

import base64
import os

import requests
from dotenv import load_dotenv

load_dotenv()

FIP_TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
API_BASE_URL = "https://api.finra.org"
DATASET_GROUP = "otcMarket"


def get_access_token() -> str:
    client_id = os.environ["FINRA_CLIENT_ID"]
    client_secret = os.environ["FINRA_CLIENT_SECRET"]
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    response = requests.post(
        FIP_TOKEN_URL,
        headers={"Authorization": f"Basic {basic}"},
    )
    response.raise_for_status()
    token_data = response.json()
    print(f"토큰 발급 성공 (만료까지 {token_data['expires_in']}초)")
    return token_data["access_token"]


def list_datasets(token: str) -> list:
    response = requests.get(
        f"{API_BASE_URL}/datasets",
        params={"group": DATASET_GROUP},
        headers={"Authorization": f"Bearer {token}"},
    )
    response.raise_for_status()
    return response.json()["datasets"]


def get_metadata(token: str, dataset_name: str) -> dict:
    response = requests.get(
        f"{API_BASE_URL}/metadata/group/{DATASET_GROUP}/name/{dataset_name}",
        headers={"Authorization": f"Bearer {token}"},
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print("=== 1) 토큰 발급 ===")
    token = get_access_token()

    print("\n=== 2) otcMarket 그룹의 전체 dataset 목록 ===")
    datasets = list_datasets(token)
    for d in datasets:
        print(f"  {d['name']}  (status={d.get('status')})")

    print("\n=== 3) 'regsho' 포함된 production dataset 찾기 ===")
    candidates = [d["name"] for d in datasets if "regsho" in d["name"].lower() and "mock" not in d["name"].lower()]
    print(f"  후보: {candidates}")

    if not candidates:
        print("  ⚠️ 못 찾음. 위 전체 목록에서 이름 직접 확인해줘.")
    else:
        dataset_name = candidates[0]
        print(f"\n=== 4) '{dataset_name}' 메타데이터(필드 목록) ===")
        metadata = get_metadata(token, dataset_name)
        print(f"  partitionFields: {metadata.get('partitionFields')}")
        for field in metadata.get("fields", []):
            print(f"  {field['name']} ({field['type']}): {field.get('description', '')}")